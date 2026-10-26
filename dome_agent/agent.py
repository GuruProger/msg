"""Autonomous Dome agent loop.

One process owns the shift: connect, observe telemetry, read operator chat,
apply deterministic safety/production rules, execute verified actions, and
write a complete JSONL audit log.  DeepSeek is advisory only and is never on
the critical path for a safety action.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from dome_control.agent_api import DomeAgentAPI
from dome_control.client import (
    DomeAuthError,
    DomeConnectionError,
    DomeError,
    DomeSessionRevoked,
    DomeTimeout,
)

from .analysis import Snapshot, as_bool
from .chat import ChatMessage, new_operator_messages, parse_operator_task
from .config import AgentConfig
from .llm import Advice, DeepSeekAdvisor
from .logging import AgentLog
from .rules import NO_VALUE, ActionSpec, OutgoingMessage, RuleEngine, RuleResult
from .state import StateStore

CHAT_POLL_INTERVAL_S = 3.0
TELEMETRY_MAX_AGE_S = 5.0
LOOP_INTERVAL_S = 1.0
LLM_MESSAGE_MIN_INTERVAL_S = 600.0


def _safe_text(value: Any, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


def _alert_key(alerts: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(alert.get("code") or ""), str(alert.get("node_id") or ""))
        for alert in alerts
    )


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _line_entry(node: Mapping[str, Any], number: Any) -> Mapping[str, Any] | None:
    try:
        wanted = int(number)
    except (TypeError, ValueError):
        return None
    for line in node.get("lines") or []:
        if not isinstance(line, Mapping):
            continue
        try:
            if int(line.get("line")) == wanted:
                return line
        except (TypeError, ValueError):
            continue
    return None


def _action_confirmed(action: ActionSpec, node: Any) -> bool | None:
    if not isinstance(node, Mapping) or not node:
        return False
    name = str(action.action or "").strip().casefold()
    state = _upper(node.get("state"))
    if name in {"line_on", "line_off"}:
        line = _line_entry(node, action.value)
        if line is None:
            return False
        expected = "ON" if name == "line_on" else "OFF"
        return _upper(line.get("state")) == expected
    if name == "reset_protection":
        line = _line_entry(node, action.value)
        if line is None:
            return False
        return _upper(line.get("protection_state")) == "ARMED"
    if name == "start_job":
        if state not in {"PRINTING", "RUNNING"}:
            return False
        expected = str(action.value or "").strip()
        current = str(node.get("file_name") or "").strip()
        return not expected or not current or current.casefold() == expected.casefold()
    if name == "resume":
        hold = _upper(node.get("hold_level"))
        return (
            not as_bool(node.get("paused"))
            and hold not in {"MANDATORY", "SAFETY", "ALARM"}
            and state not in {"PAUSED", "ALARM", "ERROR"}
        )
    if name == "home":
        return as_bool(node.get("homed"))
    if name == "turn_on":
        if "enabled" in node:
            return as_bool(node.get("enabled"))
        return state in {"ON", "RUNNING", "ACTIVE", "AVAILABLE"}
    if name == "set_mode":
        return _upper(node.get("mode")) == _upper(action.value)
    if name == "set_primary_sensor":
        return str(node.get("primary_co2_sensor") or "").strip() == str(action.value or "").strip()
    if name == "reset_interlock":
        interlocks = node.get("interlocks")
        if not isinstance(interlocks, Mapping):
            return False
        entry = interlocks.get(str(action.value))
        if not isinstance(entry, Mapping):
            return False
        return _upper(entry.get("state")) not in {"TRIPPED", "BLOCKED", "ALARM"}
    if name == "reset_error":
        return not as_bool(node.get("error")) and not node.get("error_code")
    if name == "set_fan":
        expected = _upper(action.value)
        return _upper(node.get("fan_state")) == expected or _upper(node.get("fan_mode")) == expected
    if name == "set_data_trust":
        trust = node.get("data_trust")
        value = action.value
        if not isinstance(trust, Mapping) or not isinstance(value, Mapping):
            return None
        suspect = str(value.get("node") or "")
        param = str(value.get("param") or "")
        expected = _upper(value.get("trust"))
        found = trust.get(f"{suspect}.{param}")
        if found is None:
            nested = trust.get(suspect)
            if isinstance(nested, Mapping):
                found = nested.get(param)
        if found is None:
            return None
        return _upper(found) == expected
    return None


def _observed(action: ActionSpec, node: Any) -> dict[str, Any] | None:
    if not isinstance(node, Mapping):
        return None
    name = str(action.action or "").strip().casefold()
    if name in {"line_on", "line_off", "reset_protection"}:
        line = _line_entry(node, action.value)
        if line is None:
            return None
        return {
            "line": line.get("line"),
            "state": line.get("state"),
            "protection_state": line.get("protection_state"),
            "current_a": line.get("current_a"),
            "power_w": line.get("power_w"),
        }
    if name in {"start_job", "resume", "home"}:
        keys = ("state", "paused", "homed", "file_name", "progress_pct", "hold_level", "hold_reason")
    elif name in {"turn_on", "set_mode"}:
        keys = ("state", "enabled", "mode", "airflow_pct")
    elif name == "set_fan":
        keys = ("fan_state", "fan_mode", "inverter_temp_c")
    elif name == "reset_error":
        keys = ("state", "error", "error_code")
    elif name == "set_primary_sensor":
        keys = ("primary_co2_sensor",)
    elif name == "reset_interlock":
        interlocks = node.get("interlocks")
        entry = interlocks.get(str(action.value)) if isinstance(interlocks, Mapping) else None
        return {"interlock": action.value, "entry": entry}
    elif name == "set_data_trust":
        return {"data_trust": node.get("data_trust")}
    else:
        keys = ("state",)
    return {key: node.get(key) for key in keys if key in node}


class DomeAgent:
    """Stateful autonomous shift agent."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        api: DomeAgentAPI | None = None,
        log: AgentLog | None = None,
        state: StateStore | None = None,
        rules: RuleEngine | None = None,
        advisor: DeepSeekAdvisor | None = None,
    ) -> None:
        self.config = config
        self.log = log or AgentLog(
            config.log_dir,
            secrets=config.secrets(),
        )
        self.state = state or StateStore(config.state_file)
        self.api = api or DomeAgentAPI(
            config.dome_url,
            config.dome_key,
            name=config.agent_name,
            timeout=config.telemetry_timeout_s,
            min_send_gap=config.min_send_gap_s,
        )
        self.rules = rules or RuleEngine(self.state)
        self.advisor = advisor or DeepSeekAdvisor(config, self.log)

        self._stop = asyncio.Event()
        self._stop_reason: str | None = None
        self._deadline: float | None = None
        self._role: str | None = None
        self._team: int | None = None
        self._shutdown_done = False
        self._last_chat_poll = 0.0
        self._last_decision_at = 0.0
        self._last_observation_at = 0.0
        self._last_situation: str | None = None
        self._last_observed_situation: str | None = None
        self._last_alert_key: tuple[tuple[str, str], ...] = ()
        self._dry_run_seen: set[str] = set()
        self._advice_task: asyncio.Task[None] | None = None
        self._recent_chat: deque[dict[str, str]] = deque(maxlen=8)
        self._sent_message_times: deque[float] = deque()
        self._guest_notice_logged = False
        self.stats: dict[str, int] = {
            "telemetry_frames": 0,
            "chat_messages": 0,
            "decisions": 0,
            "actions": 0,
            "action_failures": 0,
            "messages": 0,
            "reconnects": 0,
        }

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def request_stop(self, reason: str = "requested") -> None:
        if not self._stop.is_set():
            self._stop_reason = reason
            self._stop.set()

    async def run(self) -> None:
        started = time.monotonic()
        if self.config.run_seconds > 0:
            self._deadline = started + self.config.run_seconds
        self.log.write(
            "start",
            agent=self.config.agent_name,
            version=self.config.version,
            models=[self.config.deepseek_model] if self.config.llm_available else [],
            config=self.config.safe_dict(),
        )
        try:
            while not self._stop.is_set():
                if self._past_deadline():
                    self.request_stop("run_seconds")
                    break
                if self._role is None:
                    try:
                        await self._connect()
                    except (DomeError, OSError, asyncio.TimeoutError) as exc:
                        self._log_exception("connect", exc)
                        await self._sleep(self.config.reconnect_delay_s)
                    continue
                if self._role != "team":
                    await self._wait_for_shift()
                    continue
                try:
                    await self._cycle()
                except (DomeAuthError, DomeSessionRevoked) as exc:
                    await self._handle_disconnect(exc, revoked=True)
                except (DomeError, OSError, asyncio.TimeoutError) as exc:
                    await self._handle_disconnect(exc)
                except asyncio.CancelledError:
                    self.request_stop("cancelled")
                    raise
                except Exception as exc:  # keep the shift alive on a bad frame
                    self._log_exception("cycle", exc)
                    await self._sleep(LOOP_INTERVAL_S)
                else:
                    await self._sleep(LOOP_INTERVAL_S)
        except asyncio.CancelledError:
            self.request_stop("cancelled")
            raise
        except Exception as exc:
            self._log_exception("agent", exc)
            self.request_stop("error")
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._role is not None:
            try:
                self.log.write(
                    "connection",
                    event="disconnected",
                    role=self._role,
                    team=self._team,
                    details={"reason": self._stop_reason or "shutdown"},
                )
            except Exception:
                pass
        advice_task = self._advice_task
        self._advice_task = None
        if advice_task is not None and not advice_task.done():
            advice_task.cancel()
            try:
                await advice_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        try:
            await self.advisor.close()
        except Exception:
            pass
        try:
            await self.api.close()
        except Exception:
            pass
        try:
            self.log.write(
                "stop",
                reason=self._stop_reason or "normal",
                stats=dict(self.stats),
            )
        finally:
            try:
                self.log.close()
            except Exception:
                pass

    def _past_deadline(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    async def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        if self._deadline is not None:
            seconds = min(seconds, max(0.0, self._deadline - time.monotonic()))
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        auth = await self.api.connect()
        role, team = await self._detect_role(auth)
        role_key = _safe_text(role, 40).casefold()
        self._role = "team" if role_key in {"team", "admin"} else "guest"
        self._team = team
        self._guest_notice_logged = False
        self.log.write(
            "connection",
            event="auth",
            role=role_key or "unknown",
            team=team,
            details={
                "url": self.config.dome_url,
                "name": self.config.agent_name,
                "shift": "active" if self._role == "team" else "guest; waiting for shift",
            },
        )
        self._last_chat_poll = 0.0

    async def _detect_role(self, auth: Any) -> tuple[str | None, int | None]:
        role: Any = None
        team: Any = None
        if isinstance(auth, Mapping):
            role = auth.get("role")
            team = auth.get("team")
        client = getattr(self.api, "client", None)
        if role is None:
            role = getattr(client, "role", None)
        if team is None:
            team = getattr(client, "team", None)
        if role is None:
            try:
                who = await self.api.whoami()
            except DomeError:
                raise
            if isinstance(who, Mapping):
                role = who.get("role")
                team = who.get("team")
        try:
            team_value = int(team) if team is not None else None
        except (TypeError, ValueError):
            team_value = None
        return (str(role) if role is not None else None, team_value)

    async def _wait_for_shift(self) -> None:
        if not self._guest_notice_logged:
            self.log.write(
                "connection",
                event="auth",
                role="guest",
                team=self._team,
                details={"state": "guest session; reconnect and wait for the team shift"},
            )
            self._guest_notice_logged = True
        await self._sleep(self.config.guest_recheck_s)
        if self._stop.is_set():
            return
        await self._close_api_quietly()
        self._role = None

    async def _handle_disconnect(self, exc: Exception, *, revoked: bool = False) -> None:
        self.stats["reconnects"] += 1
        try:
            self.log.write(
                "connection",
                event="revoked" if revoked else "disconnected",
                role=self._role,
                team=self._team,
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception:
            pass
        self._role = None
        await self._close_api_quietly()
        await self._sleep(self.config.reconnect_delay_s)

    async def _close_api_quietly(self) -> None:
        try:
            await self.api.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # one shift cycle
    # ------------------------------------------------------------------

    async def _cycle(self) -> None:
        data = await self.api.get_telemetry(max_age=TELEMETRY_MAX_AGE_S)
        snapshot = Snapshot(data, co2_override=self.state.co2_override)
        self.log.set_dome_time(snapshot.clock)
        self.stats["telemetry_frames"] += 1

        await self._poll_chat(snapshot)

        decision = self.rules.decide(snapshot, self.state.task)
        if self._should_log_decision(snapshot, decision):
            decision_ref = self._log_decision(snapshot, decision)
        else:
            decision_ref = None

        if decision.task_changed and decision.task is not None:
            self._save_task(decision.task)

        for action in sorted(decision.actions, key=lambda item: -item.priority):
            await self._execute_action(action, decision_ref)

        for message in decision.messages:
            await self._send_operator_message(
                message.text,
                dedupe_key=message.dedupe_key,
                min_interval_s=message.min_interval_s,
                to=message.to,
            )

        self._schedule_advice(snapshot, decision)
        self._maybe_observe(snapshot, decision)

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    async def _poll_chat(self, snapshot: Snapshot) -> None:
        now = time.monotonic()
        if now - self._last_chat_poll < CHAT_POLL_INTERVAL_S:
            return
        self._last_chat_poll = now
        after_id = self.state.last_message_id
        try:
            raw = await self.api.read_messages(
                since_id=after_id if after_id > 0 else None,
                limit=50,
            )
        except (DomeSessionRevoked, DomeConnectionError, DomeTimeout):
            raise
        except DomeError as exc:
            self._log_exception("chat_read", exc)
            return
        if not raw:
            return

        raw = [message for message in raw if isinstance(message, Mapping)]
        if not raw:
            return
        max_id = max((int(message.get("id") or 0) for message in raw), default=0)
        operator_messages = new_operator_messages(
            raw,
            after_id=after_id,
            agent_name=self.config.agent_name,
            team=self._team or 3,
        )

        if after_id <= 0:
            cutoff = time.time() - 300.0
            recent = [
                message
                for message in operator_messages
                if message.time is None or message.time >= cutoff
            ]
            skipped = len(operator_messages) - len(recent)
            if skipped:
                self.log.write(
                    "decision",
                    situation="Пропущен старый backlog чата",
                    cause=f"первый запуск: {skipped} старых сообщений не являются текущим заданием",
                    confidence=0.9,
                    evidence=[],
                    plan=["начать с новых сообщений"],
                    source="rules",
                )
            operator_messages = recent

        for message in operator_messages:
            self._on_operator_message(message)

        if max_id > self.state.last_message_id:
            self._mark_message(max_id)

    def _on_operator_message(self, message: ChatMessage) -> None:
        self.stats["chat_messages"] += 1
        self._recent_chat.append({"direction": "in", "text": message.text[:700]})
        self.log.write(
            "message",
            direction="in",
            sender=message.sender_name or message.sender_role,
            role=message.sender_role,
            team=message.sender_team,
            to=message.to,
            text=message.text[:2000],
            message_id=message.id,
        )
        task = parse_operator_task(message)
        if task is not None and task.items:
            self._save_task(task.as_dict())
            self.log.write(
                "decision",
                situation="Принято новое задание Оператора",
                cause=message.text[:500],
                confidence=0.95,
                evidence=[{"message_id": message.id, "sender": message.sender_name}],
                plan=[item.text[:200] for item in task.items],
                source="rules",
                task=task.as_dict(),
            )
        self._mark_message(message.id)

    def _save_task(self, task: Mapping[str, Any]) -> None:
        try:
            self.state.set_task(dict(task))
        except OSError as exc:
            self._log_exception("state_task", exc)

    def _mark_message(self, message_id: int) -> None:
        try:
            self.state.mark_message(message_id)
        except OSError as exc:
            self._log_exception("state_chat", exc)

    # ------------------------------------------------------------------
    # decisions, actions, messages
    # ------------------------------------------------------------------

    def _should_log_decision(self, snapshot: Snapshot, decision: RuleResult) -> bool:
        if decision.actions or decision.messages or decision.task_changed:
            return True
        if decision.situation != self._last_situation:
            return True
        if _alert_key(snapshot.alerts()) != self._last_alert_key:
            return True
        return (time.monotonic() - self._last_decision_at) >= self.config.observation_interval_s

    def _log_decision(self, snapshot: Snapshot, decision: RuleResult) -> int:
        ref = self.log.write(
            "decision",
            situation=decision.situation,
            cause=decision.cause,
            confidence=decision.confidence,
            evidence=decision.evidence,
            plan=decision.plan,
            source=decision.source,
            actions=[action.as_dict() for action in decision.actions],
            messages=[message.text for message in decision.messages],
            task=decision.task,
            alerts=snapshot.alerts()[:8],
        )
        self.stats["decisions"] += 1
        self._last_decision_at = time.monotonic()
        self._last_situation = decision.situation
        return ref

    async def _execute_action(self, action: ActionSpec, decision_ref: int | None) -> None:
        if self.config.dry_run:
            if action.key in self._dry_run_seen:
                return
            self._dry_run_seen.add(action.key)
        request = action.request()
        started = time.perf_counter()
        try:
            if action.value is NO_VALUE:
                result = await self.api.control_device(
                    action.node_id,
                    action.action,
                    dry_run=self.config.dry_run,
                    verify=action.verify,
                    max_age=TELEMETRY_MAX_AGE_S,
                )
            else:
                result = await self.api.control_device(
                    action.node_id,
                    action.action,
                    action.value,
                    dry_run=self.config.dry_run,
                    verify=action.verify,
                    max_age=TELEMETRY_MAX_AGE_S,
                )
            success = bool(
                result.get("success")
                or result.get("accepted")
                or result.get("dry_run")
            )
            raw_response = result.get("response")
            response_error = (
                raw_response.get("error") if isinstance(raw_response, Mapping) else None
            )
            response = {
                "success": success,
                "accepted": bool(result.get("accepted", success)),
                "error": result.get("error") or response_error,
                "dry_run": bool(result.get("dry_run")),
                "verify_error": result.get("verify_error"),
            }
            action_ref = self.log.write(
                "action",
                request=request,
                response=response,
                reason=action.reason,
                dry_run=self.config.dry_run,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
                ref=decision_ref,
            )
            self.stats["actions"] += 1
            verified = True
            verification_error = None
            if action.verify and not self.config.dry_run:
                verified, observed, verification_error = await self._verify_action(action, result)
                self.log.write(
                    "verification",
                    ref=action_ref,
                    expected=self._expected(action),
                    observed=observed,
                    ok=verified,
                    error=verification_error,
                )
            if not success or not verified:
                self.stats["action_failures"] += 1
            if not self.config.dry_run:
                self._mark_action(action.key, success and verified)
            if (
                success
                and verified
                and not self.config.dry_run
                and action.action == "set_primary_sensor"
                and action.value is not NO_VALUE
            ):
                try:
                    self.state.set_co2_override(
                        {"node_id": str(action.value), "at": time.time(), "reason": action.reason}
                    )
                except OSError as exc:
                    self._log_exception("state_co2", exc)
        except (DomeSessionRevoked, DomeConnectionError, DomeTimeout):
            self.stats["action_failures"] += 1
            self.log.write(
                "action",
                request=request,
                response={"success": False, "error": "connection failure"},
                reason=action.reason,
                dry_run=self.config.dry_run,
                ref=decision_ref,
            )
            if not self.config.dry_run:
                self._mark_action(action.key, False)
            raise
        except Exception as exc:
            self.stats["action_failures"] += 1
            self._log_exception("action", exc, node_id=action.node_id, action=action.action)
            self.log.write(
                "action",
                request=request,
                response={"success": False, "error": f"{type(exc).__name__}: {exc}"},
                reason=action.reason,
                dry_run=self.config.dry_run,
                ref=decision_ref,
            )
            if not self.config.dry_run:
                self._mark_action(action.key, False)

    async def _verify_action(
        self,
        action: ActionSpec,
        result: Mapping[str, Any],
    ) -> tuple[bool, Any, str | None]:
        after = result.get("after")
        confirmed = _action_confirmed(action, after)
        if confirmed is True:
            return True, _observed(action, after), None
        if confirmed is None:
            diff = result.get("diff")
            return bool(diff), diff if diff is not None else _observed(action, after), result.get("verify_error")
        last = after
        last_error = result.get("verify_error")
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                data = await asyncio.wait_for(
                    self.api.get_telemetry(
                        action.node_id,
                        max_age=TELEMETRY_MAX_AGE_S,
                        next_frame=True,
                    ),
                    timeout=min(2.5, remaining),
                )
            except (asyncio.TimeoutError, DomeError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(0.2)
                continue
            node = data.get("telemetry") if isinstance(data, Mapping) else None
            if not isinstance(node, Mapping):
                last_error = "telemetry response has no node"
                continue
            last = node
            if _action_confirmed(action, node) is True:
                return True, _observed(action, node), None
        return False, _observed(action, last), last_error or "telemetry did not confirm action"

    def _mark_action(self, key: str, success: bool) -> None:
        try:
            self.state.mark_action(key, success=success)
        except OSError as exc:
            self._log_exception("state_action", exc)

    def _expected(self, action: ActionSpec) -> str:
        value = "" if action.value is NO_VALUE else f" {action.value!r}"
        if action.action in {"line_on", "line_off"}:
            return f"{action.node_id}.line {action.value} = {'ON' if action.action == 'line_on' else 'OFF'}"
        if action.action == "start_job":
            return f"{action.node_id}.state changes to PRINTING/RUNNING for {action.value!r}"
        if action.action == "resume":
            return f"{action.node_id} is no longer paused"
        if action.action == "home":
            return f"{action.node_id}.homed or state becomes operational"
        if action.action == "set_primary_sensor":
            return f"{action.node_id}.primary_co2_sensor = {action.value!r}"
        if action.action == "set_data_trust":
            return f"{action.node_id} data trust is applied"
        if action.action == "reset_interlock":
            return f"{action.node_id}.{action.value} is no longer tripped"
        if action.action == "reset_protection":
            return f"{action.node_id}.line {action.value} protection = ARMED"
        return f"{action.node_id}.{action.action}{value} is reflected in telemetry"

    async def _send_operator_message(
        self,
        text: str,
        *,
        dedupe_key: str,
        min_interval_s: float,
        to: str = "operator",
    ) -> None:
        text = text.strip()
        if not text:
            return
        try:
            due = self.state.should_notify(
                f"msg:{dedupe_key}",
                text,
                min_interval_s=min_interval_s,
            )
        except OSError as exc:
            self._log_exception("state_message", exc)
            due = True
        if not due:
            return
        if self.config.dry_run:
            self.log.write(
                "message",
                direction="out",
                to=to,
                text=text[:2000],
                dry_run=True,
                sent=False,
            )
            self._recent_chat.append({"direction": "out", "text": text[:700]})
            return
        await self._chat_throttle()
        try:
            await self.api.post_message(text, to=to, name=self.config.agent_name)
        except (DomeSessionRevoked, DomeConnectionError, DomeTimeout):
            raise
        except Exception as exc:
            self._log_exception("chat_post", exc)
            return
        self.stats["messages"] += 1
        self._sent_message_times.append(time.monotonic())
        self._recent_chat.append({"direction": "out", "text": text[:700]})
        self.log.write(
            "message",
            direction="out",
            to=to,
            text=text[:2000],
            sent=True,
        )

    async def _chat_throttle(self) -> None:
        now = time.monotonic()
        while self._sent_message_times and now - self._sent_message_times[0] >= 10.0:
            self._sent_message_times.popleft()
        if len(self._sent_message_times) >= 10:
            wait = 10.0 - (now - self._sent_message_times[0])
            if wait > 0:
                await self._sleep(wait)

    # ------------------------------------------------------------------
    # LLM advice (advisory only, after safety actions)
    # ------------------------------------------------------------------

    def _should_ask_llm(self, snapshot: Snapshot, decision: RuleResult) -> bool:
        if not self.advisor.available:
            return False
        if decision.actions or decision.messages or decision.task_changed:
            return True
        if snapshot.alerts():
            return True
        return decision.situation != "Нет требующих вмешательства событий"

    def _schedule_advice(self, snapshot: Snapshot, decision: RuleResult) -> None:
        if not self._should_ask_llm(snapshot, decision):
            return
        if self._advice_task is not None and not self._advice_task.done():
            return
        self._advice_task = asyncio.create_task(
            self._advise_and_send(snapshot, decision),
            name="dome-agent-advice",
        )

    async def _advise_and_send(self, snapshot: Snapshot, decision: RuleResult) -> None:
        context = {
            "snapshot": snapshot.brief(),
            "rule_decision": decision.as_dict(),
            "task": self.state.task,
            "recent_chat": list(self._recent_chat),
        }
        try:
            advice = await self.advisor.advise(
                context,
                purpose=f"диагностика: {decision.situation}"[:200],
            )
            if advice is not None:
                await self._send_advice(snapshot, decision, advice)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_exception("llm_advice", exc)

    async def _send_advice(
        self,
        snapshot: Snapshot,
        decision: RuleResult,
        advice: Advice,
    ) -> None:
        if not advice.operator_message or decision.messages:
            return
        if not advice.needs_operator and not snapshot.alerts():
            return
        digest = hashlib.sha1(advice.operator_message.encode("utf-8")).hexdigest()[:16]
        await self._send_operator_message(
            advice.operator_message,
            dedupe_key=f"llm:{digest}",
            min_interval_s=LLM_MESSAGE_MIN_INTERVAL_S,
        )

    # ------------------------------------------------------------------
    # observations and errors
    # ------------------------------------------------------------------

    def _maybe_observe(self, snapshot: Snapshot, decision: RuleResult) -> None:
        alerts = snapshot.alerts()
        alert_key = _alert_key(alerts)
        due = (time.monotonic() - self._last_observation_at) >= self.config.observation_interval_s
        changed = (
            alert_key != self._last_alert_key
            or decision.situation != self._last_observed_situation
        )
        if not due and not changed:
            return
        energy = snapshot.energy()
        production = snapshot.production()
        metrics = {
            "co2_ppm": snapshot.co2.chosen_ppm,
            "co2_source": snapshot.co2.chosen_id,
            "co2_false_alarm": snapshot.co2.false_alarm,
            "soc_pct": energy.get("soc_pct"),
            "printer": production["printer_3d_01"].get("state"),
            "cnc": production["cnc_01"].get("state"),
            "alerts": len(alerts),
            "task": self._task_metric(),
        }
        if metrics["co2_ppm"] is not None and energy.get("soc_pct") is not None:
            summary = (
                f"{decision.situation}. "
                f"CO2 {metrics['co2_ppm']:.0f} ppm; "
                f"АКБ {energy.get('soc_pct'):.0f}%; "
                f"принтер {metrics['printer']}; ЧПУ {metrics['cnc']}."
            )
        else:
            summary = (
                f"{decision.situation}. "
                f"принтер {metrics['printer']}; ЧПУ {metrics['cnc']}."
            )
        self.log.write(
            "observation",
            summary=summary,
            metrics=metrics,
            anomalies=[alert.get("message") for alert in alerts[:8]],
            task=self.state.task,
        )
        self._last_observation_at = time.monotonic()
        self._last_alert_key = alert_key
        self._last_observed_situation = decision.situation

    def _task_metric(self) -> str:
        task = self.state.task
        if not isinstance(task, Mapping):
            return "none"
        items = task.get("items")
        if not isinstance(items, list):
            return "none"
        statuses = [str(item.get("status") or "pending") for item in items if isinstance(item, dict)]
        return ",".join(statuses) if statuses else "none"

    def _log_exception(self, where: str, exc: BaseException, **fields: Any) -> None:
        try:
            self.log.write(
                "error",
                where=where,
                error=f"{type(exc).__name__}: {exc}"[:1000],
                traceback=traceback.format_exc()[:4000],
                **fields,
            )
        except Exception:
            pass


def _install_signal_handlers(agent: DomeAgent) -> list[signal.Signals]:
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, agent.request_stop, sig.name.lower())
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed.append(sig)
    return installed


def _remove_signal_handlers(signals: list[signal.Signals]) -> None:
    loop = asyncio.get_running_loop()
    for sig in signals:
        try:
            loop.remove_signal_handler(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass


async def run_agent(config: AgentConfig) -> None:
    agent = DomeAgent(config)
    installed = _install_signal_handlers(agent)
    try:
        await agent.run()
    finally:
        _remove_signal_handlers(installed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dome-agent",
        description="Autonomous Dome shift agent (DeepSeek advisory + deterministic rules).",
    )
    parser.add_argument("--env-file", default=".env", help="optional .env file (default: .env)")
    parser.add_argument("--url", default=None, help="override DOME_URL")
    parser.add_argument("--name", default=None, help="override AGENT_NAME")
    parser.add_argument("--log-dir", default=None, help="override AGENT_LOG_DIR")
    parser.add_argument("--state-file", default=None, help="override AGENT_STATE_FILE")
    parser.add_argument("--dry-run", action="store_true", default=None, help="analyze and log but send no commands/messages")
    parser.add_argument("--run-seconds", type=float, default=None, help="stop after N seconds (0 means indefinitely)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.url:
        overrides["dome_url"] = args.url
    if args.name:
        overrides["agent_name"] = args.name
    if args.log_dir:
        overrides["log_dir"] = Path(args.log_dir)
    if args.state_file:
        overrides["state_file"] = Path(args.state_file)
    if args.dry_run is not None:
        overrides["dry_run"] = True
    if args.run_seconds is not None:
        overrides["run_seconds"] = args.run_seconds
    try:
        config = AgentConfig.from_env(env_file=args.env_file, overrides=overrides)
    except Exception as exc:
        print(f"agent configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        asyncio.run(run_agent(config))
    except KeyboardInterrupt:
        return 130
    return 0


