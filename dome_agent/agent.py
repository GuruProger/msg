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
import re
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

from .analysis import Snapshot
from .chat import ChatMessage, new_operator_messages, parse_operator_task
from .config import AgentConfig
from .control import (
    HARD_DENY_ACTIONS,
    DomeTool,
    ModelCommand,
    ModelDecision,
    PendingProposal,
    compact_catalog,
    is_auto_protective,
)
from .gate import CommandGate
from .llm import Advice, DeepSeekAdvisor
from .logging import AgentLog
from .rules import NO_VALUE, ActionSpec, OutgoingMessage, RuleEngine, RuleResult
from .state import StateStore
from .verification import ExpectedEffect, expected_effect, meaningful_diff

CHAT_POLL_INTERVAL_S = 3.0
CHAT_RECONCILE_INTERVAL_S = 300.0
TELEMETRY_MAX_AGE_S = 5.0
LOOP_INTERVAL_S = 1.0
LLM_MESSAGE_MIN_INTERVAL_S = 600.0
LLM_CONTROL_INTERVAL_S = 10.0
PENDING_TTL_S = 300.0
ACTION_FAILURE_COOLDOWN_S = 900.0
ACTION_DRAIN_TIMEOUT_S = 7.0
ACTION_DEPENDENCY_TIMEOUT_S = 30.0
ACTION_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "blocked", "unknown"})
ACTION_NON_SUCCESS_STATUSES = frozenset({"failed", "blocked", "unknown"})

_CONFIRM_WORDS = {"да", "подтверждаю", "выполняй", "выполнить", "ок", "давай", "yes", "confirm"}
_CANCEL_WORDS = {"отмена", "отменить", "не надо", "нет", "стоп", "cancel", "no", "stop"}
_FORCE_WORDS = {"принудительно", "force", "форсируй"}


def _flat(text: Any) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", " ", str(text or "").casefold()).strip()


def _reply_kind(text: Any, *, awaiting_force: bool = False) -> str | None:
    """Classify an operator reply to a pending proposal."""
    flat = _flat(text)
    if not flat:
        return None
    words = set(flat.split())
    if words & _CANCEL_WORDS or flat in _CANCEL_WORDS:
        return "cancel"
    if words & _FORCE_WORDS or "принудительно" in flat:
        return "force"
    if words & _CONFIRM_WORDS:
        return "force" if awaiting_force else "confirm"
    return None


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _safe_text(value: Any, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


def _alert_key(alerts: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(alert.get("code") or ""), str(alert.get("node_id") or ""))
        for alert in alerts
    )


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
        gate: CommandGate | None = None,
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
        self.gate = gate or CommandGate(
            config.policy_file,
            control_enabled=config.control_enabled,
            yolo=config.yolo,
        )
        self.rules = rules or RuleEngine(self.state, yolo=config.yolo)
        self.advisor = advisor or DeepSeekAdvisor(config, self.log)

        self._stop = asyncio.Event()
        self._stop_reason: str | None = None
        self._deadline: float | None = None
        self._role: str | None = None
        self._team: int | None = None
        self._shutdown_done = False
        self._last_chat_poll = 0.0
        self._last_chat_reconcile = 0.0
        self._chat_generation_seen = self._chat_generation()
        self._last_decision_at = 0.0
        self._last_observation_at = 0.0
        self._last_situation: str | None = None
        self._last_observed_situation: str | None = None
        self._last_alert_key: tuple[tuple[str, str], ...] = ()
        self._dry_run_seen: set[str] = set()
        self._blocked_seen: set[str] = set()
        self._pending_action_keys: set[str] = set()
        self._action_queues: dict[
            str, asyncio.Queue[tuple[float, int, ActionSpec, int | None, bool]]
        ] = {}
        self._action_tasks: dict[str, asyncio.Task[None]] = {}
        self._queued_action_keys: set[str] = set()
        self._action_status: dict[str, str] = {}
        self._action_status_reason: dict[str, str] = {}
        self._action_done_events: dict[str, asyncio.Event] = {}
        self._action_seq = 0
        self._action_error: Exception | None = None
        self._recovered_actions: list[dict[str, Any]] = []
        self._advice_task: asyncio.Task[None] | None = None
        self._control_task: asyncio.Task[None] | None = None
        self._last_control_at = 0.0
        self._pending: PendingProposal | None = None
        self._incoming: list[ChatMessage] = []
        self._llm_notice_sent = False
        self.tool = DomeTool(self.api)
        self._recent_chat: deque[dict[str, str]] = deque(maxlen=8)
        self._sent_message_times: deque[float] = deque()
        self._guest_notice_logged = False
        self.stats: dict[str, int] = {
            "telemetry_frames": 0,
            "chat_messages": 0,
            "decisions": 0,
            "actions": 0,
            "action_failures": 0,
            "actions_blocked": 0,
            "proposals": 0,
            "confirmations": 0,
            "llm_commands": 0,
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
        if self.state.load_error:
            self.log.write(
                "error",
                where="state_load",
                error=self.state.load_error,
                recovered=True,
                backup=(
                    str(self.state.corrupt_path)
                    if self.state.corrupt_path is not None
                    else None
                ),
            )
        self._reconcile_pending_actions()
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
        await self._drain_action_queue()
        for task_attr in ("_control_task", "_advice_task"):
            task = getattr(self, task_attr, None)
            setattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
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

    async def _drain_action_queue(self) -> None:
        queues = list(self._action_queues.values())
        tasks = list(self._action_tasks.values())
        if not tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(queue.join() for queue in queues)),
                timeout=ACTION_DRAIN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self.log.write(
                "action_queue",
                event="drain_timeout",
                pending=sum(queue.qsize() for queue in queues),
                in_flight=len(self._queued_action_keys),
            )
        self._action_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
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
        self._last_chat_reconcile = 0.0

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
        self._raise_action_error()
        data = await self.api.get_telemetry(max_age=TELEMETRY_MAX_AGE_S)
        snapshot = Snapshot(data, co2_override=self.state.co2_override)
        self.log.set_dome_time(snapshot.clock)
        self.stats["telemetry_frames"] += 1

        await self._poll_chat(snapshot)
        await self._notify_recovered_actions()

        decision = self.rules.decide(snapshot, self.state.task)
        if decision.task_changed and decision.task is not None:
            self._save_task(decision.task)

        if self.config.mode == "yolo":
            # YOLO: the LLM is the primary decision-maker. Rule actions are
            # context only and are never executed directly.
            self._maybe_observe(snapshot, decision)
            await self._schedule_llm_control(snapshot, decision)
            return

        if self.config.mode == "normal":
            await self._handle_pending()

        if self._should_log_decision(snapshot, decision):
            decision_ref = self._log_decision(snapshot, decision)
        else:
            decision_ref = None

        if self.config.mode == "auto" or self.config.dry_run:
            self._enqueue_actions(decision.actions, decision_ref)
        else:
            # NORMAL: only the explicitly approved protective exceptions are
            # automatic. Everything else is proposed by the LLM and confirmed.
            self._enqueue_actions(
                [action for action in decision.actions if is_auto_protective(action)],
                decision_ref,
                protective=True,
            )

        for message in decision.messages:
            await self._send_operator_message(
                message.text,
                dedupe_key=message.dedupe_key,
                min_interval_s=message.min_interval_s,
                to=message.to,
            )

        if self.config.mode == "normal":
            await self._schedule_llm_control(snapshot, decision)
        else:
            self._schedule_advice(snapshot, decision)
        self._maybe_observe(snapshot, decision)

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    def _chat_generation(self) -> int:
        try:
            return max(0, int(getattr(self.api, "meeting_generation", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _reset_chat_cursor(self, reason: str) -> None:
        try:
            changed = self.state.reset_chat_cursor()
        except OSError as exc:
            self._log_exception("state_chat", exc)
            return
        if not changed:
            return
        self.log.write(
            "decision",
            situation="История чата очищена",
            cause=reason,
            confidence=1.0,
            evidence=[],
            plan=["перечитать чат с начала"],
            source="chat_epoch",
        )

    async def _poll_chat(self, snapshot: Snapshot) -> None:
        now = time.monotonic()
        if now - self._last_chat_poll < CHAT_POLL_INTERVAL_S:
            return
        self._last_chat_poll = now

        generation = self._chat_generation()
        if generation != self._chat_generation_seen:
            self._chat_generation_seen = generation
            self._reset_chat_cursor("сервер очистил историю чата")

        after_id = self.state.last_message_id
        reconcile = now - self._last_chat_reconcile >= CHAT_RECONCILE_INTERVAL_S
        if reconcile:
            self._last_chat_reconcile = now
        try:
            raw = await self.api.read_messages(
                since_id=None if (after_id <= 0 or reconcile) else after_id,
                limit=50,
            )
        except (DomeSessionRevoked, DomeConnectionError, DomeTimeout):
            raise
        except DomeError as exc:
            self._log_exception("chat_read", exc)
            return

        raw = [message for message in raw if isinstance(message, Mapping)]
        max_id = 0
        for message in raw:
            try:
                max_id = max(max_id, int(message.get("id") or 0))
            except (TypeError, ValueError):
                continue
        if reconcile and after_id > 0 and max_id < after_id:
            self._reset_chat_cursor(
                f"полное чтение чата заканчивается на id {max_id}, "
                f"локальный курсор {after_id}"
            )
            after_id = 0
        if not raw:
            return

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
        self._incoming.append(message)
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

    def _raise_action_error(self) -> None:
        error = self._action_error
        if error is None:
            return
        self._action_error = None
        raise error

    def _ensure_action_worker(self, lane: str) -> asyncio.Queue[
        tuple[float, int, ActionSpec, int | None, bool]
    ] | None:
        if self._stop.is_set():
            return None
        queue = self._action_queues.get(lane)
        if queue is None:
            queue = asyncio.Queue()
            self._action_queues[lane] = queue
        task = self._action_tasks.get(lane)
        if task is None or task.done():
            self._action_tasks[lane] = asyncio.create_task(
                self._action_worker(lane, queue),
                name=f"dome-agent-actions:{lane or 'system'}",
            )
        return queue

    def _enqueue_actions(
        self,
        actions: list[ActionSpec],
        decision_ref: int | None,
        *,
        protective: bool = False,
    ) -> None:
        by_key: dict[str, ActionSpec] = {}
        for action in actions:
            if isinstance(action, ActionSpec):
                by_key.setdefault(action.key, action)
        if not by_key:
            return
        ordered, cycles = self._order_actions(by_key)
        blocked = self._preflight_dependencies(ordered, by_key, cycles)
        for action in ordered:
            reason = blocked.get(action.key)
            if reason is not None:
                self._block_action_dependency(action, reason, decision_ref)
                continue
            self._enqueue_action(action, decision_ref, protective=protective)

    def _order_actions(
        self,
        by_key: dict[str, ActionSpec],
    ) -> tuple[list[ActionSpec], set[str]]:
        """Order actions so dependencies are queued before dependents.

        Independent actions keep the existing priority order.  A cycle is
        reported instead of being executed; the dependent gate must fail closed.
        """
        ordered: list[ActionSpec] = []
        temporary: set[str] = set()
        permanent: set[str] = set()
        cycles: set[str] = set()

        def visit(key: str, stack: list[str]) -> None:
            if key in permanent:
                return
            if key in temporary:
                try:
                    start = stack.index(key)
                except ValueError:
                    start = 0
                cycles.update(stack[start:])
                return
            action = by_key.get(key)
            if action is None:
                return
            temporary.add(key)
            stack.append(key)
            for dependency in action.depends_on:
                visit(dependency, stack)
            stack.pop()
            temporary.discard(key)
            permanent.add(key)
            ordered.append(action)

        for action in sorted(by_key.values(), key=lambda item: -item.priority):
            visit(action.key, [])
        return ordered, cycles

    def _preflight_dependencies(
        self,
        ordered: list[ActionSpec],
        by_key: dict[str, ActionSpec],
        cycles: set[str],
    ) -> dict[str, str]:
        """Return deterministic block reasons for actions in this plan."""
        blocked: dict[str, str] = {}
        for action in ordered:
            if action.key in cycles:
                blocked[action.key] = (
                    "dependency cycle: " + ", ".join(sorted(cycles))
                )
                continue
            for dependency in action.depends_on:
                if dependency in blocked:
                    blocked[action.key] = (
                        f"dependency {dependency} is blocked: {blocked[dependency]}"
                    )
                    break
                if dependency in by_key:
                    continue
                if self._dependency_satisfied(dependency):
                    continue
                if self._action_status.get(dependency) in {"queued", "running"}:
                    continue
                blocked[action.key] = (
                    f"dependency {dependency} was not scheduled or did not succeed"
                )
                break
        return blocked

    def _dependency_satisfied(self, key: str) -> bool:
        if self._action_status.get(key) == "succeeded":
            return True
        return self.state.action_succeeded(key)

    def _ensure_action_state(self, action: ActionSpec) -> None:
        if self._action_status.get(action.key) in {"queued", "running"}:
            return
        self._action_status[action.key] = "queued"
        self._action_status_reason.pop(action.key, None)
        self._action_done_events[action.key] = asyncio.Event()

    def _set_action_status(
        self,
        key: str,
        status: str,
        reason: str | None = None,
    ) -> None:
        self._action_status[key] = status
        if reason:
            self._action_status_reason[key] = reason
        else:
            self._action_status_reason.pop(key, None)
        if status in ACTION_TERMINAL_STATUSES:
            event = self._action_done_events.get(key)
            if event is not None:
                event.set()

    def _block_action(
        self,
        action: ActionSpec,
        reason: str,
        decision_ref: int | None,
        *,
        source: str,
    ) -> None:
        self._ensure_action_state(action)
        self._set_action_status(action.key, "blocked", reason)
        block_key = f"{source}:{action.key}:{reason}"
        if block_key in self._blocked_seen:
            return
        self._blocked_seen.add(block_key)
        self.stats["actions_blocked"] += 1
        fields: dict[str, Any] = {
            "request": action.request(),
            "reason": reason,
            "unsafe": False,
            "ref": decision_ref,
            "expect": getattr(action.expect, "description", None),
            "source": source,
        }
        if source == "dependency":
            fields["depends_on"] = list(action.depends_on)
        self.log.write("action_blocked", **fields)

    def _block_action_dependency(
        self,
        action: ActionSpec,
        reason: str,
        decision_ref: int | None,
    ) -> None:
        self._block_action(
            action,
            reason,
            decision_ref,
            source="dependency",
        )

    async def _wait_for_dependencies(
        self,
        action: ActionSpec,
    ) -> tuple[bool, str | None]:
        if not action.depends_on:
            return True, None
        deadline = time.monotonic() + ACTION_DEPENDENCY_TIMEOUT_S
        for dependency in action.depends_on:
            status = self._action_status.get(dependency)
            if status == "succeeded":
                continue
            if status in ACTION_NON_SUCCESS_STATUSES:
                detail = self._action_status_reason.get(dependency)
                return False, f"dependency {dependency} {status}: {detail or 'no reason'}"
            if status is None:
                if self._dependency_satisfied(dependency):
                    continue
                return False, f"dependency {dependency} was not scheduled"
            event = self._action_done_events.get(dependency)
            if event is None:
                if self._dependency_satisfied(dependency):
                    continue
                return False, f"dependency {dependency} has no completion event"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, f"dependency {dependency} timed out"
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return False, f"dependency {dependency} timed out"
            status = self._action_status.get(dependency)
            if status == "succeeded":
                continue
            if status in ACTION_NON_SUCCESS_STATUSES:
                detail = self._action_status_reason.get(dependency)
                return False, f"dependency {dependency} {status}: {detail or 'no reason'}"
            return False, f"dependency {dependency} did not finish successfully"
        return True, None

    async def _run_action(
        self,
        action: ActionSpec,
        decision_ref: int | None,
        *,
        protective: bool,
    ) -> None:
        if not self.config.dry_run and self._role != "team":
            self._block_action(
                action,
                f"session role is {self._role or 'unknown'}, expected team",
                decision_ref,
                source="role",
            )
            return
        self._ensure_action_state(action)
        ready, reason = await self._wait_for_dependencies(action)
        if not ready:
            self._block_action_dependency(
                action,
                reason or "dependency did not succeed",
                decision_ref,
            )
            return
        self._set_action_status(action.key, "running")
        try:
            if protective:
                outcome = await self._execute_protective(action, decision_ref)
            else:
                outcome = await self._execute_action(action, decision_ref)
        except asyncio.CancelledError:
            self._set_action_status(action.key, "blocked", "cancelled")
            raise
        except Exception as exc:
            self._set_action_status(
                action.key,
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
            raise
        if outcome in {"running", "pending"}:
            return
        status = outcome or "succeeded"
        self._set_action_status(action.key, status)
        if status in {"failed", "unknown"} and not self.config.dry_run:
            await self._notify_action_failure(action, status)

    def _enqueue_action(
        self,
        action: ActionSpec,
        decision_ref: int | None,
        *,
        protective: bool = False,
    ) -> None:
        if self._stop.is_set():
            return
        queue_key = f"{'protective:' if protective else ''}{action.key}"
        if queue_key in self._queued_action_keys:
            return
        self._ensure_action_state(action)
        lane = str(action.node_id or "").strip().casefold()
        queue = self._ensure_action_worker(lane)
        if queue is None:
            return
        self._queued_action_keys.add(queue_key)
        self._action_seq += 1
        queue.put_nowait(
            (-float(action.priority), self._action_seq, action, decision_ref, protective)
        )

    async def _action_worker(
        self,
        lane: str,
        queue: asyncio.Queue[tuple[float, int, ActionSpec, int | None, bool]],
    ) -> None:
        while True:
            _priority, _seq, action, decision_ref, protective = await queue.get()
            queue_key = f"{'protective:' if protective else ''}{action.key}"
            try:
                await self._run_action(
                    action,
                    decision_ref,
                    protective=protective,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._action_error = exc
                self._log_exception(
                    "action_worker",
                    exc,
                    node_id=action.node_id,
                    action=action.action,
                )
            finally:
                self._queued_action_keys.discard(queue_key)
                queue.task_done()

    async def _execute_action(self, action: ActionSpec, decision_ref: int | None) -> str:
        if self.config.dry_run:
            if action.key in self._dry_run_seen:
                return "succeeded"
            self._dry_run_seen.add(action.key)
        request = action.request()
        decision = self.gate.check(action, dry_run=self.config.dry_run)
        if not decision.allowed:
            if action.key not in self._blocked_seen:
                self._blocked_seen.add(action.key)
                self.stats["actions_blocked"] += 1
                self.log.write(
                    "action_blocked",
                    request=request,
                    reason=decision.reason,
                    unsafe=decision.unsafe,
                    ref=decision_ref,
                )
            return "blocked"

        expect = (
            action.expect
            if isinstance(action.expect, ExpectedEffect)
            else expected_effect(
                action.node_id,
                action.action,
                action.value,
                has_value=action.value is not NO_VALUE,
            )
        )
        lock_key = self._action_lock_key(action)
        if (
            not self.config.yolo
            and not self.config.dry_run
            and self.state.action_failed_recently(action.key)
        ):
            self._log_pending_block(
                action,
                request,
                decision_ref,
                "recent verification failed: retry cooldown is active",
                expect.description,
            )
            return "blocked"
        if not self._claim_action_key(lock_key):
            self._log_pending_block(
                action,
                request,
                decision_ref,
                "action is already pending verification",
                expect.description,
            )
            return "running"

        verify = action.verify and not self.config.yolo
        allow_without_before = is_auto_protective(action)
        intent_started = False
        outcome: str | None = None
        if not self.config.dry_run:
            try:
                self.state.begin_action(
                    action.key,
                    request=request,
                    ref=decision_ref,
                    expected=expect.description,
                )
                intent_started = True
            except OSError as exc:
                self._log_exception("state_intent", exc)
                self._log_pending_block(
                    action,
                    request,
                    decision_ref,
                    "cannot persist action intent",
                    expect.description,
                )
                self._release_action_key(lock_key)
                return "blocked"
        started = time.perf_counter()
        try:
            if action.value is NO_VALUE:
                result = await self.api.control_device(
                    action.node_id,
                    action.action,
                    dry_run=self.config.dry_run,
                    verify=verify,
                    max_age=TELEMETRY_MAX_AGE_S,
                    allow_without_before=allow_without_before,
                )
            else:
                result = await self.api.control_device(
                    action.node_id,
                    action.action,
                    action.value,
                    dry_run=self.config.dry_run,
                    verify=verify,
                    max_age=TELEMETRY_MAX_AGE_S,
                    allow_without_before=allow_without_before,
                )
            if result.get("blocked"):
                block_key = f"before:{action.key}"
                if block_key not in self._blocked_seen:
                    self._blocked_seen.add(block_key)
                    self.stats["actions_blocked"] += 1
                    self.log.write(
                        "action_blocked",
                        request=request,
                        reason=result.get("error") or "fresh before-telemetry unavailable",
                        unsafe=decision.unsafe,
                        ref=decision_ref,
                        expect=expect.description,
                        source="before_telemetry",
                    )
                outcome = "blocked"
                return "blocked"
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
                "unknown": bool(result.get("unknown")),
                "before_error": result.get("before_error"),
            }
            action_ref = self.log.write(
                "action",
                request=request,
                response=response,
                reason=action.reason,
                dry_run=self.config.dry_run,
                unsafe=decision.unsafe,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
                ref=decision_ref,
                expect=expect.description,
            )
            self.stats["actions"] += 1
            verified = True
            verification_error = None
            if verify and not self.config.dry_run:
                verify_started = time.monotonic()
                verified, observed, verification_error = await self._verify_action(
                    action,
                    result,
                    expect=expect,
                )
                waited_s = round(time.monotonic() - verify_started, 1)
                self.log.write(
                    "verification",
                    ref=action_ref,
                    expected=expect.description,
                    observed=observed,
                    ok=verified,
                    error=verification_error,
                    waited_s=waited_s,
                    no_effect=not verified,
                    unknown=bool(result.get("unknown")),
                )
            if not success or not verified:
                self.stats["action_failures"] += 1
            if not self.config.dry_run:
                self._mark_action(
                    action.key,
                    success and verified,
                    failure_cooldown_s=(
                        0.0 if success and verified else ACTION_FAILURE_COOLDOWN_S
                    ),
                )
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
            if not success:
                outcome = "failed"
                return "failed"
            if not verified:
                outcome = "unknown" if result.get("unknown") else "failed"
                return outcome
            outcome = "succeeded"
            return "succeeded"
        except (DomeSessionRevoked, DomeConnectionError, DomeTimeout):
            outcome = "unknown"
            self.stats["action_failures"] += 1
            self.log.write(
                "action",
                request=request,
                response={"success": False, "error": "connection failure"},
                reason=action.reason,
                dry_run=self.config.dry_run,
                unsafe=decision.unsafe,
                ref=decision_ref,
                expect=expect.description,
            )
            if not self.config.dry_run:
                self._mark_action(
                    action.key,
                    False,
                    failure_cooldown_s=ACTION_FAILURE_COOLDOWN_S,
                )
            raise
        except Exception as exc:
            outcome = "failed"
            self.stats["action_failures"] += 1
            self._log_exception("action", exc, node_id=action.node_id, action=action.action)
            self.log.write(
                "action",
                request=request,
                response={"success": False, "error": f"{type(exc).__name__}: {exc}"},
                reason=action.reason,
                dry_run=self.config.dry_run,
                unsafe=decision.unsafe,
                ref=decision_ref,
                expect=expect.description,
            )
            if not self.config.dry_run:
                self._mark_action(
                    action.key,
                    False,
                    failure_cooldown_s=ACTION_FAILURE_COOLDOWN_S,
                )
            return "failed"
        finally:
            if intent_started and outcome is not None:
                try:
                    self.state.finish_action(action.key)
                except OSError as exc:
                    self._log_exception("state_intent", exc)
            self._release_action_key(lock_key)

    @staticmethod
    def _action_lock_key(action: ActionSpec) -> str:
        value = "" if action.value is NO_VALUE else repr(action.value)
        return (
            f"{str(action.node_id or '').strip().casefold()}:"
            f"{str(action.action or '').strip().casefold()}:{value}"
        )

    def _claim_action_key(self, key: str) -> bool:
        if key in self._pending_action_keys:
            return False
        self._pending_action_keys.add(key)
        return True

    def _release_action_key(self, key: str) -> None:
        self._pending_action_keys.discard(key)

    def _log_pending_block(
        self,
        action: ActionSpec,
        request: Mapping[str, Any],
        decision_ref: int | None,
        reason: str,
        expected: str,
    ) -> None:
        if action.key in self._blocked_seen:
            return
        self._blocked_seen.add(action.key)
        self.stats["actions_blocked"] += 1
        self.log.write(
            "action_blocked",
            request=request,
            reason=reason,
            unsafe=False,
            ref=decision_ref,
            expect=expected,
        )

    async def _verify_action(
        self,
        action: ActionSpec,
        result: Mapping[str, Any],
        *,
        expect: ExpectedEffect | None = None,
    ) -> tuple[bool, Any, str | None]:
        expected = expect or (
            action.expect
            if isinstance(action.expect, ExpectedEffect)
            else expected_effect(
                action.node_id,
                action.action,
                action.value,
                has_value=action.value is not NO_VALUE,
            )
        )
        after = result.get("after")
        confirmed = expected.check(after) if isinstance(after, Mapping) else False
        if confirmed is True:
            return True, expected.observe(after), None
        if confirmed is None:
            diff = result.get("diff")
            return (
                meaningful_diff(diff),
                diff if diff is not None else expected.observe(after),
                result.get("verify_error"),
            )
        last = after
        last_error = result.get("verify_error")
        deadline = time.monotonic() + max(0.1, expected.timeout_s)
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
            if expected.check(node) is True:
                return True, expected.observe(node), None
        return False, expected.observe(last), last_error or "telemetry did not confirm action"

    def _reconcile_pending_actions(self) -> None:
        """Block actions whose outcome was lost when the process stopped."""
        self._recovered_actions = []
        try:
            recovered = self.state.reconcile_pending_actions(
                ACTION_FAILURE_COOLDOWN_S
            )
        except OSError as exc:
            self._log_exception("state_reconcile", exc)
            return
        for key, entry in recovered.items():
            self._recovered_actions.append(
                {
                    "action": key,
                    "request": entry.get("request"),
                    "expected": entry.get("expected"),
                    "ref": entry.get("ref"),
                    "started_at": entry.get("at"),
                }
            )
            self.log.write(
                "action_reconcile",
                action=key,
                request=entry.get("request"),
                expected=entry.get("expected"),
                ref=entry.get("ref"),
                status="unknown",
                started_at=entry.get("at"),
            )

    def _mark_action(
        self,
        key: str,
        success: bool,
        *,
        failure_cooldown_s: float = 0.0,
    ) -> None:
        try:
            self.state.mark_action(
                key,
                success=success,
                failure_cooldown_s=failure_cooldown_s,
            )
        except OSError as exc:
            self._log_exception("state_action", exc)

    async def _notify_recovered_actions(self) -> None:
        """Warn the operator about commands whose outcome was lost on restart."""
        if not self._recovered_actions:
            return
        entries = self._recovered_actions
        labels: list[str] = []
        for entry in entries[:5]:
            request = entry.get("request")
            node = ""
            action = ""
            if isinstance(request, Mapping):
                node = str(request.get("node_id") or "").strip()
                action = str(request.get("action") or "").strip()
            key = str(entry.get("action") or "").strip()
            labels.append(f"{node}/{action}" if node and action else key or "?")
        suffix = f" и ещё {len(entries) - 5}" if len(entries) > 5 else ""
        text = (
            "После перезапуска есть команды с неизвестным исходом: "
            f"{', '.join(labels)}{suffix}. Проверьте состояние вручную."
        )
        try:
            delivered = await self._send_operator_message(
                text,
                dedupe_key="action_reconcile",
                min_interval_s=0.0,
            )
        except Exception as exc:
            self._log_exception("action_reconcile_notify", exc)
            return
        if delivered:
            self._recovered_actions = []

    async def _notify_action_failure(
        self,
        action: ActionSpec,
        outcome: str,
    ) -> None:
        """Tell the operator when a command was rejected or not confirmed."""
        verb = "не выполнена" if outcome == "failed" else "не подтверждена"
        text = (
            f"{action.node_id}: команда {action.action} {verb}. "
            "Проверьте состояние вручную."
        )
        try:
            await self._send_operator_message(
                text,
                dedupe_key=f"action:{action.key}:{outcome}",
                min_interval_s=300.0,
            )
        except Exception as exc:
            self._log_exception(
                "action_notify",
                exc,
                node_id=action.node_id,
                action=action.action,
            )

    async def _send_operator_message(
        self,
        text: str,
        *,
        dedupe_key: str,
        min_interval_s: float,
        to: str = "operator",
    ) -> bool:
        text = text.strip()
        if not text:
            return False
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
            return True
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
            return True
        await self._chat_throttle()
        try:
            await self.api.post_message(text, to=to, name=self.config.agent_name)
        except (DomeSessionRevoked, DomeConnectionError, DomeTimeout):
            raise
        except Exception as exc:
            self._log_exception("chat_post", exc)
            return False
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
        return True

    async def _chat_throttle(self) -> None:
        now = time.monotonic()
        while self._sent_message_times and now - self._sent_message_times[0] >= 10.0:
            self._sent_message_times.popleft()
        if len(self._sent_message_times) >= 10:
            wait = 10.0 - (now - self._sent_message_times[0])
            if wait > 0:
                await self._sleep(wait)

    # ------------------------------------------------------------------
    # operator confirmation and model control
    # ------------------------------------------------------------------

    async def _handle_pending(self) -> None:
        pending = self._pending
        if pending is None:
            return
        now = time.time()
        if pending.expired(now, PENDING_TTL_S):
            self._pending = None
            await self._send_operator_message(
                "Предложение устарело (5 минут). Ничего не выполнено.",
                dedupe_key=f"pending-expired:{_digest(pending.message)}",
                min_interval_s=0,
            )
            return

        remaining: list[ChatMessage] = []
        handled_reply = False
        for message in self._incoming:
            if not handled_reply:
                kind = _reply_kind(
                    message.text,
                    awaiting_force=bool(self._pending and self._pending.awaiting_force),
                )
                if kind == "cancel":
                    self._pending = None
                    handled_reply = True
                    await self._send_operator_message(
                        "Отменено. Команды не отправлены.",
                        dedupe_key=f"cancel:{message.id}",
                        min_interval_s=0,
                    )
                    continue
                if kind == "force":
                    handled_reply = True
                    await self._execute_pending(force=True)
                    continue
                if kind == "confirm":
                    handled_reply = True
                    await self._execute_pending(force=False)
                    continue
            remaining.append(message)
        self._incoming = remaining

    async def _execute_pending(self, *, force: bool) -> None:
        pending = self._pending
        if pending is None:
            return
        self.stats["confirmations"] += 1
        results = await self._execute_model_commands(
            pending.commands,
            force=force,
            source="confirmed_force" if force else "confirmed",
        )
        blocked = [
            (command, result)
            for command, result in results
            if result.get("blocked") and result.get("forceable")
        ]
        if blocked and not force:
            blocked_commands = tuple(command for command, _ in blocked)
            reasons = "; ".join(
                str(result.get("error") or "policy denied")[:200]
                for _, result in blocked
            )
            self._pending = PendingProposal(
                commands=blocked_commands,
                message=pending.message,
                created_at=time.time(),
                source=pending.source,
                awaiting_force=True,
                analysis=pending.analysis,
                risk=pending.risk,
            )
            await self._send_operator_message(
                "Заблокировано политикой: "
                f"{reasons}.\n"
                "Выполнить принудительно? (да, принудительно / отмена)",
                dedupe_key=f"force:{_digest(reasons)}",
                min_interval_s=0,
            )
            return
        self._pending = None
        summary = self._format_model_results(results)
        await self._send_operator_message(
            summary,
            dedupe_key=f"result:{_digest(summary)}",
            min_interval_s=0,
        )

    async def _execute_protective(
        self,
        action: ActionSpec,
        decision_ref: int | None,
    ) -> str:
        gate_decision = self.gate.check(action, dry_run=self.config.dry_run)
        if gate_decision.allowed:
            return await self._execute_action(action, decision_ref)
        key = f"protective:{action.key}"
        if key not in self._blocked_seen:
            self._blocked_seen.add(key)
            self.stats["actions_blocked"] += 1
            self.log.write(
                "action_blocked",
                request=action.request(),
                reason=gate_decision.reason,
                unsafe=False,
                source="protective",
                forceable=bool(self.config.control_enabled),
                ref=decision_ref,
            )
        if not self.config.control_enabled:
            return "blocked"
        command = ModelCommand.from_action(action)
        self._pending = PendingProposal(
            commands=(command,),
            message=(
                f"Защитное действие {action.node_id}.{action.action} "
                f"заблокировано: {gate_decision.reason}"
            ),
            created_at=time.time(),
            source="protective",
            awaiting_force=True,
        )
        self.stats["proposals"] += 1
        await self._send_operator_message(
            f"Защитное действие {action.node_id}.{action.action} заблокировано: "
            f"{gate_decision.reason}.\n"
            "Выполнить принудительно? (да, принудительно / отмена)",
            dedupe_key=f"protective-force:{_digest(action.key)}",
            min_interval_s=0,
        )
        return "blocked"

    async def _schedule_llm_control(
        self,
        snapshot: Snapshot,
        decision: RuleResult,
    ) -> None:
        if not self.advisor.available:
            if not self._llm_notice_sent:
                self._llm_notice_sent = True
                self.log.write(
                    "error",
                    where="llm_control",
                    error="LLM is not configured; no model commands will run",
                )
                await self._send_operator_message(
                    "LLM недоступна: новые команды не выполняю.",
                    dedupe_key="llm-unavailable",
                    min_interval_s=600,
                )
            self._incoming.clear()
            return
        if self._control_task is not None and not self._control_task.done():
            return
        force = bool(self._incoming)
        now = time.monotonic()
        interval = max(LLM_CONTROL_INTERVAL_S, self.config.llm_min_interval_s)
        if not force and (now - self._last_control_at) < interval:
            return
        messages = list(self._incoming)
        self._incoming.clear()
        self._last_control_at = now
        self._control_task = asyncio.create_task(
            self._run_llm_control(snapshot, decision, messages, force=force),
            name="dome-agent-control",
        )

    async def _run_llm_control(
        self,
        snapshot: Snapshot,
        decision: RuleResult,
        messages: list[ChatMessage],
        *,
        force: bool,
    ) -> None:
        try:
            context = await self._control_context(snapshot, decision, messages)
            model_decision = await self.advisor.decide(
                context,
                mode=self.config.mode,
                force=force or bool(messages),
                purpose="dome control",
            )
            if model_decision is None:
                if messages:
                    await self._send_operator_message(
                        "LLM не ответила; новые команды не выполняю.",
                        dedupe_key="llm-no-response",
                        min_interval_s=60,
                    )
                return
            self._log_model_decision(model_decision)
            if self.config.mode == "yolo":
                results = await self._execute_model_commands(
                    model_decision.commands,
                    force=False,
                    source="yolo",
                )
                self.stats["llm_commands"] += len(model_decision.commands)
                if model_decision.operator_message:
                    await self._send_operator_message(
                        model_decision.operator_message,
                        dedupe_key=f"llm:{_digest(model_decision.operator_message)}",
                        min_interval_s=0,
                    )
                if results:
                    summary = self._format_model_results(results)
                    if not model_decision.operator_message or any(
                        not result.get("ok") for _, result in results
                    ):
                        await self._send_operator_message(
                            summary,
                            dedupe_key=f"llm-result:{_digest(summary)}",
                            min_interval_s=0,
                        )
                return

            if model_decision.commands:
                pending = PendingProposal(
                    commands=model_decision.commands,
                    message=model_decision.operator_message,
                    created_at=time.time(),
                    source="llm",
                    analysis=model_decision.analysis,
                    risk=model_decision.risk,
                )
                if self._pending is not None:
                    # Never replace an operator-facing proposal behind their
                    # back. The pending item expires or is confirmed/cancelled.
                    if self._pending.signature != pending.signature:
                        await self._send_operator_message(
                            "Есть ожидающее предложение. Сначала подтвердите или "
                            "отмените его.",
                            dedupe_key="pending-busy",
                            min_interval_s=30,
                        )
                    return
                self._pending = pending
                self.stats["proposals"] += 1
                text = model_decision.operator_message or "Предлагаю выполнить действия."
                text += "\n\nПодтвердить? (да / выполняй; отмена)"
                await self._send_operator_message(
                    text,
                    dedupe_key=f"proposal:{_digest(text)}",
                    min_interval_s=0,
                )
            elif model_decision.operator_message:
                await self._send_operator_message(
                    model_decision.operator_message,
                    dedupe_key=f"llm:{_digest(model_decision.operator_message)}",
                    min_interval_s=0,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_exception("llm_control", exc)

    async def _control_context(
        self,
        snapshot: Snapshot,
        decision: RuleResult,
        messages: list[ChatMessage],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "mode": self.config.mode,
            "snapshot": snapshot.brief(),
            "rule_decision": decision.as_dict(),
            "task": self.state.task,
            "recent_chat": list(self._recent_chat),
            "new_operator_messages": [message.text[:1000] for message in messages],
            "pending": None,
            "hard_denied_actions": sorted(HARD_DENY_ACTIONS),
            "auto_protective": [
                "smart_panel_01.line_off",
                "printer_3d_01.stop_job",
                "cnc_01.stop_job",
                "solar_inverter_01.set_fan=ON",
                "supply_ventilation_01.turn_on",
                "fume_extraction_01.turn_on",
                "supply_ventilation_01.set_mode=FRESH_AIR",
                "any.turn_off",
            ],
            "policy": self.gate.allowed_summary(),
        }
        if self._pending is not None:
            pending = self._pending.as_dict()
            pending["age_s"] = round(time.time() - self._pending.created_at, 1)
            context["pending"] = pending
        try:
            catalog = await self.api.catalog()
        except Exception as exc:
            context["catalog_error"] = f"{type(exc).__name__}: {exc}"
        else:
            context["catalog"] = compact_catalog(catalog)
        return context

    def _log_model_decision(self, model_decision: ModelDecision) -> int:
        plan = [
            f"{command.tool}:{command.node_id}.{command.action}".strip(".")
            for command in model_decision.commands
        ]
        ref = self.log.write(
            "decision",
            situation=model_decision.analysis[:500] or "LLM decision",
            cause="",
            confidence=model_decision.confidence,
            evidence=[],
            plan=plan,
            source="llm",
            mode=self.config.mode,
            commands=[command.as_dict() for command in model_decision.commands],
            operator_message=model_decision.operator_message,
            risk=model_decision.risk,
            warnings=list(model_decision.warnings),
        )
        self.stats["decisions"] += 1
        self._last_decision_at = time.monotonic()
        return ref

    def _check_model_effect(
        self,
        action: ActionSpec,
        response: Mapping[str, Any],
    ) -> tuple[bool, Any, str | None]:
        expect = (
            action.expect
            if isinstance(action.expect, ExpectedEffect)
            else expected_effect(
                action.node_id,
                action.action,
                action.value,
                has_value=action.value is not NO_VALUE,
            )
        )
        after = response.get("after")
        confirmed = expect.check(after) if isinstance(after, Mapping) else False
        if confirmed is True:
            return True, expect.observe(after), None
        if confirmed is None:
            diff = response.get("diff")
            return (
                meaningful_diff(diff),
                diff if diff is not None else expect.observe(after),
                response.get("verify_error"),
            )
        return (
            False,
            expect.observe(after),
            response.get("verify_error") or "telemetry did not confirm action",
        )

    async def _execute_model_commands(
        self,
        commands: tuple[ModelCommand, ...],
        *,
        force: bool,
        source: str,
    ) -> list[tuple[ModelCommand, dict[str, Any]]]:
        results: list[tuple[ModelCommand, dict[str, Any]]] = []
        for command in commands:
            results.append(
                (
                    command,
                    await self._execute_model_command(
                        command,
                        force=force,
                        source=source,
                    ),
                )
            )
        return results

    async def _execute_model_command(
        self,
        command: ModelCommand,
        *,
        force: bool,
        source: str,
    ) -> dict[str, Any]:
        request = command.as_dict()
        unsafe = bool(self.config.yolo or force)
        action: ActionSpec | None = None
        if command.tool == "control":
            if not self.config.control_enabled and not self.config.yolo:
                return self._block_model(
                    command,
                    "control disabled: set AGENT_ALLOW_CONTROL=1",
                    source=source,
                    forceable=False,
                    unsafe=unsafe,
                )
            if command.action.casefold() in HARD_DENY_ACTIONS:
                return self._block_model(
                    command,
                    f"action {command.action!r} is hard-denied",
                    source=source,
                    forceable=False,
                    unsafe=unsafe,
                )
            action = ActionSpec(
                key=f"llm:{_digest(command.signature)}",
                node_id=command.node_id,
                action=command.action,
                value=command.value,
                reason=command.reason,
                expect=(
                    command.expect
                    if isinstance(command.expect, ExpectedEffect)
                    else expected_effect(
                        command.node_id,
                        command.action,
                        command.value,
                        has_value=command.value is not NO_VALUE,
                    )
                ),
            )
            if not force and not self.config.yolo:
                gate_decision = self.gate.check(action, dry_run=self.config.dry_run)
                if not gate_decision.allowed:
                    return self._block_model(
                        command,
                        gate_decision.reason,
                        source=source,
                        forceable=True,
                        unsafe=bool(gate_decision.unsafe),
                    )
            if (
                not force
                and not self.config.yolo
                and not self.config.dry_run
                and self.state.action_failed_recently(action.key)
            ):
                return self._block_model(
                    command,
                    "recent verification failed: retry cooldown is active",
                    source=source,
                    forceable=True,
                    unsafe=unsafe,
                )
            lock_key = self._action_lock_key(action)
            if not self._claim_action_key(lock_key):
                return self._block_model(
                    command,
                    "action is already pending verification",
                    source=source,
                    forceable=False,
                    unsafe=unsafe,
                )
            try:
                result = await self.tool.execute(
                    command,
                    force=force or self.config.yolo,
                    verify=not self.config.yolo,
                    dry_run=self.config.dry_run,
                )
            finally:
                self._release_action_key(lock_key)
        elif command.tool == "send_message" and self.config.dry_run:
            result = {
                "ok": True,
                "command": request,
                "result": {"dry_run": True, "sent": False},
            }
        else:
            result = await self.tool.execute(command, dry_run=self.config.dry_run)

        response = result.get("result")
        if not isinstance(response, Mapping):
            response = {
                "success": bool(result.get("ok")),
                "error": result.get("error"),
            }
        else:
            response = dict(response)

        result_ok = bool(result.get("ok"))
        ok = result_ok
        verified = True
        verification_observed: Any = None
        verification_error = response.get("verify_error")
        if (
            action is not None
            and result_ok
            and not self.config.yolo
            and not self.config.dry_run
        ):
            verified, verification_observed, verification_error = self._check_model_effect(
                action,
                response,
            )
            ok = result_ok and verified
            response["verified"] = verified
            response["observed"] = verification_observed
            response["verify_error"] = verification_error
        if ok:
            self.stats["actions"] += 1
        else:
            self.stats["action_failures"] += 1
        log_response = {
            key: value
            for key, value in response.items()
            if key not in {"after", "diff"}
        }
        action_ref = self.log.write(
            "action",
            request=request,
            response=log_response,
            reason=command.reason,
            dry_run=self.config.dry_run,
            unsafe=unsafe,
            source=source,
            force=force,
            expect=getattr(action.expect, "description", None) if action is not None else None,
        )
        if (
            action is not None
            and result_ok
            and not self.config.yolo
            and not self.config.dry_run
        ):
            self.log.write(
                "verification",
                ref=action_ref,
                expected=getattr(action.expect, "description", None),
                observed=verification_observed,
                ok=verified,
                error=verification_error,
                waited_s=0.0,
                no_effect=not verified,
            )
        if action is not None and not self.config.dry_run:
            self._mark_action(
                action.key,
                ok,
                failure_cooldown_s=0.0 if ok else ACTION_FAILURE_COOLDOWN_S,
            )
        if not ok and verification_error:
            result["error"] = verification_error
        result["ok"] = ok
        result["result"] = response
        return result

    def _block_model(
        self,
        command: ModelCommand,
        reason: str,
        *,
        source: str,
        forceable: bool,
        unsafe: bool,
    ) -> dict[str, Any]:
        self.stats["actions_blocked"] += 1
        self.log.write(
            "action_blocked",
            request=command.as_dict(),
            reason=reason,
            unsafe=unsafe,
            source=source,
            forceable=forceable,
        )
        return {
            "ok": False,
            "blocked": True,
            "forceable": forceable,
            "error": reason,
            "command": command.as_dict(),
        }

    def _format_model_results(
        self,
        results: list[tuple[ModelCommand, dict[str, Any]]],
    ) -> str:
        if not results:
            return "Команды не требуются."
        lines: list[str] = []
        for command, result in results:
            label = f"{command.tool}:{command.node_id}.{command.action}".strip(":")
            if result.get("ok"):
                status = "OK"
            elif result.get("blocked"):
                status = "BLOCKED"
            else:
                status = "FAIL"
            error = str(result.get("error") or "")[:300]
            lines.append(f"{status} {label}" + (f": {error}" if error else ""))
        return "Результат:\n" + "\n".join(lines[:8])

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
        description="Autonomous Dome shift agent (LLM-controlled normal/YOLO modes).",
    )
    parser.add_argument("--env-file", default=".env", help="optional .env file (default: .env)")
    parser.add_argument("--url", default=None, help="override DOME_URL")
    parser.add_argument("--name", default=None, help="override AGENT_NAME")
    parser.add_argument("--log-dir", default=None, help="override AGENT_LOG_DIR")
    parser.add_argument("--state-file", default=None, help="override AGENT_STATE_FILE")
    parser.add_argument("--dry-run", action="store_true", default=None, help="analyze and log but send no commands/messages")
    parser.add_argument("--allow-control", action="store_true", default=None, help="allow control commands (normal/auto modes)")
    parser.add_argument(
        "--mode",
        choices=("normal", "auto", "yolo"),
        default=None,
        help="normal: LLM proposals + confirmation; auto: rules only; yolo: LLM executes immediately",
    )
    parser.add_argument("--yolo", action="store_true", default=None, help="UNSAFE alias for --mode yolo")
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
    if args.allow_control is not None:
        overrides["control_enabled"] = True
    if args.mode:
        overrides["mode"] = args.mode
    if args.yolo is not None:
        overrides["mode"] = "yolo"
    if args.run_seconds is not None:
        overrides["run_seconds"] = args.run_seconds
    try:
        config = AgentConfig.from_env(env_file=args.env_file, overrides=overrides)
    except Exception as exc:
        print(f"agent configuration error: {exc}", file=sys.stderr)
        return 2
    if config.yolo:
        print(
            "WARNING: YOLO mode enabled: the LLM executes commands immediately; "
            "policy, cooldowns and verification are bypassed. Hard-denied actions "
            "are still blocked. Use only on a test dome.",
            file=sys.stderr,
        )
    try:
        asyncio.run(run_agent(config))
    except KeyboardInterrupt:
        return 130
    return 0


