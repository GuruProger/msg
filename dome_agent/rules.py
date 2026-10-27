"""Deterministic safety and production rules.

The LLM is advisory only.  This module is the authority for actions: every
command has a reason, a cooldown key, and an explicit safety precondition.
The agent logs the returned decision and then executes/verifies the actions.
"""

from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .analysis import Snapshot, as_bool, as_float, as_int, normalize_interlocks
from .chat import check_cnc_completion, check_print_completion
from .state import StateStore

NO_VALUE = object()

_PRODUCTION_BLOCKING_ALERTS = frozenset({
    "MACHINE_ERROR",
    "MACHINE_ALARM",
    "MACHINE_HOLD_ALARM",
    "FIRE",
    "FIRE_NOT_READY",
    "SMOKE",
    "CONDENSATE",
    "CO2_TRIP",
    "SEISMIC",
    "CHEM",
    "RADIATION",
    "FUME_ALARM",
    "SEAL_LEAK",
    "INVERTER_ERROR",
    "INVERTER_HEAT",
    "OVERVOLTAGE",
    "PANEL_LINE_ALARM",
    "BATTERY_LOW",
    "ACCESS",
})


def _state(value: Any) -> str:
    return str(value or "").strip().upper()


def _file_names(node: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for item in node.get("files") or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("file_name")
        else:
            name = item
        if name:
            names.append(str(name))
    return names


def _file_present(node: Mapping[str, Any], filename: str) -> bool:
    target = filename.casefold()
    return any(name.casefold() == target for name in _file_names(node))


def _interlock(snapshot: Snapshot, name: str) -> dict[str, Any]:
    interlocks = normalize_interlocks(
        snapshot.node("dome_automation_01").get("interlocks")
    )
    return interlocks.get(str(name), {})


def _interlock_state(snapshot: Snapshot, name: str) -> str:
    return _state(_interlock(snapshot, name).get("state"))


def _interlock_known_tripped(snapshot: Snapshot, name: str) -> bool:
    return _interlock_state(snapshot, name) in {"TRIPPED", "BLOCKED", "ALARM"}


def _interlock_tripped(snapshot: Snapshot, name: str) -> bool:
    """Return True unless the interlock is explicitly in a safe state.

    The API has been observed returning interlocks as a list of objects.  A
    malformed or unknown shape must fail closed; treating it as an empty map
    would incorrectly clear IL_CO2 and allow a resume.
    """
    raw = snapshot.node("dome_automation_01").get("interlocks")
    if raw is None:
        return True
    if not isinstance(raw, (Mapping, list, tuple)):
        return True
    if isinstance(raw, (list, tuple)) and raw and not normalize_interlocks(raw):
        return True
    interlock = _interlock(snapshot, name)
    if not interlock:
        return True
    return _state(interlock.get("state")) not in {"ARMED", "CLEAR", "OK", "NORMAL"}


def _powder_present(snapshot: Snapshot) -> bool:
    for node_id in ("smoke_detector_01", "smoke_detector_02"):
        node = snapshot.node(node_id)
        if as_bool(node.get("smoke_detected")):
            return True
        if _state(node.get("alarm_state")) not in {"", "NORMAL"}:
            return True
    return False


def _smoke_data_known(snapshot: Snapshot) -> bool:
    for node_id in ("smoke_detector_01", "smoke_detector_02"):
        if not snapshot.online(node_id):
            return False
        if snapshot.node(node_id).get("smoke_detected") is None:
            return False
    return True


def _fire_ready(snapshot: Snapshot) -> bool:
    if not snapshot.online("fire_suppression_01"):
        return False
    fire = snapshot.node("fire_suppression_01")
    return (
        not as_bool(fire.get("alarm"))
        and not as_bool(fire.get("automation_blocked"))
        and as_bool(fire.get("ready"))
        and _state(fire.get("state")) == "READY"
    )


def _power_ready(snapshot: Snapshot) -> bool:
    if not snapshot.online("smart_panel_01"):
        return False
    panel = snapshot.node("smart_panel_01")
    if panel.get("bus_powered") is False:
        return False
    voltage = as_float(panel.get("bus_voltage_v"))
    return voltage is not None and voltage >= 200.0


def _condensate_present(snapshot: Snapshot) -> bool:
    climate = snapshot.node("climate_sensor_01")
    humidity = as_float(climate.get("humidity_pct"))
    temperature = as_float(climate.get("temperature_c"))
    dew = as_float(climate.get("dew_point_c"))
    if humidity is not None and humidity > 90:
        return True
    return (
        temperature is not None
        and dew is not None
        and (temperature - dew) < 1.0
    )


def _climate_data_known(snapshot: Snapshot) -> bool:
    if not snapshot.online("climate_sensor_01"):
        return False
    climate = snapshot.node("climate_sensor_01")
    return (
        as_float(climate.get("humidity_pct")) is not None
        and as_float(climate.get("temperature_c")) is not None
        and as_float(climate.get("dew_point_c")) is not None
    )


def _air_quality_block(snapshot: Snapshot) -> tuple[str, str] | None:
    if not snapshot.online("air_quality_sensor_01"):
        return "air_unknown", "Датчик воздуха недоступен: запуск производства запрещён."
    air = snapshot.node("air_quality_sensor_01")
    co = as_float(air.get("co_ppm"))
    voc = as_float(air.get("voc_index"))
    if as_bool(air.get("warming_up")) or co is None or voc is None:
        return "air_unknown", "Датчик воздуха не готов или его показания неизвестны: запуск производства запрещён."
    if co > 10.0:
        return "co_high", f"CO {co:.2f} ppm: запуск производства запрещён."
    if voc > 400.0:
        return "voc_high", f"VOC index {voc:.0f}: запуск производства запрещён."
    return None


def _diesel_ready(snapshot: Snapshot) -> bool:
    if not snapshot.online("dizel_1"):
        return False
    diesel = snapshot.node("dizel_1")
    if diesel.get("fault_code") or diesel.get("fault_reason"):
        return False
    return _state(diesel.get("state")) not in {"FAULT", "ERROR", "OFFLINE", "BLOCKED"}


def _fuel_quality_ok(snapshot: Snapshot) -> bool:
    if not snapshot.online("fuel_tank_01"):
        return False
    tank = snapshot.node("fuel_tank_01")
    quality = _state(tank.get("fuel_quality"))
    state = _state(tank.get("state"))
    return quality in {"", "GOOD", "OK", "NORMAL"} and state in {"", "NORMAL", "OK"}


def _fume_ready(snapshot: Snapshot) -> bool:
    if not snapshot.online("fume_extraction_01"):
        return False
    fume = snapshot.node("fume_extraction_01")
    if not as_bool(fume.get("enabled")) or as_bool(fume.get("alarm")):
        return False
    if _state(fume.get("motor_state")) in {"FAULT", "ERROR", "OFFLINE"}:
        return False
    airflow = as_float(fume.get("airflow_pct"))
    return airflow is not None and airflow >= 50.0


def _choose_alt_co2(snapshot: Snapshot) -> str | None:
    suspect_ids = set(snapshot.co2.suspect_ids)
    suspect_zones = {
        reading.zone
        for reading in snapshot.co2.readings
        if reading.node_id in suspect_ids and reading.zone
    }
    candidates = [
        reading
        for reading in snapshot.co2.readings
        if reading.valid
        and reading.current
        and reading.node_id not in suspect_ids
        and (not suspect_zones or reading.zone in suspect_zones)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda reading: (reading.confidence or 0.0, -abs((reading.ppm or 0.0) - (snapshot.co2.median_ppm or 0.0))), reverse=True)
    return candidates[0].node_id


def _line(snapshot: Snapshot, number: int) -> dict[str, Any]:
    return snapshot.line(number)


def _line_blocked(snapshot: Snapshot, number: int) -> bool:
    return (as_float(_line(snapshot, number).get("lockout_remaining_s"), 0.0) or 0.0) > 0


def _line_tripped(snapshot: Snapshot, number: int) -> bool:
    return _state(_line(snapshot, number).get("protection_state")) in {"TRIPPED", "RCD_TRIPPED"}


def _line_alarmed(snapshot: Snapshot, number: int) -> bool:
    return as_bool(_line(snapshot, number).get("alarm"))


def _line_ready(snapshot: Snapshot, number: int) -> bool:
    line = _line(snapshot, number)
    return bool(line) and not _line_blocked(snapshot, number) and not _line_alarmed(snapshot, number) and not _line_tripped(snapshot, number)


def _machine(snapshot: Snapshot, node_id: str) -> dict[str, Any]:
    return snapshot.node(node_id)


def _machine_closed(snapshot: Snapshot, node_id: str) -> bool:
    node = _machine(snapshot, node_id)
    return _state(node.get("state")) in {"COMPLETED", "IDLE"} and not node.get("paused") and not node.get("hold_level")


def _file_for_item(item: Mapping[str, Any], node: Mapping[str, Any]) -> str | None:
    explicit = str(item.get("file") or "").strip()
    if explicit:
        return explicit
    files = _file_names(node)
    # A single available file is unambiguous; never guess between several.
    return files[0] if len(files) == 1 else None


def _task_items(task: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(task, Mapping):
        return []
    items = task.get("items")
    if not isinstance(items, list):
        return []
    return [copy.deepcopy(item) for item in items if isinstance(item, dict)]


def _set_items(task: Mapping[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(task))
    updated["items"] = items
    return updated


def _mark(items: list[dict[str, Any]], index: int, status: str, **fields: Any) -> bool:
    item = items[index]
    changed = item.get("status") != status or any(item.get(key) != value for key, value in fields.items())
    item["status"] = status
    item.update(fields)
    return changed


def _new_status(item: Mapping[str, Any]) -> str:
    return str(item.get("status") or "pending")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    key: str
    node_id: str
    action: str
    value: Any = NO_VALUE
    reason: str = ""
    verify: bool = True
    priority: int = 50
    depends_on: tuple[str, ...] = ()
    expect: Any | None = field(default=None, compare=False, repr=False)

    def request(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "type": "control",
            "node_id": self.node_id,
            "action": self.action,
        }
        if self.value is not NO_VALUE:
            request["value"] = self.value
        return request

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "node_id": self.node_id,
            "action": self.action,
            "value": None if self.value is NO_VALUE else self.value,
            "reason": self.reason,
            "verify": self.verify,
            "priority": self.priority,
            "depends_on": list(self.depends_on),
            "expect": getattr(self.expect, "description", None),
        }


@dataclass(frozen=True, slots=True)
class OutgoingMessage:
    text: str
    dedupe_key: str
    to: str = "operator"
    min_interval_s: float = 900.0


@dataclass(slots=True)
class RuleResult:
    situation: str
    cause: str
    confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    actions: list[ActionSpec] = field(default_factory=list)
    messages: list[OutgoingMessage] = field(default_factory=list)
    source: str = "rules"
    task: dict[str, Any] | None = None
    task_changed: bool = False
    rag_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "situation": self.situation,
            "cause": self.cause,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "plan": self.plan,
            "source": self.source,
            "actions": [action.as_dict() for action in self.actions],
            "messages": [message.text for message in self.messages],
            "rag_refs": self.rag_refs,
        }


class RuleEngine:
    """Stateful rule engine with cooldown guards.

    ``state`` is optional for unit tests.  When supplied, it prevents the same
    corrective action from being retried on every telemetry frame.
    """

    def __init__(
        self,
        state: StateStore | None = None,
        *,
        clock=time.time,
        yolo: bool = False,
    ) -> None:
        self.state = state
        self.clock = clock
        self.yolo = bool(yolo)

    def _allowed(self, key: str, cooldown_s: float) -> bool:
        if self.yolo:
            return True
        if self.state is None:
            return True
        now = self.clock()
        if self.state.action_failed_recently(key, now=now):
            return False
        return self.state.action_allowed(key, cooldown_s, now=now)

    def _action(
        self,
        key: str,
        node_id: str,
        action: str,
        *,
        value: Any = NO_VALUE,
        reason: str,
        cooldown_s: float = 0.0,
        verify: bool = True,
        priority: int = 50,
        depends_on: Iterable[str] = (),
    ) -> ActionSpec | None:
        if not self._allowed(key, cooldown_s):
            return None
        from .verification import expected_effect

        return ActionSpec(
            key=key,
            node_id=node_id,
            action=action,
            value=value,
            reason=reason,
            verify=verify,
            priority=priority,
            depends_on=tuple(depends_on),
            expect=expected_effect(
                node_id,
                action,
                value,
                has_value=value is not NO_VALUE,
            ),
        )

    def decide(
        self,
        snapshot: Snapshot,
        task: Mapping[str, Any] | None = None,
    ) -> RuleResult:
        safety = self._safety(snapshot, task)
        if safety is not None:
            return safety

        task_result = self._task(snapshot, task)
        if task_result is not None:
            return task_result

        return RuleResult(
            situation="Нет требующих вмешательства событий",
            cause="все контролируемые параметры в норме",
            confidence=0.9,
            plan=["продолжать наблюдение"],
            source="rules",
        )

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _safety(self, snapshot: Snapshot, task: Mapping[str, Any] | None = None) -> RuleResult | None:
        evidence: list[dict[str, Any]] = []

        # Fire/smoke has absolute priority: no automatic production resume.
        if _powder_present(snapshot):
            return RuleResult(
                situation="Обнаружен дым",
                cause="срабатывание дымового датчика",
                confidence=0.99,
                evidence=[{"node": "smoke_detector", "param": "smoke_detected", "value": True}],
                plan=["не запускать и не возобновлять производство", "сообщить Оператору"],
                messages=[OutgoingMessage("Дым в куполе. Новые запуски и автоматическое возобновление заблокированы; нужна проверка Оператора.", "smoke", min_interval_s=120)],
            )

        # False CO2 alarm: mark the outlier unreliable, switch primary sensor,
        # then release the interlock.  This must happen before any resume.
        false_alarm = snapshot.co2.false_alarm and snapshot.co2.suspect_ids
        if false_alarm:
            suspect = snapshot.co2.suspect_ids[0]
            alternate = _choose_alt_co2(snapshot)
            actions: list[ActionSpec] = []
            trust_key = f"co2_trust:{suspect}"
            trust_action = self._action(
                trust_key,
                "dome_automation_01",
                "set_data_trust",
                value={"node": suspect, "param": "co2_ppm", "trust": "UNRELIABLE"},
                reason=f"{suspect} даёт изолированное значение CO2 ({(snapshot.co2.chosen_ppm or 0):.0f} ppm у большинства)",
                cooldown_s=1800,
                priority=100,
            )
            if trust_action:
                actions.append(trust_action)
            primary_action: ActionSpec | None = None
            if alternate:
                primary_action = self._action(
                    f"co2_primary:{alternate}",
                    "dome_automation_01",
                    "set_primary_sensor",
                    value=alternate,
                    reason=f"переключить основной датчик CO2 на {alternate}",
                    cooldown_s=1800,
                    priority=100,
                    depends_on=(trust_action.key,) if trust_action else (),
                )
                if primary_action:
                    actions.append(primary_action)
            reset_action: ActionSpec | None = None
            if _interlock_known_tripped(snapshot, "IL_CO2") and primary_action is not None:
                reset_action = self._action(
                    "co2_reset_interlock",
                    "dome_automation_01",
                    "reset_interlock",
                    value="IL_CO2",
                    reason="ложное срабатывание IL_CO2, сброс только после подтверждённой смены источника",
                    cooldown_s=300,
                    priority=100,
                    depends_on=(primary_action.key,),
                )
                if reset_action:
                    actions.append(reset_action)
            if actions:
                if reset_action is not None:
                    message = (
                        f"Датчик {suspect} дал ложное значение CO2; переключаю основной "
                        f"источник на {alternate} и сниму IL_CO2 только после подтверждения "
                        "переключения."
                    )
                elif primary_action is not None:
                    message = (
                        f"Датчик {suspect} дал ложное значение CO2; переключаю основной "
                        f"источник на {alternate}. IL_CO2 автоматически не снимаю."
                    )
                else:
                    message = (
                        f"Датчик {suspect} дал ложное значение CO2; альтернативного "
                        "источника нет, IL_CO2 не сбрасываю автоматически."
                    )
                return RuleResult(
                    situation="IL_CO2 мог сработать по ложному датчику",
                    cause=f"изолированное показание {suspect}",
                    confidence=0.9,
                    evidence=[
                        {"node": reading.node_id, "param": "co2_ppm", "value": reading.ppm}
                        for reading in snapshot.co2.readings
                    ],
                    plan=[action.reason for action in actions],
                    actions=actions,
                    messages=[OutgoingMessage(message, "co2_false_alarm")],
                )

        # A real CO2 event: improve ventilation, but never resume production.
        if snapshot.co2.real_event or (snapshot.co2.chosen_ppm or 0) >= 2000:
            actions = []
            vent = snapshot.node("supply_ventilation_01")
            if _state(vent.get("state")) != "ON":
                action = self._action(
                    "co2_vent_on",
                    "supply_ventilation_01",
                    "turn_on",
                    reason="CO2 выше 2000 ppm",
                    cooldown_s=300,
                    priority=95,
                )
                if action:
                    actions.append(action)
            if _state(vent.get("mode")) != "FRESH_AIR":
                action = self._action(
                    "co2_fresh_air",
                    "supply_ventilation_01",
                    "set_mode",
                    value="FRESH_AIR",
                    reason="перевести вентиляцию на свежий воздух при CO2",
                    cooldown_s=600,
                    priority=94,
                )
                if action:
                    actions.append(action)
            confirmation = (
                "подтверждён несколькими датчиками"
                if snapshot.co2.real_event
                else "не подтверждён вторым датчиком"
            )
            if actions:
                vent_text = "Усиливаю вентиляцию"
            elif _state(vent.get("state")) == "ON" and _state(vent.get("mode")) == "FRESH_AIR":
                vent_text = "Вентиляция уже работает на свежем воздухе"
            else:
                vent_text = "Вентиляция требует усиления"
            return RuleResult(
                situation="CO2 выше порога",
                cause="подтверждённое повышение CO2",
                confidence=0.95,
                evidence=[
                    {"node": reading.node_id, "param": "co2_ppm", "value": reading.ppm}
                    for reading in snapshot.co2.readings
                ],
                plan=["усилить вентиляцию", "не возобновлять станки до снижения CO2"],
                actions=actions,
                messages=[OutgoingMessage(
                    f"CO2 {snapshot.co2.chosen_ppm:.0f} ppm {confirmation}. {vent_text}; станки остаются на паузе.",
                    "co2_real",
                    min_interval_s=300,
                )],
            )

        # If CO2 is safe again but IL_CO2 is still tripped (the usual state
        # after a false alarm was fixed), release it before resuming.
        co2_ppm = snapshot.co2.chosen_ppm
        if (
            _interlock_known_tripped(snapshot, "IL_CO2")
            and not snapshot.co2.false_alarm
            and co2_ppm is not None
            and co2_ppm < 1800
        ):
            action = self._action(
                "co2_reset_interlock_after_clear",
                "dome_automation_01",
                "reset_interlock",
                value="IL_CO2",
                reason="CO2 в норме, IL_CO2 можно перевзвести",
                cooldown_s=120,
                priority=90,
            )
            if action:
                return RuleResult(
                    situation="IL_CO2 остался после нормализации CO2",
                    cause="интерлок требует явного сброса",
                    confidence=0.9,
                    evidence=[{"node": "dome_automation_01", "param": "IL_CO2.state", "value": _state(_interlock(snapshot, "IL_CO2").get("state"))}],
                    plan=["снять IL_CO2", "после проверки возобновить станки"],
                    actions=[action],
                )

        # Resume machines only when the reason is known to be gone.
        resume_actions: list[ActionSpec] = []
        resume_messages: list[OutgoingMessage] = []
        for node_id in ("printer_3d_01", "cnc_01"):
            node = _machine(snapshot, node_id)
            if not node or not snapshot.online(node_id):
                continue
            paused = as_bool(node.get("paused")) or _state(node.get("state")) == "PAUSED"
            hold = node.get("hold_level")
            lockout = as_float(node.get("lockout_remaining_s"), 0.0) or 0.0
            if lockout > 0:
                continue
            if not paused and not hold:
                continue
            reason = _state(node.get("hold_reason"))
            clear, why = self._hold_cleared(snapshot, reason)
            if not clear:
                if reason in {"BED_COLLISION", "FILAMENT_RUNOUT", "IL_SMOKE"}:
                    resume_messages.append(OutgoingMessage(
                        f"{node_id}: {reason}. Нужно вмешательство Оператора, автоматический resume запрещён.",
                        f"{node_id}:{reason}",
                        min_interval_s=1800,
                    ))
                continue
            attempts = as_int(node.get("resume_attempts"), 0) or 0
            if attempts >= 3:
                resume_messages.append(OutgoingMessage(
                    f"{node_id}: лимит resume исчерпан ({attempts}/3), станок в ALARM.",
                    f"{node_id}:resume_limit",
                    min_interval_s=1800,
                ))
                continue
            action = self._action(
                f"resume:{node_id}:{reason or 'manual'}",
                node_id,
                "resume",
                reason=f"{node_id}: причина удержания устранена ({why})",
                cooldown_s=90,
                priority=85,
            )
            if action:
                resume_actions.append(action)
        if resume_actions:
            return RuleResult(
                situation="Станки удержаны, причина удержания исчезла",
                cause="сервер не снимает паузу автоматически",
                confidence=0.9,
                evidence=[
                    {"node": action.node_id, "param": "hold_reason", "value": _machine(snapshot, action.node_id).get("hold_reason")}
                    for action in resume_actions
                ],
                plan=[action.reason for action in resume_actions],
                actions=resume_actions,
                messages=resume_messages,
            )
        if resume_messages:
            return RuleResult(
                situation="Станок требует Оператора",
                cause="удержание нельзя снять автоматически",
                confidence=0.9,
                plan=["не пытаться resume без устранения причины"],
                messages=resume_messages,
            )

        # Inverter E09: do not reset in wet weather.  Wait for a dry window.
        inverter = snapshot.node("solar_inverter_01")
        error_code = str(inverter.get("error_code") or "").upper()
        if error_code:
            weather = snapshot.node("weather_station_01")
            humidity = as_float(snapshot.node("climate_sensor_01").get("humidity_pct"), 100.0) or 100.0
            rain = as_float(weather.get("precipitation_mm_h"), 0.0) or 0.0
            temp = as_float(snapshot.node("climate_sensor_01").get("temperature_c"))
            dew = as_float(snapshot.node("climate_sensor_01").get("dew_point_c"))
            dry = rain <= 0.0 and humidity < 85.0 and (temp is None or dew is None or temp - dew > 2.0)
            if dry:
                action = self._action(
                    f"inverter_reset:{error_code}",
                    "solar_inverter_01",
                    "reset_error",
                    reason=f"ошибка {error_code}, погода сухая",
                    cooldown_s=900,
                    priority=70,
                )
                if action:
                    return RuleResult(
                        situation=f"Инвертор в ошибке {error_code}",
                        cause="ошибка инвертора; сухая погода позволяет сброс",
                        confidence=0.75,
                        evidence=[{"node": "solar_inverter_01", "param": "error_code", "value": error_code}],
                        plan=["сбросить ошибку", "проверить PV и АКБ"],
                        actions=[action],
                    )
            return RuleResult(
                situation=f"Инвертор в ошибке {error_code}",
                cause="влажно/сыро; сброс E09 может повториться",
                confidence=0.85,
                evidence=[{"node": "solar_inverter_01", "param": "error_code", "value": error_code}],
                plan=["ждать сухую погоду", "не сбрасывать ошибку вслепую"],
                messages=[OutgoingMessage(
                    f"Инвертор: {error_code}. Жду сухую погоду перед сбросом, производство под риском.",
                    f"inverter:{error_code}",
                )],
            )

        # Inverter overheat: force the fan on once, then wait for cooling.
        inv_temp = as_float(inverter.get("inverter_temp_c"))
        if inv_temp is not None and inv_temp > 75:
            action = self._action(
                "inverter_fan_on",
                "solar_inverter_01",
                "set_fan",
                value="ON",
                reason=f"температура инвертора {inv_temp:.1f} C",
                cooldown_s=300,
                priority=80,
            )
            if action:
                return RuleResult(
                    situation="Перегрев инвертора",
                    cause="высокая температура",
                    confidence=0.9,
                    evidence=[{"node": "solar_inverter_01", "param": "inverter_temp_c", "value": inv_temp}],
                    plan=["включить вентилятор", "снизить нагрузку при сохранении перегрева"],
                    actions=[action],
                )

        # Phantom reserve line: do not leave an unexplained 50 W load on.
        line8 = _line(snapshot, 8)
        phantom = as_float(line8.get("power_w"), 0.0) or 0.0
        if _state(line8.get("state")) == "ON" and phantom >= 20.0 and _line_ready(snapshot, 8):
            action = self._action(
                "line8_phantom_off",
                "smart_panel_01",
                "line_off",
                value=8,
                reason=f"резервная линия 8 потребляет {phantom:.0f} Вт без назначенной нагрузки",
                cooldown_s=900,
                priority=60,
            )
            if action:
                return RuleResult(
                    situation="Фантомное потребление на линии 8",
                    cause="возможно окисление/влажный контакт",
                    confidence=0.8,
                    evidence=[{"node": "smart_panel_01", "param": "lines[7].power_w", "value": phantom}],
                    plan=["отключить линию 8", "сообщить Оператору о необходимости осмотра"],
                    actions=[action],
                    messages=[OutgoingMessage(
                        f"Линия 8 потребляет {phantom:.0f} Вт без нагрузки. Отключаю; проверьте контакт.",
                        "line8_phantom",
                    )],
                )

        # Restore a line required by a task only after protection is re-armed.
        for line_number in (3, 4):
            if not self._line_needed(snapshot, task, line_number):
                continue
            line = _line(snapshot, line_number)
            if not line:
                continue
            if _line_alarmed(snapshot, line_number):
                return RuleResult(
                    situation=f"Авария линии {line_number}",
                    cause=f"alarm_code={line.get('alarm_code')}",
                    confidence=0.9,
                    evidence=[{"node": "smart_panel_01", "param": f"lines[{line_number - 1}].alarm_code", "value": line.get("alarm_code")}],
                    plan=["не включать линию до устранения причины", "сообщить Оператору"],
                    messages=[OutgoingMessage(
                        f"Линия {line_number} в аварии ({line.get('alarm_code')}). Нужен осмотр Оператора.",
                        f"line{line_number}:alarm",
                    )],
                )
            if _line_tripped(snapshot, line_number):
                action = self._action(
                    f"line_reset:{line_number}",
                    "smart_panel_01",
                    "reset_protection",
                    value=line_number,
                    reason=f"линия {line_number} сработала по защите",
                    cooldown_s=300,
                    priority=74,
                )
                if action:
                    return RuleResult(
                        situation=f"Линия {line_number} отключена защитой",
                        cause=f"protection={line.get('protection_state')}",
                        confidence=0.85,
                        evidence=[{"node": "smart_panel_01", "param": f"lines[{line_number - 1}].protection_state", "value": line.get("protection_state")}],
                        plan=["взвести защиту", "затем включить линию"],
                        actions=[action],
                    )
            elif _state(line.get("state")) != "ON" and _line_ready(snapshot, line_number):
                action = self._action(
                    f"line_on:{line_number}",
                    "smart_panel_01",
                    "line_on",
                    value=line_number,
                    reason=f"линия {line_number} нужна для задания",
                    cooldown_s=120,
                    priority=72,
                )
                if action:
                    return RuleResult(
                        situation=f"Линия {line_number} выключена",
                        cause="для задания требуется питание",
                        confidence=0.8,
                        evidence=[{"node": "smart_panel_01", "param": f"lines[{line_number - 1}].state", "value": line.get("state")}],
                        plan=[f"включить линию {line_number}"],
                        actions=[action],
                    )

        # Battery / diesel: only start the generator when the grid cannot
        # cover the load.  The server itself handles load shedding.
        energy = snapshot.energy()
        soc = energy.get("soc_pct")
        deficit = energy.get("deficit_w") or 0.0
        grid = _state(energy.get("grid_state"))
        diesel_state = _state(energy.get("diesel_state"))
        fuel = as_float(energy.get("fuel_pct"))
        low_energy = soc is not None and soc < 20 and (deficit > 0 or grid not in {"AVAILABLE", "UP", "OK"})
        if low_energy and diesel_state not in {"ON", "RUNNING"}:
            panel = snapshot.node("dome_automation_01")
            autostart = snapshot.online("dome_automation_01") and as_bool(panel.get("diesel_autostart"), True)
            if not autostart:
                return RuleResult(
                    situation="Низкий заряд АКБ",
                    cause="автозапуск дизель-генератора отключён",
                    confidence=0.9,
                    evidence=[{"node": "battery_01", "param": "soc_pct", "value": soc}, {"node": "solar_inverter_01", "param": "power_deficit_w", "value": deficit}],
                    plan=["не запускать дизель автоматически", "сообщить Оператору"],
                    messages=[OutgoingMessage(f"Низкий заряд АКБ ({soc:.1f}%), но автозапуск дизель-генератора отключён. Нужно решение Оператора.", "battery_low_autostart_off", min_interval_s=1800)],
                )
            if not _diesel_ready(snapshot):
                return RuleResult(
                    situation="Низкий заряд АКБ",
                    cause="дизель-генератор неисправен или недоступен",
                    confidence=0.9,
                    evidence=[{"node": "battery_01", "param": "soc_pct", "value": soc}, {"node": "dizel_1", "param": "fault_code", "value": snapshot.node("dizel_1").get("fault_code")}],
                    plan=["не запускать неисправный дизель автоматически", "сообщить Оператору"],
                    messages=[OutgoingMessage(f"Низкий заряд АКБ ({soc:.1f}%), но дизель-генератор неисправен или недоступен. Автозапуск запрещён.", "battery_low_diesel_fault", min_interval_s=1800)],
                )
            if not _fuel_quality_ok(snapshot):
                return RuleResult(
                    situation="Низкий заряд АКБ",
                    cause="топливо не соответствует требованиям",
                    confidence=0.9,
                    evidence=[{"node": "battery_01", "param": "soc_pct", "value": soc}, {"node": "fuel_tank_01", "param": "fuel_quality", "value": snapshot.node("fuel_tank_01").get("fuel_quality")}],
                    plan=["не запускать дизель на непригодном топливе", "сообщить Оператору"],
                    messages=[OutgoingMessage(f"Низкий заряд АКБ ({soc:.1f}%), но качество топлива не подтверждено. Автозапуск запрещён.", "battery_low_fuel_bad", min_interval_s=1800)],
                )
            if fuel is not None and fuel > 10:
                action = self._action(
                    "diesel_start_low_soc",
                    "dizel_1",
                    "turn_on",
                    reason=f"SOC {soc:.1f}% и дефицит/нет сети",
                    cooldown_s=600,
                    priority=78,
                )
                if action:
                    return RuleResult(
                        situation="Низкий заряд АКБ",
                        cause="нет достаточной генерации/сети",
                        confidence=0.8,
                        evidence=[{"node": "battery_01", "param": "soc_pct", "value": soc}, {"node": "solar_inverter_01", "param": "power_deficit_w", "value": deficit}],
                        plan=["запустить дизель-генератор", "следить за SOC"],
                        actions=[action],
                        messages=[OutgoingMessage(f"Низкий заряд АКБ ({soc:.1f}%). Запускаю дизель-генератор.", "battery_low", min_interval_s=600)],
                    )
            return RuleResult(
                situation="Низкий заряд АКБ",
                cause="уровень топлива неизвестен или ниже резерва",
                confidence=0.9,
                evidence=[{"node": "battery_01", "param": "soc_pct", "value": soc}, {"node": "fuel_tank_01", "param": "level_pct", "value": fuel}],
                plan=["не запускать дизель без топлива", "сообщить Оператору"],
                messages=[OutgoingMessage(f"Низкий заряд АКБ ({soc:.1f}%), но топлива недостаточно. Автозапуск запрещён.", "battery_low_fuel", min_interval_s=1800)],
            )

        return None

    def _hold_cleared(self, snapshot: Snapshot, reason: str) -> tuple[bool, str]:
        if not reason:
            return False, "причина удержания не указана; автоматический resume запрещён"
        if reason == "IL_CO2":
            if snapshot.co2.false_alarm or snapshot.co2.real_event:
                return False, "CO2 ещё требует разбирательства"
            state = _interlock_state(snapshot, "IL_CO2")
            if not state:
                return False, "IL_CO2 неизвестен; автоматический resume запрещён"
            if _interlock_tripped(snapshot, "IL_CO2"):
                return False, f"IL_CO2 {state}"
            co2_ppm = snapshot.co2.chosen_ppm
            if co2_ppm is None:
                return False, "CO2 неизвестен; автоматический resume запрещён"
            if co2_ppm >= 1800:
                return False, "CO2 ещё высокий"
            return True, "CO2 в норме"
        if reason == "IL_CONDENSATE":
            if not _climate_data_known(snapshot):
                return False, "данные климата недоступны или неполны; автоматический resume запрещён"
            if _condensate_present(snapshot):
                return False, "конденсат/влажность ещё выше нормы"
            return True, "влажность и точка росы в норме"
        if reason == "IL_SMOKE":
            if not _smoke_data_known(snapshot):
                return False, "дымовые датчики недоступны или состояние дыма неизвестно; автоматический resume запрещён"
            if _powder_present(snapshot):
                return False, "дым ещё обнаружен"
            state = _interlock_state(snapshot, "IL_SMOKE")
            if not state:
                return False, "IL_SMOKE неизвестен; автоматический resume запрещён"
            if _interlock_tripped(snapshot, "IL_SMOKE"):
                return False, f"IL_SMOKE {state}; нужен сброс Оператором"
            if not _fire_ready(snapshot):
                return False, "пожарная автоматика не готова или недоступна"
            return True, "дым не обнаружен, IL_SMOKE сброшен"
        if reason == "IL_VOC_BOX":
            if not snapshot.online("air_quality_sensor_01"):
                return False, "датчик воздуха недоступен; автоматический resume запрещён"
            air = snapshot.node("air_quality_sensor_01")
            voc = as_float(air.get("voc_index"))
            if voc is None:
                return False, "VOC неизвестен; автоматический resume запрещён"
            if voc > 400:
                return False, f"VOC {voc:.0f}"
            if not _fume_ready(snapshot):
                return False, "вытяжка не даёт рабочий поток"
            return True, "VOC и вытяжка в норме"
        if reason == "UNDERVOLTAGE":
            if not snapshot.online("smart_panel_01"):
                return False, "щиток недоступен; напряжение не подтверждено"
            panel = snapshot.node("smart_panel_01")
            if panel.get("bus_powered") is False:
                return False, "шина без питания"
            voltage = as_float(panel.get("bus_voltage_v"))
            if voltage is None:
                return False, "напряжение неизвестно; автоматический resume запрещён"
            if voltage < 200:
                return False, f"напряжение {voltage:.1f} V"
            return True, "напряжение восстановилось"
        if reason in {"BED_COLLISION", "FILAMENT_RUNOUT"}:
            return False, f"{reason} требует Оператора"
        return False, f"неизвестная причина {reason}; автоматический resume запрещён"

    def _line_needed(self, snapshot: Snapshot, task: Mapping[str, Any] | None, line_number: int) -> bool:
        if not task:
            return False
        for item in _task_items(task):
            kind = str(item.get("kind") or "")
            status = _new_status(item)
            if status in {"done", "needs_info"}:
                continue
            if kind == "print" and line_number == 3:
                return True
            if kind == "cnc" and line_number == 4:
                return True
            if kind in {"line_on", "line_off"} and as_int(item.get("line"), -1) == line_number:
                return True
        return False

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def _task(self, snapshot: Snapshot, task: Mapping[str, Any] | None) -> RuleResult | None:
        if not isinstance(task, Mapping):
            return None
        items = _task_items(task)
        if not items:
            return None

        for index, item in enumerate(items):
            status = _new_status(item)
            if status in {"done", "needs_info"}:
                continue
            kind = str(item.get("kind") or "")
            if kind in {"monitor", "report"}:
                changed = _mark(items, index, "done")
                return RuleResult(
                    situation="Задание на наблюдение",
                    cause="Оператор просил контролировать смену",
                    confidence=0.9,
                    plan=["наблюдать и докладывать"],
                    messages=[OutgoingMessage(self._brief_text(snapshot), "shift_report", min_interval_s=300)],
                    task=_set_items(task, items) if changed else None,
                    task_changed=changed,
                )
            if kind in {"print", "cnc"}:
                result = self._task_machine(snapshot, task, items, index, kind)
                if result is not None:
                    return result
                continue
            if kind in {"line_on", "line_off"}:
                result = self._task_line(snapshot, task, items, index, kind, as_int(item.get("line")))
                if result is not None:
                    return result
                continue
            if kind == "inspect":
                changed = _mark(items, index, "done")
                return RuleResult(
                    situation="Запрошена проверка",
                    cause="команда Оператора",
                    confidence=0.7,
                    plan=["собрать телеметрию и доложить"],
                    messages=[OutgoingMessage(self._brief_text(snapshot), "inspect_report", min_interval_s=60)],
                    task=_set_items(task, items) if changed else None,
                    task_changed=changed,
                )
            # Unknown item: do not guess.
            changed = _mark(items, index, "needs_info", note="unsupported task item")
            return RuleResult(
                situation="Задание не распознано",
                cause="нет безопасного автоматического сценария",
                confidence=0.5,
                plan=["уточнить у Оператора"],
                messages=[OutgoingMessage(
                    f"Не понял задание: «{item.get('text', '')}». Уточните, пожалуйста.",
                    f"unknown:{index}",
                    min_interval_s=300,
                )],
                task=_set_items(task, items) if changed else None,
                task_changed=changed,
            )
        return None

    def _task_machine(
        self,
        snapshot: Snapshot,
        task: Mapping[str, Any],
        items: list[dict[str, Any]],
        index: int,
        kind: str,
    ) -> RuleResult | None:
        node_id = "printer_3d_01" if kind == "print" else "cnc_01"
        node = _machine(snapshot, node_id)
        status = _new_status(items[index])
        filename = _file_for_item(items[index], node)

        if not snapshot.online(node_id):
            action = self._action(
                f"reconnect:{node_id}",
                node_id,
                "reconnect",
                reason=f"{node_id} offline",
                cooldown_s=60,
                verify=False,
                priority=80,
            )
            if action:
                return self._action_result(task, items, index, action, "Узел не на связи")
            return None

        if status == "started":
            completion = check_print_completion(node) if kind == "print" else check_cnc_completion(node)
            if completion.ok:
                _mark(items, index, "done", completed_at=self.clock(), completion=completion.as_dict())
                return RuleResult(
                    situation=f"{node_id} завершил задание",
                    cause="состояние и прогресс подтверждены",
                    confidence=0.95,
                    evidence=[{"node": node_id, "param": "completion", "value": completion.observed}],
                    plan=["зафиксировать выполнение", "сообщить Оператору"],
                    messages=[OutgoingMessage(
                        f"{node_id}: задание «{filename or node.get('file_name') or '?'}» выполнено.",
                        f"{node_id}:done:{task.get('message_id')}",
                        min_interval_s=0,
                    )],
                    task=_set_items(task, items),
                    task_changed=True,
                )
            if _state(node.get("state")) in {"ERROR", "ALARM"} or node.get("error_code"):
                _mark(items, index, "needs_info", note=f"machine error {node.get('error_code') or node.get('alarm_code')}")
                return RuleResult(
                    situation=f"{node_id} в ошибке",
                    cause=str(node.get("error_code") or node.get("alarm_code") or "unknown"),
                    confidence=0.95,
                    plan=["остановить задачу", "сообщить Оператору"],
                    messages=[OutgoingMessage(
                        f"{node_id}: ошибка {node.get('error_code') or node.get('alarm_code')}. Задание снято с выполнения.",
                        f"{node_id}:error:{task.get('message_id')}",
                        min_interval_s=300,
                    )],
                    task=_set_items(task, items),
                    task_changed=True,
                )
            return None

        if not filename:
            _mark(items, index, "needs_info", note="file name required")
            files = _file_names(node)
            return RuleResult(
                situation=f"Не указан файл для {node_id}",
                cause="задание не содержит однозначного имени файла",
                confidence=0.8,
                plan=["уточнить имя файла"],
                messages=[OutgoingMessage(
                    f"Уточните файл для {node_id}. Доступны: {', '.join(files) if files else 'список пуст'}.",
                    f"{node_id}:need_file:{task.get('message_id')}",
                    min_interval_s=300,
                )],
                task=_set_items(task, items),
                task_changed=True,
            )

        if not _file_present(node, filename):
            _mark(items, index, "needs_info", note=f"file {filename} not in files")
            return RuleResult(
                situation=f"Файл {filename} отсутствует",
                cause="start_job принимает только файлы из списка files",
                confidence=0.9,
                evidence=[{"node": node_id, "param": "files", "value": _file_names(node)}],
                plan=["не запускать задание", "уточнить файл у Оператора"],
                messages=[OutgoingMessage(
                    f"Файла {filename} нет на {node_id}. Доступны: {', '.join(_file_names(node)) or 'нет'}.",
                    f"{node_id}:missing_file:{task.get('message_id')}",
                    min_interval_s=300,
                )],
                task=_set_items(task, items),
                task_changed=True,
            )

        # Already running the requested job: do not send start_job again.
        if _state(node.get("state")) in {"PRINTING", "RUNNING"}:
            current = str(node.get("file_name") or "")
            if not current or current.casefold() == filename.casefold():
                _mark(items, index, "started", started_at=self.clock())
                return RuleResult(
                    situation=f"{node_id} уже выполняет задание",
                    cause="state уже RUNNING/PRINTING",
                    confidence=0.9,
                    plan=["наблюдать за выполнением"],
                    task=_set_items(task, items),
                    task_changed=True,
                )
            return None

        if _state(node.get("hold_level")) == "ALARM":
            _mark(items, index, "needs_info", note="machine hold ALARM")
            return RuleResult(
                situation=f"{node_id} заблокирован",
                cause=f"hold_reason={node.get('hold_reason')}",
                confidence=0.95,
                plan=["не запускать задание"],
                messages=[OutgoingMessage(
                    f"{node_id} в ALARM ({node.get('hold_reason')}). Задание ждёт.",
                    f"{node_id}:hold_alarm:{task.get('message_id')}",
                    min_interval_s=600,
                )],
                task=_set_items(task, items),
                task_changed=True,
            )

        if node.get("hold_level") or as_bool(node.get("paused")):
            _mark(items, index, "pending", note="waiting for resume")
            return None

        safety = self._task_safety_block(snapshot, node_id, kind)
        if safety is not None:
            _mark(items, index, "pending", note=safety[0])
            return RuleResult(
                situation=f"Задание {node_id} ждёт",
                cause=safety[0],
                confidence=0.9,
                plan=["устранить блокирующую причину", "затем запустить"],
                messages=[OutgoingMessage(safety[1], f"blocked:{node_id}:{safety[0]}", min_interval_s=300)],
                task=_set_items(task, items),
                task_changed=True,
            )

        line_number = 3 if kind == "print" else 4
        if not _line_ready(snapshot, line_number):
            if _line_tripped(snapshot, line_number):
                action = self._action(
                    f"task_line_reset:{line_number}",
                    "smart_panel_01",
                    "reset_protection",
                    value=line_number,
                    reason=f"для {node_id} нужна линия {line_number}",
                    cooldown_s=300,
                    priority=70,
                )
                if action:
                    return self._action_result(task, items, index, action, f"Линия {line_number} отключена защитой")
            return None

        if kind == "print":
            if as_bool(node.get("bed_clear")) is False:
                _mark(items, index, "pending", note="bed occupied")
                return RuleResult(
                    situation="Стол принтера занят",
                    cause="bed_clear=false; нужен remove_part Оператором",
                    confidence=0.95,
                    plan=["не запускать печать", "попросить Оператора снять деталь"],
                    messages=[OutgoingMessage(
                        "Стол 3D-принтера занят. Снимите деталь и откройте бокс, затем я запущу печать.",
                        f"printer:bed:{task.get('message_id')}",
                        min_interval_s=300,
                    )],
                    task=_set_items(task, items),
                    task_changed=True,
                )
            if not as_bool(node.get("homed")):
                action = self._action(
                    f"task_home:{node_id}",
                    node_id,
                    "home",
                    reason="перед start_job нужен home",
                    cooldown_s=60,
                    priority=68,
                )
                if action:
                    _mark(items, index, "homing", note="home sent")
                    return self._action_result(task, items, index, action, "Принтер не homed")
            if _state(node.get("state")) not in {"IDLE", "COMPLETED"}:
                return None
            action = self._action(
                f"task_start:{node_id}:{filename}",
                node_id,
                "start_job",
                value=filename,
                reason=f"запуск задания Оператора: {filename}",
                cooldown_s=120,
                priority=65,
            )
            if action:
                _mark(items, index, "starting", note="start_job sent")
                return self._action_result(task, items, index, action, "Старт печати")
            return None

        # CNC
        fume = snapshot.node("fume_extraction_01")
        if not as_bool(fume.get("enabled")):
            action = self._action(
                "cnc_fume_on",
                "fume_extraction_01",
                "turn_on",
                reason="фрезеровка требует работающей вытяжки",
                cooldown_s=120,
                priority=67,
            )
            if action:
                return self._action_result(task, items, index, action, "Вытяжка выключена")
            return None
        if _state(node.get("state")) not in {"IDLE", "COMPLETED"}:
            return None
        if not as_bool(node.get("homed")):
            action = self._action(
                f"task_home:{node_id}",
                node_id,
                "home",
                reason="перед start_job нужен home",
                cooldown_s=60,
                priority=68,
            )
            if action:
                _mark(items, index, "homing", note="home sent")
                return self._action_result(task, items, index, action, "Фрезер не homed")
            return None
        action = self._action(
            f"task_start:{node_id}:{filename}",
            node_id,
            "start_job",
            value=filename,
            reason=f"запуск задания Оператора: {filename}",
            cooldown_s=120,
            priority=65,
        )
        if action:
            _mark(items, index, "starting", note="start_job sent")
            return self._action_result(task, items, index, action, "Старт фрезеровки")
        return None

    def _task_line(
        self,
        snapshot: Snapshot,
        task: Mapping[str, Any],
        items: list[dict[str, Any]],
        index: int,
        kind: str,
        line_number: int | None,
    ) -> RuleResult | None:
        if line_number is None:
            _mark(items, index, "needs_info", note="line number required")
            return RuleResult(
                situation="Не указана линия",
                cause="задание должно содержать номер линии 1..8",
                confidence=0.8,
                plan=["уточнить номер линии"],
                messages=[OutgoingMessage("Уточните номер линии (1–8).", f"line:need:{index}", min_interval_s=300)],
                task=_set_items(task, items),
                task_changed=True,
            )
        line = _line(snapshot, line_number)
        current = _state(line.get("state"))
        if kind == "line_on":
            if current == "ON":
                _mark(items, index, "done", completed_at=self.clock())
                return RuleResult(
                    situation=f"Линия {line_number} уже включена",
                    cause="состояние совпало с заданием",
                    confidence=0.95,
                    plan=["зафиксировать выполнение"],
                    task=_set_items(task, items),
                    task_changed=True,
                )
            if _line_alarmed(snapshot, line_number):
                _mark(items, index, "needs_info", note="line alarm")
                return RuleResult(
                    situation=f"Линия {line_number} в аварии",
                    cause=str(line.get("alarm_code")),
                    confidence=0.95,
                    plan=["не включать", "вызвать Оператора"],
                    messages=[OutgoingMessage(
                        f"Линия {line_number} в аварии ({line.get('alarm_code')}). Включение запрещено до осмотра.",
                        f"line:{line_number}:alarm",
                    )],
                    task=_set_items(task, items),
                    task_changed=True,
                )
            if _line_tripped(snapshot, line_number):
                action = self._action(
                    f"task_line_reset:{line_number}",
                    "smart_panel_01",
                    "reset_protection",
                    value=line_number,
                    reason=f"линия {line_number} сработала по защите",
                    cooldown_s=300,
                    priority=70,
                )
                if action:
                    return self._action_result(task, items, index, action, f"Линия {line_number} tripped")
                return None
            if _line_blocked(snapshot, line_number):
                return None
            action = self._action(
                f"task_line_on:{line_number}",
                "smart_panel_01",
                "line_on",
                value=line_number,
                reason=f"включить линию {line_number} по заданию",
                cooldown_s=120,
                priority=60,
            )
            if action:
                return self._action_result(task, items, index, action, f"Включение линии {line_number}")
            return None

        if current == "OFF":
            _mark(items, index, "done", completed_at=self.clock())
            return RuleResult(
                situation=f"Линия {line_number} уже выключена",
                cause="состояние совпало с заданием",
                confidence=0.95,
                plan=["зафиксировать выполнение"],
                task=_set_items(task, items),
                task_changed=True,
            )
        action = self._action(
            f"task_line_off:{line_number}",
            "smart_panel_01",
            "line_off",
            value=line_number,
            reason=f"выключить линию {line_number} по заданию",
            cooldown_s=120,
            priority=60,
        )
        if action:
            return self._action_result(task, items, index, action, f"Отключение линии {line_number}")
        return None

    def _task_safety_block(self, snapshot: Snapshot, node_id: str, kind: str) -> tuple[str, str] | None:
        if snapshot.co2.real_event or (snapshot.co2.chosen_ppm or 0) >= 2000:
            return "co2_high", f"CO2 {(snapshot.co2.chosen_ppm or 0):.0f} ppm: запуск производства запрещён."
        if not _smoke_data_known(snapshot):
            return "smoke_unknown", "Дымовые датчики недоступны или их показания неизвестны: запуск производства запрещён."
        if _powder_present(snapshot):
            return "smoke", "Дым в куполе: запуск производства запрещён."
        if not _fire_ready(snapshot):
            return "fire_not_ready", "Пожарная автоматика не готова или недоступна: запуск производства запрещён."
        if not _climate_data_known(snapshot):
            return "climate_unknown", "Данные климата недоступны или неполны: запуск производства запрещён."
        if not _power_ready(snapshot):
            return "power_not_ready", "Напряжение шины неизвестно или ниже нормы: запуск производства запрещён."
        air_block = _air_quality_block(snapshot)
        if air_block is not None:
            return air_block
        if _condensate_present(snapshot):
            return "condensate", "Риск конденсата: запуск станков запрещён."
        if node_id == "cnc_01":
            fume = snapshot.node("fume_extraction_01")
            if not as_bool(fume.get("enabled")):
                return "fume_off", "Вытяжка выключена: фрезеровка запрещена."
            if not _fume_ready(snapshot):
                return "fume_flow", "Вытяжка не даёт рабочий поток: фрезеровка запрещена."
        for alert in snapshot.alerts():
            if alert.get("severity") != "critical":
                continue
            code = str(alert.get("code") or "")
            if code in _PRODUCTION_BLOCKING_ALERTS:
                return code, str(alert.get("message"))
        return None

    def _action_result(
        self,
        task: Mapping[str, Any],
        items: list[dict[str, Any]],
        index: int,
        action: ActionSpec,
        situation: str,
    ) -> RuleResult:
        return RuleResult(
            situation=situation,
            cause=action.reason,
            confidence=0.85,
            plan=[action.reason],
            actions=[action],
            task=_set_items(task, items),
            task_changed=True,
        )

    def _brief_text(self, snapshot: Snapshot) -> str:
        energy = snapshot.energy()
        production = snapshot.production()
        co2 = snapshot.co2.chosen_ppm
        parts = [
            f"Время купола: {snapshot.clock}",
            f"CO2 {co2:.0f} ppm" if co2 is not None else "CO2 нет данных",
            f"АКБ {energy.get('soc_pct'):.0f}%" if energy.get("soc_pct") is not None else "АКБ нет данных",
            f"принтер {production['printer_3d_01'].get('state')}",
            f"ЧПУ {production['cnc_01'].get('state')}",
        ]
        alerts = snapshot.alerts()
        if alerts:
            parts.append("Аварии: " + "; ".join(str(alert.get("message")) for alert in alerts[:4]))
        else:
            parts.append("Аварий нет")
        return ". ".join(parts) + "."
