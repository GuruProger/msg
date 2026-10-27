"""Telemetry normalization and small, deterministic analyses.

This module does not decide what to do.  It converts the raw server frame
into stable values and assessments that the rule engine can reason about.
In particular, :func:`analyze_co2` implements the documented rule: never
trust one CO2 sensor when two independent sensors disagree with it.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

CO2_SENSORS = ("climate_sensor_01", "climate_sensor_02", "climate_sensor_03")
CO2_TRIP_PPM = 2000.0
CO2_STUFFY_PPM = 1000.0
CO2_CONSENSUS_SPREAD_PPM = 200.0
CO2_OUTLIER_GAP_PPM = 400.0
CO2_OUTLIER_RATIO = 1.8

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool) or value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def as_int(value: Any, default: int | None = None) -> int | None:
    number = as_float(value)
    return default if number is None else int(number)


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "ok", "armed"}
    return default


def _state(value: Any) -> str:
    return str(value or "").strip().upper()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def normalize_interlocks(value: Any) -> dict[str, dict[str, Any]]:
    """Normalize the API interlock map and the observed list form.

    The live API may return either ``{"IL_CO2": {...}}`` or
    ``[{"id": "IL_CO2", ...}]``.  Unknown entries are ignored here; callers
    that make safety decisions must treat a malformed raw value as unknown,
    not as ARMED.
    """
    if isinstance(value, Mapping):
        return {
            str(name): dict(entry)
            for name, entry in value.items()
            if isinstance(entry, Mapping)
        }
    if isinstance(value, (list, tuple)):
        result: dict[str, dict[str, Any]] = {}
        for entry in value:
            if not isinstance(entry, Mapping):
                continue
            name = (
                entry.get("id")
                or entry.get("name")
                or entry.get("interlock")
                or entry.get("key")
            )
            if name is None:
                continue
            result[str(name)] = dict(entry)
        return result
    return {}


def _node_online(node: Mapping[str, Any]) -> bool:
    if not node:
        return False
    if node.get("online") is False:
        return False
    return _state(node.get("status")) not in {"UNKNOWN", "OFFLINE", "FAILED"}


def _sensor_valid(node: Mapping[str, Any], *, key: str = "co2_ppm") -> tuple[bool, str]:
    if not _node_online(node):
        return False, "offline"
    if as_float(node.get(key)) is None:
        return False, "no_value"
    state = _state(node.get("sensor_state") or node.get("state") or "OK")
    if state in {"UNRELIABLE", "FAILED", "FAULT", "ERROR", "OFFLINE"}:
        return False, f"sensor_state={state}"
    confidence = as_float(node.get("confidence"))
    if confidence is not None and confidence < 0.2:
        return False, "low_confidence"
    return True, state


@dataclass(frozen=True, slots=True)
class Co2Reading:
    node_id: str
    ppm: float | None
    valid: bool
    online: bool
    state: str
    confidence: float | None
    reason: str
    zone: str
    reading_delay_s: float | None

    @property
    def current(self) -> bool:
        """Whether this value can participate in a current quorum.

        Delay semantics are not confirmed yet.  Any positive ``reading_delay_s``
        means the packet may contain an older physical measurement, so the
        reading is retained for history but excluded from current decisions.
        """
        return self.valid and (self.reading_delay_s is None or self.reading_delay_s <= 0.0)

    @property
    def quality(self) -> str:
        if not self.valid:
            return "INVALID"
        return "CURRENT" if self.current else "DELAYED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ppm": self.ppm,
            "valid": self.valid,
            "online": self.online,
            "state": self.state,
            "confidence": self.confidence,
            "reason": self.reason,
            "zone": self.zone,
            "reading_delay_s": self.reading_delay_s,
            "current": self.current,
            "quality": self.quality,
        }


@dataclass(frozen=True, slots=True)
class Co2Assessment:
    readings: tuple[Co2Reading, ...]
    chosen_id: str | None
    chosen_ppm: float | None
    median_ppm: float | None
    consensus: bool
    false_alarm: bool
    real_event: bool
    suspect_ids: tuple[str, ...]
    unreliable_ids: tuple[str, ...]
    override_used: bool
    override_ignored: bool
    delayed_ids: tuple[str, ...]
    reason: str

    @property
    def trip_candidate(self) -> bool:
        return self.chosen_ppm is not None and self.chosen_ppm >= CO2_TRIP_PPM

    @property
    def stuffy(self) -> bool:
        return self.chosen_ppm is not None and self.chosen_ppm >= CO2_STUFFY_PPM

    @property
    def valid_count(self) -> int:
        return sum(1 for reading in self.readings if reading.valid)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chosen_id": self.chosen_id,
            "chosen_ppm": self.chosen_ppm,
            "median_ppm": self.median_ppm,
            "consensus": self.consensus,
            "false_alarm": self.false_alarm,
            "real_event": self.real_event,
            "trip_candidate": self.trip_candidate,
            "stuffy": self.stuffy,
            "valid_count": self.valid_count,
            "suspect_ids": list(self.suspect_ids),
            "unreliable_ids": list(self.unreliable_ids),
            "override_used": self.override_used,
            "override_ignored": self.override_ignored,
            "delayed_ids": list(self.delayed_ids),
            "reason": self.reason,
            "readings": [reading.as_dict() for reading in self.readings],
        }


def _isolated_indices(values: list[tuple[int, float]]) -> set[int]:
    """Return indices that are isolated from at least two mutually close peers."""
    isolated: set[int] = set()
    for index, value in values:
        peers = [
            other_value
            for other_index, other_value in values
            if other_index != index and abs(other_value - value) <= CO2_CONSENSUS_SPREAD_PPM
        ]
        if len(peers) >= 2:
            continue
        # A single isolated high/low reading is suspect when the rest agree.
        rest = [other_value for other_index, other_value in values if other_index != index]
        if len(rest) >= 2 and max(rest) - min(rest) <= CO2_CONSENSUS_SPREAD_PPM:
            if abs(value - statistics.median(rest)) >= CO2_OUTLIER_GAP_PPM:
                isolated.add(index)
        elif len(rest) == 1 and abs(value - rest[0]) >= 1000.0:
            isolated.add(index)
    # Ratio-based rule catches a high reading next to a tight low cluster.
    if len(values) >= 3:
        median = statistics.median(value for _, value in values)
        for index, value in values:
            if median > 0 and value >= median * CO2_OUTLIER_RATIO and value - median >= CO2_OUTLIER_GAP_PPM:
                others = [other for other_index, other in values if other_index != index]
                if others and max(others) - min(others) <= CO2_CONSENSUS_SPREAD_PPM:
                    isolated.add(index)
    return isolated


def analyze_co2(
    nodes: Mapping[str, Any],
    override: Mapping[str, Any] | None = None,
) -> Co2Assessment:
    """Assess CO2 using at least two independent sensors when available.

    A single sensor reporting a physically impossible jump (for example
    3043 ppm while the other two report ~780 ppm) is marked suspect.  The
    assessment never hides the raw readings: it exposes the consensus value,
    suspect ids, and delayed channels.  A reading with positive
    ``reading_delay_s`` may describe an older measurement, so it is retained
    for history but excluded from the current false-alarm quorum.
    """
    readings: list[Co2Reading] = []
    ids = list(CO2_SENSORS)
    for node_id in sorted(nodes):
        if node_id.startswith("climate_sensor_") and node_id not in ids:
            ids.append(node_id)
    for node_id in ids:
        node = _dict(nodes.get(node_id))
        valid, reason = _sensor_valid(node)
        ppm = as_float(node.get("co2_ppm"))
        readings.append(
            Co2Reading(
                node_id=node_id,
                ppm=ppm,
                valid=valid,
                online=_node_online(node),
                state=_state(node.get("sensor_state") or node.get("state") or "UNKNOWN"),
                confidence=as_float(node.get("confidence")),
                reason=reason,
                zone=str(node.get("zone") or "").strip().upper(),
                reading_delay_s=as_float(node.get("reading_delay_s")),
            )
        )

    valid_values = [
        (index, reading.ppm)
        for index, reading in enumerate(readings)
        if reading.valid and reading.ppm is not None
    ]
    values = [float(value) for _, value in valid_values]
    current_values = [
        (index, float(value))
        for index, value in valid_values
        if readings[index].current
    ]
    # Only readings comparable in time can create a current "two against one"
    # quorum.  Delayed reserve channels remain visible in the assessment but
    # cannot refute a fresh alarm or make a fresh high reading look isolated.
    suspect_indices = _isolated_indices(current_values)
    suspect_ids = tuple(readings[index].node_id for index in sorted(suspect_indices))

    false_alarm = False
    for index in suspect_indices:
        suspect_ppm = readings[index].ppm or 0.0
        if suspect_ppm < CO2_TRIP_PPM:
            continue
        zone = readings[index].zone
        if not zone:
            continue
        same_zone_safe = [
            value
            for other_index, value in current_values
            if other_index != index
            and readings[other_index].zone == zone
            and value < CO2_TRIP_PPM
        ]
        if len(same_zone_safe) >= 2:
            false_alarm = True
            break

    if false_alarm:
        reliable_values = [
            (index, float(value))
            for index, value in valid_values
            if index not in suspect_indices
        ]
    else:
        # A suspect reading is only excluded after the false-alarm quorum is
        # established.  Otherwise keep it in the safety path and choose the
        # highest current value below, so an unconfirmed spike cannot clear IL.
        reliable_values = [(index, float(value)) for index, value in valid_values]
    reliable = [value for _, value in reliable_values]
    median = statistics.median(reliable) if reliable else (statistics.median(values) if values else None)

    automation = _dict(nodes.get("dome_automation_01"))
    primary_id = str(automation.get("primary_co2_sensor") or "").strip() or None
    override_id = str((override or {}).get("node_id") or "").strip() or None
    override_used = False
    override_ignored = False
    chosen_index: int | None = None
    chosen_ppm: float | None = None

    valid_by_id = {
        readings[index].node_id: (index, float(value))
        for index, value in valid_values
    }
    current_high = [
        (index, value)
        for index, value in current_values
        if value >= CO2_TRIP_PPM
    ]

    if current_high and not false_alarm:
        # Safety-first choice: a current high reading wins over delayed,
        # other-zone, or historical overrides until it is confirmed false.
        chosen_index, chosen_ppm = max(current_high, key=lambda item: item[1])
    else:
        if override_id:
            candidate = valid_by_id.get(override_id)
            if (
                candidate is not None
                and candidate[0] not in suspect_indices
                and readings[candidate[0]].current
            ):
                chosen_index, chosen_ppm = candidate
                override_used = True
            else:
                override_ignored = True

        if chosen_index is None and reliable_values:
            current_reliable = [
                (index, value)
                for index, value in reliable_values
                if readings[index].current
            ]
            pool = current_reliable or reliable_values
            chosen_index, chosen_ppm = min(
                pool,
                key=lambda item: abs(item[1] - float(median)),
            )

        if chosen_index is None and values:
            # No trustworthy cluster: keep the highest reading so safety rules
            # fail closed instead of silently trusting the lowest sensor.
            chosen_index, chosen_ppm = max(
                ((index, float(value)) for index, value in valid_values),
                key=lambda item: item[1],
            )

    chosen_id = readings[chosen_index].node_id if chosen_index is not None else None
    spread = (max(reliable) - min(reliable)) if len(reliable) >= 2 else None
    consensus = bool(len(reliable) >= 2 and spread is not None and spread <= CO2_CONSENSUS_SPREAD_PPM)
    high_reliable = sum(1 for value in reliable if value >= CO2_TRIP_PPM)
    real_event = bool(high_reliable >= 2 and (consensus or high_reliable >= max(2, len(reliable))))
    delayed_ids = tuple(
        reading.node_id for reading in readings if reading.valid and not reading.current
    )

    if not values:
        reason = "no reliable CO2 sensor"
    elif false_alarm:
        reason = "high current reading is isolated; same-zone current sensors agree on a safe level"
    elif real_event:
        reason = "CO2 is high on at least two independent sensors"
    elif consensus:
        reason = "CO2 sensors agree"
    elif current_high:
        reason = "current CO2 is high; delayed or other-zone readings cannot refute it"
    elif len(values) == 1:
        reason = "only one CO2 sensor is usable; treat as unconfirmed"
    elif suspect_ids:
        reason = "one or more CO2 sensors disagree; consensus excluded them"
    else:
        reason = "CO2 sensors disagree; value is unconfirmed"

    return Co2Assessment(
        readings=tuple(readings),
        chosen_id=chosen_id,
        chosen_ppm=chosen_ppm,
        median_ppm=median,
        consensus=consensus,
        false_alarm=false_alarm,
        real_event=real_event,
        suspect_ids=suspect_ids,
        unreliable_ids=tuple(reading.node_id for reading in readings if not reading.valid),
        override_used=override_used,
        override_ignored=override_ignored,
        delayed_ids=delayed_ids,
        reason=reason,
    )


def _game_clock(game_time: Mapping[str, Any]) -> str:
    day = game_time.get("day", "?")
    hour_value = as_float(game_time.get("hour"))
    if hour_value is not None:
        hours = int(hour_value) % 24
        minutes = int(round((hour_value - int(hour_value)) * 60))
        if minutes >= 60:
            hours = (hours + 1) % 24
            minutes = 0
        return f"day {day}, {hours:02d}:{minutes:02d}"
    if game_time.get("time"):
        return f"day {day}, {game_time['time']}"
    return f"day {day}"


class Snapshot:
    """Read-only normalized view over one telemetry frame."""

    def __init__(
        self,
        data: Mapping[str, Any] | None,
        *,
        co2_override: Mapping[str, Any] | None = None,
        received_at: float | None = None,
    ) -> None:
        self.data: dict[str, Any] = dict(data or {})
        nodes = self.data.get("nodes")
        self.nodes: dict[str, Any] = nodes if isinstance(nodes, dict) else {}
        game_time = self.data.get("game_time")
        self.game_time: dict[str, Any] = game_time if isinstance(game_time, dict) else {}
        self.received_at = time.time() if received_at is None else received_at
        self.clock = _game_clock(self.game_time)
        self.co2 = analyze_co2(self.nodes, override=co2_override)
        self._alerts_cache: list[dict[str, Any]] | None = None

    def node(self, node_id: str) -> dict[str, Any]:
        return _dict(self.nodes.get(node_id))

    def online(self, node_id: str) -> bool:
        return _node_online(self.node(node_id))

    def reading(self, node_id: str, key: str, default: float | None = None) -> float | None:
        return as_float(self.node(node_id).get(key), default)

    def line(self, number: int) -> dict[str, Any]:
        for line in self.node("smart_panel_01").get("lines") or []:
            if isinstance(line, dict) and as_int(line.get("line")) == number:
                return line
        return {}

    def line_power(self, number: int) -> float:
        return as_float(self.line(number).get("power_w"), 0.0) or 0.0

    def line_on(self, number: int) -> bool:
        return _state(self.line(number).get("state")) == "ON"

    def machine(self, node_id: str) -> dict[str, Any]:
        return self.node(node_id)

    def machine_busy(self, node_id: str) -> bool:
        return _state(self.node(node_id).get("state")) in {
            "PRINTING",
            "RUNNING",
            "PAUSED",
            "HOLDING",
            "HOMING",
            "PROBING",
        }

    def machine_hold(self, node_id: str) -> str | None:
        value = self.node(node_id).get("hold_level")
        return str(value) if value else None

    def energy(self) -> dict[str, Any]:
        inverter = self.node("solar_inverter_01")
        battery = self.node("battery_01")
        diesel = self.node("dizel_1")
        fuel = self.node("fuel_tank_01")
        return {
            "soc_pct": as_float(battery.get("soc_pct")),
            "battery_state": battery.get("state"),
            "pv_power_w": as_float(self.node("solar_panels_01").get("power_w")),
            "wind_power_w": as_float(self.node("wind_turbine_01").get("power_w")),
            "load_power_w": as_float(inverter.get("load_power_w")),
            "grid_power_w": as_float(inverter.get("grid_power_w")),
            "grid_state": inverter.get("grid_state"),
            "deficit_w": as_float(inverter.get("power_deficit_w")),
            "inverter_temp_c": as_float(inverter.get("inverter_temp_c")),
            "inverter_error": inverter.get("error_code"),
            "diesel_state": diesel.get("state"),
            "diesel_power_w": as_float(diesel.get("power_w")),
            "fuel_pct": as_float(fuel.get("level_pct")),
            "autonomy_h": as_float(self.node("dome_automation_01").get("autonomy_forecast_h")),
        }

    def panel_summary(self) -> dict[str, Any]:
        panel = self.node("smart_panel_01")
        lines = []
        for line in panel.get("lines") or []:
            if not isinstance(line, dict):
                continue
            lines.append(
                {
                    "line": as_int(line.get("line")),
                    "name": line.get("name"),
                    "on": _state(line.get("state")) == "ON",
                    "power_w": as_float(line.get("power_w")),
                    "current_a": as_float(line.get("current_a")),
                    "rated_a": as_float(line.get("rated_current_a")),
                    "alarm": as_bool(line.get("alarm")),
                    "alarm_code": line.get("alarm_code"),
                    "protection": line.get("protection_state"),
                    "lockout_s": as_float(line.get("lockout_remaining_s")),
                }
            )
        return {
            "total_power_w": as_float(panel.get("total_power_w")),
            "bus_voltage_v": as_float(panel.get("bus_voltage_v")),
            "lines_on": as_int(panel.get("lines_on_count")),
            "lines": lines,
        }

    def production(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for node_id in ("printer_3d_01", "cnc_01"):
            node = self.node(node_id)
            result[node_id] = {
                "online": _node_online(node),
                "state": node.get("state"),
                "paused": as_bool(node.get("paused")),
                "file": node.get("file_name"),
                "progress_pct": as_float(node.get("progress_pct")),
                "z_mm": as_float(node.get("z_mm")),
                "hold_level": node.get("hold_level"),
                "hold_reason": node.get("hold_reason"),
                "resume_attempts": as_int(node.get("resume_attempts"), 0),
                "lockout_s": as_float(node.get("lockout_remaining_s")),
                "error_code": node.get("error_code"),
                "alarm": as_bool(node.get("alarm")),
                "alarm_code": node.get("alarm_code"),
                "bed_clear": node.get("bed_clear"),
                "homed": node.get("homed"),
                "webhook_state": node.get("webhook_state"),
                "real_state": node.get("real_state"),
            }
        return result

    def alerts(self) -> list[dict[str, Any]]:
        """Collect actionable alarms, most severe first.

        The result is cached for this immutable telemetry frame.  Callers get
        a shallow copy so a stray list mutation cannot poison later decisions.
        """
        if self._alerts_cache is not None:
            return list(self._alerts_cache)
        alerts: list[dict[str, Any]] = []

        def add(
            severity: str,
            code: str,
            node_id: str,
            message: str,
            **details: Any,
        ) -> None:
            alerts.append(
                {
                    "severity": severity,
                    "code": code,
                    "node_id": node_id,
                    "message": message,
                    "details": details,
                }
            )

        for node_id in ("smoke_detector_01", "smoke_detector_02"):
            node = self.node(node_id)
            if not self.online(node_id):
                add("critical", "SMOKE_OFFLINE", node_id, "Smoke detector telemetry unavailable")
            elif node.get("smoke_detected") is None:
                add("critical", "SMOKE_UNKNOWN", node_id, "Smoke detector reading unavailable")
            elif as_bool(node.get("smoke_detected")) or _state(node.get("alarm_state")) not in {"", "NORMAL"}:
                add("critical", "SMOKE", node_id, "Smoke detected", zone=node.get("zone"))

        fire = self.node("fire_suppression_01")
        if not self.online("fire_suppression_01"):
            add("critical", "FIRE_UNKNOWN", "fire_suppression_01", "Fire suppression telemetry unavailable")
        elif as_bool(fire.get("alarm")) or _state(fire.get("state")) == "DISCHARGED":
            add("critical", "FIRE", "fire_suppression_01", "Fire suppression alarm")
        elif fire.get("fault_code") or not as_bool(fire.get("ready")):
            add("critical", "FIRE_NOT_READY", "fire_suppression_01", "Fire suppression is not ready", fault=fire.get("fault_code"))

        if self.co2.real_event:
            add(
                "critical",
                "CO2_TRIP",
                self.co2.chosen_id or "climate_sensor",
                f"CO2 {self.co2.chosen_ppm:.0f} ppm confirmed by multiple sensors",
                chosen=self.co2.chosen_ppm,
            )
        elif self.co2.false_alarm:
            add(
                "warning",
                "CO2_SENSOR_SUSPECT",
                ",".join(self.co2.suspect_ids) or "climate_sensor",
                "Isolated CO2 reading rejected; other sensors agree",
                chosen=self.co2.chosen_ppm,
                suspect=list(self.co2.suspect_ids),
            )
        elif self.co2.trip_candidate and not self.co2.consensus:
            add(
                "warning",
                "CO2_UNCONFIRMED",
                self.co2.chosen_id or "climate_sensor",
                f"CO2 {self.co2.chosen_ppm:.0f} ppm is not confirmed by a second sensor",
                chosen=self.co2.chosen_ppm,
            )
        elif self.co2.stuffy:
            add("warning", "CO2_STUFFY", self.co2.chosen_id or "climate_sensor", f"CO2 {self.co2.chosen_ppm:.0f} ppm: ventilate", chosen=self.co2.chosen_ppm)

        climate = self.node("climate_sensor_01")
        humidity = as_float(climate.get("humidity_pct"))
        temp = as_float(climate.get("temperature_c"))
        dew = as_float(climate.get("dew_point_c"))
        if not self.online("climate_sensor_01"):
            add("warning", "CLIMATE_UNKNOWN", "climate_sensor_01", "Climate telemetry unavailable")
        elif humidity is None or temp is None or dew is None:
            add("warning", "CLIMATE_UNKNOWN", "climate_sensor_01", "Climate telemetry incomplete")
        elif humidity > 90:
            add("critical", "CONDENSATE", "climate_sensor_01", f"Humidity {humidity:.0f}% > 90%")
        elif (temp - dew) < 1.0:
            add("critical", "CONDENSATE", "climate_sensor_01", f"Dew point margin {temp - dew:.2f} C < 1 C")

        air = self.node("air_quality_sensor_01")
        co = as_float(air.get("co_ppm"))
        voc = as_float(air.get("voc_index"))
        if not self.online("air_quality_sensor_01"):
            add("warning", "AIR_UNKNOWN", "air_quality_sensor_01", "Air quality telemetry unavailable")
        elif as_bool(air.get("warming_up")) or co is None or voc is None:
            add("warning", "AIR_UNKNOWN", "air_quality_sensor_01", "Air quality telemetry incomplete")
        else:
            if co > 10:
                add("critical" if co > 25 else "warning", "CO", "air_quality_sensor_01", f"CO {co:.2f} ppm")
            if voc > 400:
                add("warning", "VOC", "air_quality_sensor_01", f"VOC index {voc:.0f}")

        radiation = self.node("radiation_sensor_01")
        dose = as_float(radiation.get("dose_rate_usv_h"))
        alarm_level = as_float(radiation.get("alarm_threshold"), 1.0)
        warning_level = as_float(radiation.get("warning_threshold"), 0.3)
        if _state(radiation.get("alarm_state")) == "ALARM" or (dose is not None and alarm_level is not None and dose >= alarm_level):
            add("critical", "RADIATION", "radiation_sensor_01", f"Dose {dose} uSv/h")
        elif dose is not None and warning_level is not None and dose >= warning_level:
            add("warning", "RADIATION_WARNING", "radiation_sensor_01", f"Dose {dose} uSv/h")

        chem = self.node("chem_sensor_01")
        concentration = as_float(chem.get("concentration_ppm"))
        chem_alarm = as_float(chem.get("alarm_threshold"), 20.0)
        chem_warning = as_float(chem.get("warning_threshold"), 5.0)
        if _state(chem.get("alarm_state")) == "ALARM" or (concentration is not None and chem_alarm is not None and concentration >= chem_alarm):
            add("critical", "CHEM", "chem_sensor_01", f"Chemical concentration {concentration} ppm", substance=chem.get("substance"))
        elif concentration is not None and chem_warning is not None and concentration >= chem_warning:
            add("warning", "CHEM_WARNING", "chem_sensor_01", f"Chemical concentration {concentration} ppm")

        seismic = self.node("seysmo_01")
        if as_bool(seismic.get("quake_detected")) or _state(seismic.get("alarm_state")) == "ALARM":
            add("critical", "SEISMIC", "seysmo_01", "Seismic event detected")

        battery = self.node("battery_01")
        soc = as_float(battery.get("soc_pct"))
        if soc is not None and soc < 25:
            add("critical" if soc < 15 else "warning", "BATTERY_LOW", "battery_01", f"Battery SOC {soc:.1f}%")
        delta = as_float(battery.get("cell_voltage_delta_mv"))
        if delta is not None and delta > 50:
            add("warning", "BATTERY_IMBALANCE", "battery_01", f"Cell delta {delta:.1f} mV")

        inverter = self.node("solar_inverter_01")
        if inverter.get("error_code"):
            add("critical", "INVERTER_ERROR", "solar_inverter_01", f"Inverter error {inverter.get('error_code')}: {inverter.get('error_text') or ''}".strip())
        inv_temp = as_float(inverter.get("inverter_temp_c"))
        if inv_temp is not None and inv_temp > 75:
            add("critical", "INVERTER_HEAT", "solar_inverter_01", f"Inverter temperature {inv_temp:.1f} C")
        deficit = as_float(inverter.get("power_deficit_w"))
        if deficit is not None and deficit > 0:
            add("warning", "POWER_DEFICIT", "solar_inverter_01", f"Power deficit {deficit:.0f} W")

        diesel = self.node("dizel_1")
        if diesel.get("fault_code"):
            add("warning", "DIESEL_FAULT", "dizel_1", f"Diesel fault {diesel.get('fault_code')}: {diesel.get('fault_reason') or ''}".strip())
        fuel = self.node("fuel_tank_01")
        fuel_level = as_float(fuel.get("level_pct"))
        if fuel_level is not None and fuel_level < 20:
            add("warning", "FUEL_LOW", "fuel_tank_01", f"Fuel {fuel_level:.0f}%")
        water = as_float(fuel.get("water_content_ppm"))
        if water is not None and water > 500:
            add("warning", "FUEL_WATER", "fuel_tank_01", f"Fuel water content {water:.0f} ppm")

        tank = self.node("water_tank_01")
        level = as_float(tank.get("level_pct"))
        if level is not None and level < 20:
            add("critical" if level < 10 else "warning", "WATER_LOW", "water_tank_01", f"Water tank {level:.0f}%")
        water_filter = self.node("water_filter_01")
        if as_bool(water_filter.get("warning_active")) or _state(water_filter.get("filter_state")) not in {"", "OK"}:
            add("warning", "WATER_FILTER", "water_filter_01", f"Water filter {water_filter.get('filter_state')}")

        access = self.node("access_control_01")
        if as_bool(access.get("alarm")) or as_bool(access.get("lockdown")):
            add("critical", "ACCESS", "access_control_01", f"Access alarm: {access.get('alarm_reason') or 'lockdown'}")

        sealing = self.node("dome_sealing_01")
        if as_bool(sealing.get("leak_detected")):
            add("critical", "SEAL_LEAK", "dome_sealing_01", "Dome leak detected")
        elif sealing.get("fault_code"):
            add("warning", "SEAL_FAULT", "dome_sealing_01", f"Sealing fault {sealing.get('fault_code')}")

        fume = self.node("fume_extraction_01")
        if as_bool(fume.get("alarm")):
            add("critical", "FUME_ALARM", "fume_extraction_01", f"Fume extraction alarm {fume.get('alarm_code') or ''}".strip())
        airflow = as_float(fume.get("airflow_pct"))
        if airflow is not None and airflow < 50:
            add("warning", "FUME_FLOW", "fume_extraction_01", f"Fume airflow {airflow:.0f}%")
        motor_temp = as_float(fume.get("motor_temp_c"))
        if motor_temp is not None and motor_temp > 80:
            add("warning", "FUME_MOTOR_HEAT", "fume_extraction_01", f"Fume motor {motor_temp:.1f} C")

        ventilation = self.node("supply_ventilation_01")
        if as_bool(ventilation.get("alarm")):
            add("warning", "VENTILATION_ALARM", "supply_ventilation_01", f"Ventilation alarm {ventilation.get('alarm_code') or ''}".strip())
        clog = as_float(ventilation.get("filter_clog_pct"))
        if clog is not None and clog > 80:
            add("warning", "VENTILATION_FILTER", "supply_ventilation_01", f"Ventilation filter clog {clog:.0f}%")

        for line in self.panel_summary()["lines"]:
            number = line.get("line")
            node_id = f"smart_panel_01.line{number}"
            if line.get("alarm"):
                add("critical", "PANEL_LINE_ALARM", node_id, f"Line {number} alarm {line.get('alarm_code') or ''}".strip())
            protection = _state(line.get("protection"))
            if protection and protection != "ARMED":
                add("warning", "PANEL_PROTECTION", node_id, f"Line {number} protection {line.get('protection')}")
            voltage = as_float(self.line(int(number)).get("voltage_v")) if number is not None else None
            if voltage is not None and voltage > 245:
                add("critical", "OVERVOLTAGE", node_id, f"Line {number} voltage {voltage:.1f} V")
            current = line.get("current_a")
            rated = line.get("rated_a")
            if current is not None and rated and current > rated * 0.9:
                add("warning", "OVERCURRENT", node_id, f"Line {number} current {current:.2f}/{rated:.1f} A")

        for node_id in ("printer_3d_01", "cnc_01"):
            node = self.node(node_id)
            if node.get("error_code"):
                add("critical", "MACHINE_ERROR", node_id, f"{node_id} error {node.get('error_code')}")
            if as_bool(node.get("alarm")):
                add("critical", "MACHINE_ALARM", node_id, f"{node_id} alarm {node.get('alarm_code') or ''}".strip())
            if _state(node.get("hold_level")) == "ALARM":
                add("critical", "MACHINE_HOLD_ALARM", node_id, f"{node_id} hold ALARM: {node.get('hold_reason')}")
            elif _state(node.get("hold_level")) == "WARNING":
                add("warning", "MACHINE_HOLD_WARNING", node_id, f"{node_id} hold WARNING: {node.get('hold_reason')}")
            if node.get("part_defect"):
                add("warning", "PART_DEFECT", node_id, f"{node_id} part defect: {node.get('part_defect')}")

        weather = self.node("weather_station_01")
        wind = as_float(weather.get("wind_speed_ms"))
        if wind is not None and wind > 20:
            add("critical", "STORM", "weather_station_01", f"Wind {wind:.1f} m/s")
        rain = as_float(weather.get("precipitation_mm_h"))
        if rain is not None and rain > 0:
            add("warning", "PRECIPITATION", "weather_station_01", f"Precipitation {rain:.1f} mm/h")

        network = self.node("network_link_01")
        if _state(network.get("link_state")) in {"DOWN", "FAILED"}:
            add("critical", "NETWORK_DOWN", "network_link_01", "Main network link down")
        loss = as_float(network.get("packet_loss_pct"))
        if loss is not None and loss > 5:
            add("warning", "PACKET_LOSS", "network_link_01", f"Packet loss {loss:.1f}%")
        flaps = as_int(network.get("link_flaps"), 0) or 0
        if flaps > 3:
            add("warning", "LINK_FLAPS", "network_link_01", f"Link flaps {flaps}")

        inventory = self.node("material_inventory_01")
        for warning in inventory.get("low_stock_warnings") or []:
            add("warning", "LOW_STOCK", "material_inventory_01", str(warning))

        automation = self.node("dome_automation_01")
        for name, interlock in normalize_interlocks(automation.get("interlocks")).items():
            if not isinstance(interlock, dict):
                continue
            state = _state(interlock.get("state"))
            if state and state not in {"ARMED", "CLEAR", "NORMAL"}:
                add("warning", "INTERLOCK", "dome_automation_01", f"{name} {state}: {interlock.get('reason') or interlock.get('title') or ''}".strip())

        alerts.sort(key=lambda alert: -_SEVERITY_ORDER.get(str(alert.get("severity")), 0))
        self._alerts_cache = alerts
        return list(alerts)

    def brief(self) -> dict[str, Any]:
        """Compact, structured snapshot for logs and LLM prompts."""
        panel = self.panel_summary()
        production = self.production()
        energy = self.energy()
        climate = {
            "temperature_c": self.reading("climate_sensor_01", "temperature_c"),
            "humidity_pct": self.reading("climate_sensor_01", "humidity_pct"),
            "dew_point_c": self.reading("climate_sensor_01", "dew_point_c"),
            "co2": self.co2.as_dict(),
            "co_ppm": self.reading("air_quality_sensor_01", "co_ppm"),
            "voc_index": self.reading("air_quality_sensor_01", "voc_index"),
        }
        return {
            "clock": self.clock,
            "game_time": self.game_time,
            "panel": {
                "total_power_w": panel["total_power_w"],
                "bus_voltage_v": panel["bus_voltage_v"],
                "lines_on": panel["lines_on"],
                "lines": [
                    {
                        "line": line["line"],
                        "on": line["on"],
                        "power_w": line["power_w"],
                        "alarm_code": line["alarm_code"],
                        "protection": line["protection"],
                    }
                    for line in panel["lines"]
                ],
            },
            "energy": energy,
            "climate": climate,
            "production": production,
            "alerts": [
                {
                    "severity": alert["severity"],
                    "code": alert["code"],
                    "node_id": alert["node_id"],
                    "message": alert["message"],
                }
                for alert in self.alerts()
            ],
            "interlocks": normalize_interlocks(self.node("dome_automation_01").get("interlocks")),
            "recent_events": list(self.node("dome_automation_01").get("events") or [])[-5:],
        }

    def worst_severity(self) -> str:
        alerts = self.alerts()
        if not alerts:
            return "info"
        return str(alerts[0].get("severity") or "info")

    def has_code(self, *codes: str) -> bool:
        wanted = set(codes)
        return any(str(alert.get("code")) in wanted for alert in self.alerts())

    def find_alert(self, code: str) -> dict[str, Any] | None:
        return next((alert for alert in self.alerts() if alert.get("code") == code), None)


def merge_alerts(*collections: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for collection in collections:
        merged.extend(dict(alert) for alert in collection)
    merged.sort(key=lambda alert: -_SEVERITY_ORDER.get(str(alert.get("severity")), 0))
    return merged


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
