"""CLI control panel for the Dome WebSocket API.

Examples:
    python -m dome_control status
    python -m dome_control nodes
    python -m dome_control control smart_panel_01 line_off 8
    python -m dome_control query printer_3d_01 list_files
    python -m dome_control say "Печать запущена"
    python -m dome_control watch

The module is intentionally read/control oriented. It is the transport and
operator-console part of the future autonomous agent, not the agent itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

if __package__:
    from .client import MISSING, DomeAuthError, DomeClient, DomeError, DomeTimeout, retry_policy
else:  # allow `python dome_control/panel.py ...` as well as `python -m dome_control ...`
    from client import MISSING, DomeAuthError, DomeClient, DomeError, DomeTimeout, retry_policy

DEFAULT_NAME = "team3-panel"
RULE = "-" * 100


# --- small formatting helpers ------------------------------------------------


def _get(node: dict[str, Any], key: str, default: Any = "—") -> Any:
    value = node.get(key)
    return default if value is None else value


def _num(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{value}{unit}"
    return f"{number:.{digits}f}{unit}"


def _bool(value: Any) -> str:
    if value is None:
        return "—"
    return "да" if value else "нет"


def _value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _shorten(text: Any, limit: int = 120) -> str:
    value = str(text).replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _game_time(data: dict[str, Any]) -> str:
    game_time = data.get("game_time") or {}
    if not game_time:
        return "—"
    day = game_time.get("day", "—")
    if game_time.get("hour") is not None:
        hour = float(game_time["hour"])
        hours = int(hour) % 24
        minutes = int(round((hour - int(hour)) * 60))
        if minutes == 60:
            hours = (hours + 1) % 24
            minutes = 0
        return f"сутки {day}, {hours:02d}:{minutes:02d}"
    if game_time.get("time"):
        return f"сутки {day}, {game_time['time']}"
    return f"сутки {day}"


# --- section formatters ------------------------------------------------------


def _format_energy(nodes: dict[str, Any]) -> str:
    solar = nodes.get("solar_panels_01") or {}
    inverter = nodes.get("solar_inverter_01") or {}
    battery = nodes.get("battery_01") or {}
    wind = nodes.get("wind_turbine_01") or {}
    diesel = nodes.get("dizel_1") or {}
    fuel = nodes.get("fuel_tank_01") or {}

    parts = [
        f"PV {_num(solar.get('power_w'), 0)} Вт"
        f" (MPPT {_get(solar, 'active_mppt_count')}/{_get(solar, 'channels_count')})",
        f"ветер {_num(wind.get('power_w'), 0)} Вт",
        f"АКБ {_num(battery.get('soc_pct'), 0)}% {_get(battery, 'state')}",
        f"инвертор {_get(inverter, 'mode')}, нагрузка {_num(inverter.get('load_power_w'), 0)} Вт",
        f"сеть {_get(inverter, 'grid_state')} {_num(inverter.get('grid_power_w'), 0)} Вт",
        f"дизель {_get(diesel, 'state')}, топливо {_num(fuel.get('level_pct'), 0)}%",
    ]
    if inverter.get("power_deficit_w"):
        parts.append(f"дефицит {_num(inverter.get('power_deficit_w'), 0)} Вт")
    if inverter.get("error"):
        parts.append(f"ОШИБКА {inverter.get('error_text') or inverter.get('error_code')}")
    if diesel.get("fault_code"):
        parts.append(f"дизель fault {diesel.get('fault_code')}")
    if inverter.get("inverter_temp_c") is not None and float(inverter["inverter_temp_c"]) > 60:
        parts.append(f"инвертор {_num(inverter.get('inverter_temp_c'), 1)}°C")
    return " · ".join(parts)


def _format_panel(nodes: dict[str, Any]) -> str:
    panel = nodes.get("smart_panel_01") or {}
    lines = panel.get("lines") or []
    header = (
        f"{_num(panel.get('total_power_w'), 0)} Вт · "
        f"шина {_num(panel.get('bus_voltage_v'), 1)} В · "
        f"линий ON {_get(panel, 'lines_on_count')}"
    )
    if not lines:
        return header

    rendered = [header]
    for line in lines:
        if not isinstance(line, dict):
            continue
        state = _get(line, "state")
        bits = [
            f"L{_get(line, 'line')}",
            f"{_num(line.get('rated_current_a'), 0)}A",
            state,
            f"{_num(line.get('power_w'), 1)} Вт",
        ]
        name = line.get("name")
        if name:
            bits.append(str(name))
        protection = line.get("protection_state")
        if protection and protection != "ARMED":
            bits.append(f"[{protection}]")
        if line.get("lockout_remaining_s"):
            bits.append(f"lockout {_num(line['lockout_remaining_s'], 0)}с")
        if line.get("alarm"):
            bits.append(f"ALARM {_get(line, 'alarm_code', '')}".strip())
        rendered.append("  " + " · ".join(bits))
    return "\n".join(rendered)


def _format_climate(nodes: dict[str, Any]) -> str:
    climate = nodes.get("climate_sensor_01") or {}
    air = nodes.get("air_quality_sensor_01") or {}
    vent = nodes.get("supply_ventilation_01") or {}
    fume = nodes.get("fume_extraction_01") or {}

    first = " · ".join(
        [
            f"t {_num(climate.get('temperature_c'), 1)}°C",
            f"RH {_num(climate.get('humidity_pct'), 0)}%",
            f"CO2 {_num(climate.get('co2_ppm'), 0)} ppm",
            f"CO {_num(air.get('co_ppm'), 2)} ppm",
            f"VOC {_num(air.get('voc_index'), 0)}",
        ]
    )
    vent_state = _get(vent, "state")
    if vent_state == "—" and vent.get("enabled") is not None:
        vent_state = "ON" if vent.get("enabled") else "OFF"
    filter_state = _get(vent, "filter_state")
    if filter_state == "—" and vent.get("filter_ok") is not None:
        filter_state = "OK" if vent.get("filter_ok") else "ЗАБИТ"
    second = (
        f"приток {vent_state} {_get(vent, 'mode')} "
        f"{_num(vent.get('airflow_m3_h'), 0)} м³/ч · фильтр {filter_state}"
    )
    if vent.get("filter_clog_pct") is not None:
        second += f" · засор {_num(vent.get('filter_clog_pct'), 0)}%"
    if vent.get("heat_exchanger_efficiency_pct") is not None:
        second += f" · рекуперация {_num(vent.get('heat_exchanger_efficiency_pct'), 0)}%"

    fume_state = _get(fume, "state")
    if fume_state == "—" and fume.get("enabled") is not None:
        fume_state = "ON" if fume.get("enabled") else "OFF"
    third = f"вытяжка {fume_state}"
    if fume.get("airflow_pct") is not None:
        third += f" · поток {_num(fume.get('airflow_pct'), 0)}%"
    if fume.get("fan_rpm") is not None:
        third += f" · RPM {_num(fume.get('fan_rpm'), 0)}"
    return "\n".join([first, second, third])


def _machine_parts(node: dict[str, Any]) -> list[str]:
    parts = [str(_get(node, "state", _get(node, "status")))]
    if node.get("progress_pct") is not None:
        parts.append(f"{_num(node.get('progress_pct'), 0)}%")

    file_name = node.get("file_name") or node.get("job_file")
    if not file_name:
        files = node.get("files") or []
        if files:
            first = files[0]
            file_name = first.get("name") if isinstance(first, dict) else first
    if file_name:
        parts.append(f"файл {_shorten(file_name, 70)}")

    hold = node.get("hold_level")
    if hold:
        parts.append(f"HOLD {hold}/{_get(node, 'hold_reason')}")
    if node.get("resume_attempts"):
        parts.append(f"resume={node['resume_attempts']}")
    if node.get("lockout_remaining_s"):
        parts.append(f"lockout {_num(node['lockout_remaining_s'], 0)}с")
    if node.get("bed_clear") is False:
        parts.append("стол занят")
    if node.get("error"):
        parts.append(f"ОШИБКА {_get(node, 'error_code', '')}".strip())
    if node.get("alarm"):
        parts.append(f"ALARM {_get(node, 'alarm_code', '')}".strip())
    return parts


def _format_production(nodes: dict[str, Any]) -> str:
    printer = nodes.get("printer_3d_01") or {}
    cnc = nodes.get("cnc_01") or {}

    printer_parts = _machine_parts(printer)
    if printer.get("nozzle_temp_c") is not None:
        printer_parts.append(
            f"сопло {_num(printer.get('nozzle_temp_c'), 0)}"
            f"/{_num(printer.get('nozzle_target_c'), 0)}°C"
        )
    if printer.get("bed_temp_c") is not None:
        printer_parts.append(
            f"стол {_num(printer.get('bed_temp_c'), 0)}"
            f"/{_num(printer.get('bed_target_c'), 0)}°C"
        )
    if printer.get("z_mm") is not None:
        printer_parts.append(f"Z {_num(printer.get('z_mm'), 1)} мм")

    cnc_parts = _machine_parts(cnc)
    if cnc.get("spindle_rpm") is not None:
        cnc_parts.append(f"шпиндель {_num(cnc.get('spindle_rpm'), 0)} об/мин")
    if cnc.get("x_mm") is not None:
        cnc_parts.append(
            f"X {_num(cnc.get('x_mm'), 1)} Y {_num(cnc.get('y_mm'), 1)} "
            f"Z {_num(cnc.get('z_mm'), 1)}"
        )

    return "принтер: " + " · ".join(printer_parts) + "\nЧПУ: " + " · ".join(cnc_parts)


def _format_greenhouse(node: dict[str, Any], label: str) -> str:
    parts = [label]
    temperatures = node.get("temperature_c") or []
    humidity = node.get("humidity_pct") or []
    fans = node.get("fans_pwm") or []
    doors = node.get("doors") or []
    locks = node.get("locks")
    if locks is None:
        lock = node.get("lock")
        locks = [lock] if lock is not None else []

    if temperatures:
        parts.append("t " + "/".join(_num(v, 1) for v in temperatures) + "°C")
    if humidity:
        parts.append("RH " + "/".join(_num(v, 0) for v in humidity) + "%")
    if fans:
        parts.append("вент [" + ", ".join(_num(v, 0) for v in fans) + "]")
    if doors:
        parts.append("двери [" + ", ".join(str(v) for v in doors) + "]")
    if locks:
        parts.append("замки [" + ", ".join(str(v) for v in locks) + "]")
    if node.get("relay_fan_level") is not None:
        parts.append(f"relay {node['relay_fan_level']}/10")
    if node.get("vacuum_on") is not None:
        parts.append("пылесос " + ("ON" if node.get("vacuum_on") else "OFF"))
    return " · ".join(parts)


def _format_greenhouses(nodes: dict[str, Any]) -> str:
    fablab = nodes.get("thermal_insulation_01") or {}
    chemlab = nodes.get("thermal_insulation_02") or {}
    return "\n".join(
        [
            _format_greenhouse(fablab, "ФабЛаб"),
            _format_greenhouse(chemlab, "Химлаб"),
        ]
    )


def _format_water(nodes: dict[str, Any]) -> str:
    tank = nodes.get("water_tank_01") or {}
    pump = nodes.get("water_pump_01") or {}
    water_filter = nodes.get("water_filter_01") or {}

    filter_state = _get(water_filter, "filter_state", _get(water_filter, "state"))
    wear = water_filter.get("wear_pct")
    if wear is None:
        wear = water_filter.get("filter_life_pct")
    parts = [
        f"бак {_num(tank.get('level_pct'), 0)}% ({_num(tank.get('volume_l'), 0)} л)",
        f"насос {_get(pump, 'state')} {_num(pump.get('flow_l_min'), 1)} л/мин "
        f"{_num(pump.get('pressure_bar'), 1)} бар {_num(pump.get('power_w'), 0)} Вт",
        f"фильтр {filter_state} износ {_num(wear, 0)}%",
    ]
    if water_filter.get("pressure_drop_bar") is not None:
        parts.append(f"ΔP {_num(water_filter.get('pressure_drop_bar'), 2)} бар")
    if water_filter.get("hours_to_replacement") is not None:
        parts.append(f"до замены {_num(water_filter.get('hours_to_replacement'), 0)} ч")
    return " · ".join(parts)


def _format_doors(access: dict[str, Any]) -> str:
    doors = access.get("doors") or {}
    if isinstance(doors, dict):
        items = doors.items()
    elif isinstance(doors, list):
        items = ((door.get("id", "?"), door) for door in doors if isinstance(door, dict))
    else:
        return ""
    parts = []
    for name, door in items:
        if not isinstance(door, dict):
            continue
        parts.append(
            f"{name} {_get(door, 'state')}/{_get(door, 'locked')} "
            f"АКБ {_num(door.get('lock_battery_pct'), 0)}%"
        )
    return " · ".join(parts)


def _format_safety(nodes: dict[str, Any]) -> str:
    sealing = nodes.get("dome_sealing_01") or {}
    fire = nodes.get("fire_suppression_01") or {}
    smoke_fablab = nodes.get("smoke_detector_01") or {}
    smoke_storage = nodes.get("smoke_detector_02") or {}
    access = nodes.get("access_control_01") or {}
    radiation = nodes.get("radiation_sensor_01") or {}
    chem = nodes.get("chem_sensor_01") or {}

    def smoke_state(node: dict[str, Any]) -> str:
        if node.get("smoke_detected"):
            return "ТРЕВОГА"
        alarm = node.get("alarm_state")
        if isinstance(alarm, str):
            return "OK" if alarm in ("OK", "NORMAL") else "ТРЕВОГА"
        return "ТРЕВОГА" if alarm is True else "OK"
    first = " · ".join(
        [
            f"герметизация {_get(sealing, 'state')}",
            f"ΔP {_num(sealing.get('pressure_diff_pa'), 0)} Па",
            f"утечка {_num(sealing.get('leak_rate_pct'), 1)}%",
            f"клапаны {_value(sealing.get('valve_positions_pct'))}",
            f"пожар {_get(fire, 'state')} {_num(fire.get('pressure_bar'), 1)} бар",
            f"зона {_get(fire, 'active_zone')}",
            f"дым FABLAB {smoke_state(smoke_fablab)} / STORAGE {smoke_state(smoke_storage)}",
        ]
    )
    second = (
        f"доступ {_get(access, 'system_state')} · lockdown {_bool(access.get('lockdown'))}"
    )
    doors = _format_doors(access)
    if doors:
        second += " · " + doors
    third = (
        f"радиация {_num(radiation.get('dose_rate_usv_h'), 2)} мкЗв/ч"
        f" ({_get(radiation, 'alarm_state')}) · "
        f"химия {_get(chem, 'substance')} {_num(chem.get('concentration_ppm'), 2)} ppm"
        f" ({_get(chem, 'alarm_state')})"
    )
    return "\n".join([first, second, third])


def _format_comms(nodes: dict[str, Any]) -> str:
    network = nodes.get("network_link_01") or {}
    radio = nodes.get("radio_01") or {}
    backup = nodes.get("backup_comms_01") or {}
    edge = nodes.get("edge_compute_01") or {}

    first = " · ".join(
        [
            f"сеть {_get(network, 'active_path')} {_get(network, 'link_state')}",
            f"потери {_num(network.get('packet_loss_pct'), 1)}%",
            f"RX {_num(network.get('rx_power_dbm'), 2)} dBm",
            f"резерв {_get(network, 'reserve_link_state')}",
        ]
    )
    second = " · ".join(
        [
            f"радио {_num(radio.get('frequency_mhz'), 1)} МГц {_get(radio, 'protocol')} "
            f"{_num(radio.get('signal_quality_pct'), 0)}%",
            f"резерв {_get(backup, 'active_channel')}/{_get(backup, 'protocol')} "
            f"{_num(backup.get('latency_ms'), 0)} мс",
            f"edge CPU {_num(edge.get('cpu_load_pct'), 0)}% "
            f"GPU {_num(edge.get('gpu_load_pct'), 0)}%",
        ]
    )
    return first + "\n" + second


def _format_automation(nodes: dict[str, Any]) -> str:
    automation = nodes.get("dome_automation_01") or {}
    interlocks = automation.get("interlocks") or {}
    active = [
        f"{name}={_value(state.get('state') if isinstance(state, dict) else state)}"
        for name, state in interlocks.items()
        if (state.get("state") if isinstance(state, dict) else state) not in (None, "ARMED")
    ]
    line = f"режим {_get(automation, 'mode')} · интерлоки: " + (
        ", ".join(active) if active else "все ARMED"
    )
    trust = automation.get("data_trust") or {}
    if trust:
        line += " · доверие: " + ", ".join(f"{key}={value}" for key, value in trust.items())
    if automation.get("primary_co2_sensor"):
        line += f" · основной CO2 {automation['primary_co2_sensor']}"
    if automation.get("autonomy_forecast_h") is not None:
        line += f" · автономность {_num(automation.get('autonomy_forecast_h'), 1)} ч"
    if automation.get("ventilation_auto") is not None:
        line += f" · приток авто {_bool(automation.get('ventilation_auto'))}"
    if automation.get("diesel_autostart") is not None:
        line += f" · дизель автостарт {_bool(automation.get('diesel_autostart'))}"

    events = automation.get("events") or []
    if events:
        recent = []
        for event in events[-3:]:
            if not isinstance(event, dict):
                continue
            text = " ".join(
                str(part)
                for part in (event.get("type"), event.get("interlock"), event.get("message"))
                if part
            )
            recent.append(_shorten(text, 100))
        if recent:
            line += "\n  события: " + " | ".join(recent)
    return line


def _format_chat(nodes: dict[str, Any]) -> str:
    meeting = nodes.get("meeting_01") or {}
    messages = meeting.get("recent_messages") or []
    if not meeting and not messages:
        return ""
    parts = [f"сообщений {_get(meeting, 'message_count', len(messages))}"]
    for message in messages[-3:]:
        if not isinstance(message, dict):
            continue
        sender = message.get("sender") or {}
        role = sender.get("role") or "?"
        name = sender.get("name") or role
        parts.append(f"[{role}/{name}] {_shorten(message.get('text'), 110)}")
    return "\n".join(parts)


def _format_alerts(nodes: dict[str, Any]) -> str:
    alerts: list[str] = []
    for node_id, node in sorted(nodes.items()):
        if not isinstance(node, dict):
            continue
        if node.get("online") is False:
            alerts.append(f"{node_id}: offline")
        if node.get("status") == "UNKNOWN":
            alerts.append(f"{node_id}: status UNKNOWN (устаревшие данные)")
        if node.get("hold_level"):
            alerts.append(
                f"{node_id}: hold {node['hold_level']} ({_get(node, 'hold_reason')})"
            )
        if node.get("alarm"):
            code = node.get("alarm_code") or node.get("alarm_reason") or ""
            alerts.append(f"{node_id}: alarm {code}".strip())
        if node.get("error"):
            code = node.get("error_code") or node.get("error_text") or ""
            alerts.append(f"{node_id}: error {code}".strip())
        if node.get("fault_code"):
            alerts.append(f"{node_id}: fault {node['fault_code']}")
        if node.get("smoke_detected"):
            alerts.append(f"{node_id}: дым")
        alarm_state = node.get("alarm_state")
        if isinstance(alarm_state, str) and alarm_state not in ("OK", "NORMAL", "ARMED"):
            alerts.append(f"{node_id}: {alarm_state}")
        elif alarm_state is True:
            alerts.append(f"{node_id}: alarm")
        if node.get("lockdown"):
            alerts.append(f"{node_id}: lockdown")
        if node.get("leak_detected"):
            alerts.append(f"{node_id}: утечка")

    panel = nodes.get("smart_panel_01") or {}
    for line in panel.get("lines") or []:
        if not isinstance(line, dict):
            continue
        problems: list[str] = []
        if line.get("alarm"):
            problems.append(str(line.get("alarm_code") or line.get("alarm_reason") or "alarm"))
        protection = line.get("protection_state")
        if protection and protection != "ARMED":
            problems.append(f"protection {protection}")
        if problems:
            alerts.append(f"L{line.get('line')}: {', '.join(problems)}")

    climate = nodes.get("climate_sensor_01") or {}
    co2 = climate.get("co2_ppm")
    if isinstance(co2, (int, float)):
        if co2 > 2000:
            alerts.append(f"CO2 {co2:.0f} ppm > 2000 (порог IL_CO2)")
        elif co2 > 1000:
            alerts.append(f"CO2 {co2:.0f} ppm повышен")

    battery = nodes.get("battery_01") or {}
    soc = battery.get("soc_pct")
    if isinstance(soc, (int, float)) and soc < 25:
        alerts.append(f"АКБ {soc:.0f}% < 25% (риск load shed)")

    fuel = nodes.get("fuel_tank_01") or {}
    fuel_level = fuel.get("level_pct")
    if isinstance(fuel_level, (int, float)) and fuel_level < 20:
        alerts.append(f"топливо {fuel_level:.0f}% < 20%")

    tank = nodes.get("water_tank_01") or {}
    water_level = tank.get("level_pct")
    if isinstance(water_level, (int, float)) and water_level < 20:
        alerts.append(f"вода {water_level:.0f}% < 20%")

    fire = nodes.get("fire_suppression_01") or {}
    if fire.get("active_zone"):
        alerts.append(f"пожаротушение активно: {fire['active_zone']}")

    access = nodes.get("access_control_01") or {}
    if access.get("alarm"):
        alerts.append(f"доступ: {access.get('alarm_reason') or 'alarm'}")

    network = nodes.get("network_link_01") or {}
    loss = network.get("packet_loss_pct")
    if isinstance(loss, (int, float)) and loss > 5:
        alerts.append(f"сеть: потери {loss:.1f}%")

    unique = list(dict.fromkeys(alerts))
    if not unique:
        return "  явных тревог нет"
    return "\n".join(f"  ! {item}" for item in unique)


# --- public formatters -------------------------------------------------------


def format_status(
    data: dict[str, Any],
    *,
    role: str | None = None,
    team: int | None = None,
) -> str:
    nodes = data.get("nodes") or {}
    header = f"КУПОЛ  {_game_time(data)}"
    if role:
        header += f"   роль: {role}"
        if team is not None:
            header += f", команда {team}"

    sections: list[tuple[str, str]] = [
        ("ЭНЕРГИЯ", _format_energy(nodes)),
        ("ЩИТОК", _format_panel(nodes)),
        ("КЛИМАТ", _format_climate(nodes)),
        ("ПРОИЗВОДСТВО", _format_production(nodes)),
        ("ТЕПЛИЦЫ", _format_greenhouses(nodes)),
        ("ВОДА", _format_water(nodes)),
        ("БЕЗОПАСНОСТЬ", _format_safety(nodes)),
        ("СВЯЗЬ", _format_comms(nodes)),
        ("АВТОМАТИКА", _format_automation(nodes)),
        ("ЧАТ", _format_chat(nodes)),
        ("ТРЕВОГИ", _format_alerts(nodes)),
    ]

    output = [header]
    for title, body in sections:
        if body:
            output.extend(["", title, body])
    return "\n".join(output)


def _node_summary(node_id: str, node: dict[str, Any]) -> str:
    if node_id == "solar_panels_01":
        return f"PV {_num(node.get('power_w'), 0)} Вт, MPPT {_get(node, 'active_mppt_count')}"
    if node_id == "solar_inverter_01":
        return (
            f"{_get(node, 'mode')} · нагрузка {_num(node.get('load_power_w'), 0)} Вт · "
            f"сеть {_get(node, 'grid_state')}"
        )
    if node_id == "battery_01":
        return f"{_num(node.get('soc_pct'), 0)}% {_get(node, 'state')} · {_num(node.get('power_w'), 0)} Вт"
    if node_id == "smart_panel_01":
        return (
            f"{_num(node.get('total_power_w'), 0)} Вт · "
            f"линий ON {_get(node, 'lines_on_count')}"
        )
    if node_id == "printer_3d_01":
        return " · ".join(_machine_parts(node))
    if node_id == "cnc_01":
        return " · ".join(_machine_parts(node))
    if node_id == "climate_sensor_01":
        return (
            f"{_num(node.get('temperature_c'), 1)}°C · "
            f"RH {_num(node.get('humidity_pct'), 0)}% · "
            f"CO2 {_num(node.get('co2_ppm'), 0)} ppm"
        )
    if node_id.startswith("thermal_insulation_"):
        temperatures = node.get("temperature_c") or []
        fans = node.get("fans_pwm") or []
        return (
            f"t {'/'.join(_num(v, 1) for v in temperatures)}°C · "
            f"вент [{', '.join(_num(v, 0) for v in fans)}]"
        )
    if node_id == "water_tank_01":
        return f"{_num(node.get('level_pct'), 0)}% ({_num(node.get('volume_l'), 0)} л)"
    if node_id == "weather_station_01":
        return (
            f"{_num(node.get('temperature_c'), 1)}°C · "
            f"ветер {_num(node.get('wind_speed_ms'), 1)} м/с"
        )

    keys = (
        "state",
        "status",
        "power_w",
        "temperature_c",
        "level_pct",
        "soc_pct",
        "pressure_bar",
        "airflow_m3_h",
        "alarm_state",
    )
    bits = [
        f"{key}={_value(node[key])}"
        for key in keys
        if node.get(key) is not None
    ]
    return " · ".join(bits[:3]) or "—"


def format_nodes(catalog: list[dict[str, Any]], data: dict[str, Any]) -> str:
    nodes = data.get("nodes") or {}
    by_id = {
        node.get("node_id"): node
        for node in catalog
        if isinstance(node, dict) and node.get("node_id")
    }
    order = [node_id for node_id in by_id if node_id in nodes]
    order.extend(node_id for node_id in nodes if node_id not in by_id)

    lines = [f"{'node':<28} {'on':<3} {'status':<9} summary"]
    for node_id in order:
        node = nodes.get(node_id)
        if not isinstance(node, dict):
            continue
        online = "ON" if node.get("online", True) else "OFF"
        status = str(_get(node, "status"))
        summary = _node_summary(node_id, node)
        title = by_id.get(node_id, {}).get("title")
        if title:
            summary += f"  — {title}"
        lines.append(f"{node_id:<28} {online:<3} {status:<9} {summary}")
    return "\n".join(lines)


def format_describe(catalog: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for node in catalog:
        if not isinstance(node, dict):
            continue
        lines.append(
            f"{node.get('node_id')} — {node.get('title', '')} "
            f"[{node.get('system', '')}/{node.get('category', '')}]"
        )
        controls = node.get("controls") or []
        if controls:
            lines.append("  controls: " + ", ".join(controls))
        aliases = node.get("control_aliases") or {}
        alias_text = [
            f"{name} -> {', '.join(values)}"
            for name, values in aliases.items()
            if values
        ]
        if alias_text:
            lines.append("  aliases: " + "; ".join(alias_text))
        operator_controls = node.get("operator_controls") or []
        if operator_controls:
            lines.append("  operator: " + ", ".join(operator_controls))
        queries = node.get("queries") or []
        if queries:
            lines.append("  queries: " + ", ".join(queries))
        service_actions = node.get("service_actions") or []
        if service_actions:
            lines.append("  service: " + ", ".join(service_actions))
        lines.append("")
    return "\n".join(lines).rstrip()


def format_chat_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        sender = message.get("sender") or {}
        role = sender.get("role") or "?"
        team = sender.get("team")
        name = sender.get("name") or role
        prefix = f"#{message.get('id', '?')} [{role}"
        if team is not None:
            prefix += f"/team{team}"
        prefix += f"/{name}]"
        if message.get("to"):
            prefix += f" -> {message['to']}"
        lines.append(f"{prefix}: {message.get('text', '')}")
    return "\n".join(lines) or "сообщений нет"


def _format_response(response: dict[str, Any]) -> str:
    if response.get("success"):
        return f"OK · mode={response.get('mode', '—')}"
    return f"ОШИБКА: {response.get('error', '—')}"


# --- configuration -----------------------------------------------------------


def load_dotenv(path: str | None) -> None:
    if not path:
        return
    candidate = Path(path)
    if not candidate.exists():
        return
    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_team_key(path: str) -> str | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("team:"):
            return line.split(":", 1)[1].strip()
    return None


def _arg(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def resolve_settings(args: argparse.Namespace) -> tuple[str, str, str, float]:
    load_dotenv(_arg(args, "env_file", ".env"))
    url = str(_arg(args, "url") or os.getenv("DOME_URL") or "").strip()
    key = str(_arg(args, "key") or os.getenv("DOME_KEY") or "").strip()
    key_file = str(_arg(args, "key_file") or os.getenv("DOME_KEY_FILE") or "").strip()
    if not key and key_file:
        key = read_team_key(key_file) or ""
    if not url:
        raise SystemExit("Не задан DOME_URL (--url или переменная окружения).")
    if not key:
        raise SystemExit("Не задан DOME_KEY (--key, DOME_KEY или --key-file).")
    name = os.getenv("AGENT_NAME", DEFAULT_NAME)
    timeout = float(_arg(args, "timeout", 30.0) or 30.0)
    return url, key, name, timeout


@asynccontextmanager
async def _client(args: argparse.Namespace) -> AsyncIterator[DomeClient]:
    url, key, name, timeout = resolve_settings(args)
    client = DomeClient(
        url,
        key,
        name=name,
        request_timeout=timeout,
        open_timeout=min(timeout, 10.0),
        auth_timeout=min(timeout, 10.0),
    )
    await client.connect_with_retry(*retry_policy(timeout))
    try:
        yield client
    finally:
        await client.close()


def parse_value(raw: str | None) -> Any:
    if raw is None:
        return MISSING
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


# --- demo mode ---------------------------------------------------------------


def load_demo_telemetry(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        raise SystemExit(f"Файл телеметрии не найден: {path}")

    with candidate.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 4_000_000))
        chunk = handle.read().decode("utf-8", errors="replace")

    for raw_line in reversed(chunk.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "telemetry" and "nodes" not in record:
            continue
        nodes = record.get("nodes") or (record.get("data") or {}).get("nodes")
        if not nodes:
            continue
        game_time = record.get("dome") or (record.get("data") or {}).get("game_time")
        if isinstance(game_time, dict) and "time" in game_time and "hour" not in game_time:
            parts = str(game_time["time"]).split(":")
            try:
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
            except (TypeError, IndexError, ValueError):
                pass
            else:
                game_time = {
                    "day": game_time.get("day"),
                    "hour": hours + minutes / 60,
                }
        return {"nodes": nodes, "game_time": game_time}
    raise SystemExit(f"В файле нет строки telemetry: {path}")


# --- commands ----------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    command = args.command

    if command == "status":
        if args.demo:
            data = load_demo_telemetry(args.demo)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print(format_status(data, role="demo", team=3))
            return 0
        async with _client(args) as client:
            data = await client.wait_for_telemetry(timeout=args.timeout)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print(format_status(data, role=client.role, team=client.team))
            if client.role == "guest":
                print(
                    "\nРоль guest: доступен только meeting_01, телеметрия купола закрыта.",
                    file=sys.stderr,
                )
            return 0

    if command == "nodes":
        if args.demo:
            data = load_demo_telemetry(args.demo)
            if args.json:
                print(
                    json.dumps(
                        {"catalog": [], "nodes": data.get("nodes", {})},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(format_nodes([], data))
            return 0
        async with _client(args) as client:
            catalog = await client.describe_nodes()
            data = await client.wait_for_telemetry(timeout=args.timeout)
            if args.json:
                print(
                    json.dumps(
                        {"catalog": catalog, "nodes": data.get("nodes", {})},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(format_nodes(catalog, data))
            return 0

    if command == "describe":
        async with _client(args) as client:
            catalog = await client.describe_nodes()
            if args.json:
                print(json.dumps(catalog, ensure_ascii=False, indent=2))
            else:
                print(format_describe(catalog))
            return 0

    if command == "control":
        value = parse_value(args.value)
        payload: dict[str, Any] = {
            "type": "control",
            "node_id": args.node_id,
            "action": args.action,
        }
        if value is not MISSING:
            payload["value"] = value
        if args.demo:
            print("DRY-RUN:", json.dumps(payload, ensure_ascii=False))
            return 0
        async with _client(args) as client:
            response = await client.control(args.node_id, args.action, value)
            if args.json:
                print(json.dumps(response, ensure_ascii=False, indent=2))
            else:
                print(_format_response(response))
            if not response.get("success"):
                return 1
            if not args.no_verify:
                try:
                    data = await client.wait_for_telemetry(
                        timeout=min(args.timeout, 10.0),
                        next_frame=True,
                    )
                except DomeError as exc:
                    print(f"\nПроверка не удалась: {exc}", file=sys.stderr)
                    return 0
                node = (data.get("nodes") or {}).get(args.node_id)
                print("\nТелеметрия после команды:")
                if node is None:
                    print("  узел отсутствует в телеметрии")
                elif args.json:
                    print(json.dumps(node, ensure_ascii=False, indent=2))
                else:
                    print("  " + _node_summary(args.node_id, node))
            return 0

    if command == "query":
        params = parse_value(args.params) if args.params else {}
        if not isinstance(params, dict):
            raise SystemExit("params должен быть JSON-объектом.")
        payload = {
            "type": "query",
            "node_id": args.node_id,
            "query": args.query,
            "params": params,
        }
        if args.demo:
            print("DRY-RUN:", json.dumps(payload, ensure_ascii=False))
            return 0
        async with _client(args) as client:
            response = await client.query(args.node_id, args.query, params)
            if args.json:
                print(json.dumps(response, ensure_ascii=False, indent=2))
            elif not response.get("success"):
                print(_format_response(response))
            else:
                print(json.dumps(response.get("result"), ensure_ascii=False, indent=2))
            return 0 if response.get("success") else 1

    if command == "say":
        if args.demo:
            payload = {
                "type": "control",
                "node_id": "meeting_01",
                "action": "post_message",
                "value": {
                    "text": args.text,
                    "to": args.to,
                    "name": args.name or DEFAULT_NAME,
                },
            }
            print("DRY-RUN:", json.dumps(payload, ensure_ascii=False))
            return 0
        async with _client(args) as client:
            response = await client.post_message(
                args.text,
                to=args.to,
                name=args.name,
            )
            if args.json:
                print(json.dumps(response, ensure_ascii=False, indent=2))
            else:
                print(_format_response(response))
            return 0 if response.get("success") else 1

    if command == "chat":
        async with _client(args) as client:
            messages = await client.read_messages(
                since_id=args.since_id,
                limit=args.limit,
            )
            if args.json:
                print(json.dumps(messages, ensure_ascii=False, indent=2))
            else:
                print(format_chat_messages(messages))
            return 0

    if command == "watch":
        async with _client(args) as client:
            data = await client.wait_for_telemetry(timeout=args.timeout)
            while True:
                if args.raw:
                    print(json.dumps(data, ensure_ascii=False))
                else:
                    print(format_status(data, role=client.role, team=client.team))
                    print(RULE)
                data = await client.wait_for_telemetry(
                    timeout=max(args.interval, 1.0) + 15.0,
                    next_frame=True,
                )

    if command == "send":
        try:
            payload = json.loads(args.payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Некорректный JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("payload должен быть JSON-объектом.")
        if args.demo:
            print("DRY-RUN:", json.dumps(payload, ensure_ascii=False))
            return 0
        async with _client(args) as client:
            response = await client.request(**payload)
            print(json.dumps(response, ensure_ascii=False, indent=2))
            return 0 if response.get("success", True) else 1

    if command == "ping":
        async with _client(args) as client:
            response = await client.ping()
            print(json.dumps(response, ensure_ascii=False, indent=2))
            return 0 if response.get("success") else 1

    raise SystemExit(f"Неизвестная команда: {command}")


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS, help="ws://host:8765")
    common.add_argument("--key", default=argparse.SUPPRESS, help="team key")
    common.add_argument(
        "--key-file",
        default=argparse.SUPPRESS,
        help="file containing 'team: <key>'",
    )
    common.add_argument("--env-file", default=argparse.SUPPRESS, help="path to .env")
    common.add_argument(
        "--timeout",
        type=float,
        default=argparse.SUPPRESS,
        help="request timeout, seconds",
    )
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="raw JSON output",
    )
    common.add_argument(
        "--demo",
        default=argparse.SUPPRESS,
        metavar="TELEMETRY_JSONL",
        help="offline demo from a telemetry.jsonl file",
    )

    parser = argparse.ArgumentParser(
        prog="dome-panel",
        description="Dome WebSocket API control panel (Team 3)",
    )
    parser.add_argument("--url", help="ws://host:8765 (or DOME_URL)")
    parser.add_argument("--key", help="team key (or DOME_KEY)")
    parser.add_argument(
        "--key-file",
        help="file containing 'team: <key>' (or DOME_KEY_FILE)",
    )
    parser.add_argument("--env-file", default=".env", help="path to .env")
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout, seconds")
    parser.add_argument("--json", action="store_true", help="raw JSON output")
    parser.add_argument(
        "--demo",
        metavar="TELEMETRY_JSONL",
        help="offline demo from a telemetry.jsonl file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", parents=[common], help="current dome summary")
    subparsers.add_parser("nodes", parents=[common], help="node catalog and live state")
    subparsers.add_parser("describe", parents=[common], help="full describe_nodes catalog")

    control = subparsers.add_parser(
        "control",
        parents=[common],
        help="send a control action",
    )
    control.add_argument("node_id")
    control.add_argument("action")
    control.add_argument("value", nargs="?", help="JSON value; bare text is a string")
    control.add_argument(
        "--no-verify",
        action="store_true",
        help="do not wait for the next telemetry frame",
    )

    query = subparsers.add_parser("query", parents=[common], help="run a node query")
    query.add_argument("node_id")
    query.add_argument("query")
    query.add_argument("params", nargs="?", help="JSON object")

    say = subparsers.add_parser("say", parents=[common], help="post a message to meeting_01")
    say.add_argument("text")
    say.add_argument("--to", default="operator")
    say.add_argument("--name", default=None)

    chat = subparsers.add_parser("chat", parents=[common], help="read meeting_01 messages")
    chat.add_argument("--limit", type=int, default=50)
    chat.add_argument("--since-id", type=int, default=None)

    watch = subparsers.add_parser(
        "watch",
        parents=[common],
        help="print status on every telemetry frame",
    )
    watch.add_argument("--interval", type=float, default=2.0)
    watch.add_argument("--raw", action="store_true")

    send = subparsers.add_parser("send", parents=[common], help="send a raw JSON request")
    send.add_argument("payload")

    subparsers.add_parser("ping", parents=[common], help="ping the server")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nОстановлено.", file=sys.stderr)
        return 130
    except DomeAuthError as exc:
        print(f"Ошибка авторизации: {exc}", file=sys.stderr)
        return 2
    except DomeTimeout as exc:
        print(f"Таймаут: {exc}", file=sys.stderr)
        return 1
    except (DomeError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
