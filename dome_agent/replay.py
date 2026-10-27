"""Offline replay of historical shifts through the deterministic rules.

The live dome is not required for this check.  Historical telemetry from
``Dome_RAG-main/Logs/<day>/telemetry.jsonl`` is fed through the same
:class:`~dome_agent.analysis.Snapshot` and :class:`~dome_agent.rules.RuleEngine`
that the real agent uses, while every proposed action is checked against the
policy gate.

The replay deliberately does not send commands, call an LLM or write to the
agent state.  It answers a narrower question: **which decisions and commands
would the rules produce on real data, and which of them would be blocked?**

Anomaly injection (``--inject stale,duplicate,out_of_order,corrupt``) makes the
same report cover the failure modes that are otherwise hard to reproduce on a
live stand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .analysis import Snapshot
from .control import is_auto_protective
from .gate import CommandGate, GateError
from .rules import RuleEngine
from .state import StateStore

STALE_AFTER_S = 180.0
NODE_STALE_AFTER_S = 180.0
DEFAULT_MIN_GAP_S = 0.35
INJECTIONS = ("stale", "duplicate", "out_of_order", "corrupt")


class _Corrupt:
    """Marker for a line that must be counted as invalid JSON."""


CORRUPT = _Corrupt()


def _parse_ts(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.timestamp()


def _frame_hash(record: Mapping[str, Any]) -> str:
    payload = {
        "nodes": record.get("nodes"),
        "dome": record.get("dome") or record.get("game_time"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a historical log record into Snapshot input."""
    data = dict(record)
    if "game_time" not in data and isinstance(data.get("dome"), Mapping):
        data["game_time"] = dict(data["dome"])
    return data


@dataclass
class TelemetryFrame:
    day: str
    index: int
    ts: float | None
    data: dict[str, Any]
    anomalies: list[str] = field(default_factory=list)


@dataclass
class DayReplay:
    day: str
    frames: list[TelemetryFrame]
    invalid_lines: int = 0
    source_bytes: int = 0


def load_day(day_dir: Path, *, inject: Iterable[str] = ()) -> DayReplay:
    """Read one shift.  Corrupt lines are counted, not raised."""
    path = Path(day_dir) / "telemetry.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"telemetry log not found: {path}")
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    inject = tuple(dict.fromkeys(str(item) for item in inject))
    unknown = set(inject) - set(INJECTIONS)
    if unknown:
        raise ValueError(f"unknown injection(s): {', '.join(sorted(unknown))}")

    records: list[dict[str, Any] | _Corrupt] = []
    invalid = 0
    for line in raw_lines:
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(value, dict):
            invalid += 1
            continue
        records.append(value)

    if "corrupt" in inject and records:
        # Insert a syntactically invalid line in the middle of the shift.
        records.insert(len(records) // 2, CORRUPT)
    if "duplicate" in inject and records:
        for cut in sorted({len(records) // 3, 2 * len(records) // 3}, reverse=True):
            if cut < len(records):
                records.insert(cut, json.loads(json.dumps(records[cut - 1])))
    if "out_of_order" in inject and len(records) > 4:
        for cut in (len(records) // 4, 3 * len(records) // 4):
            if isinstance(records[cut - 1], dict) and isinstance(records[cut], dict):
                records[cut - 1], records[cut] = records[cut], records[cut - 1]

    frames: list[TelemetryFrame] = []
    seen_hashes: set[str] = set()
    previous_ts: float | None = None
    injected_stale = False
    for value in records:
        if value is CORRUPT:
            invalid += 1
            continue
        assert isinstance(value, dict)
        ts = _parse_ts(value.get("ts"))
        anomalies: list[str] = []
        frame_hash = _frame_hash(value)
        if frame_hash in seen_hashes:
            anomalies.append("duplicate")
        seen_hashes.add(frame_hash)
        if ts is not None and previous_ts is not None:
            if ts < previous_ts:
                anomalies.append("out_of_order")
            elif ts - previous_ts > STALE_AFTER_S:
                anomalies.append("stale")
        if ts is not None:
            previous_ts = ts
        if "stale" in inject and frames and not injected_stale:
            # Make one frame visibly stale without changing the source data.
            anomalies.append("stale")
            injected_stale = True
        frames.append(
            TelemetryFrame(
                day=Path(day_dir).name,
                index=len(frames),
                ts=ts,
                data=_normalize(value),
                anomalies=anomalies,
            )
        )
    return DayReplay(
        day=Path(day_dir).name,
        frames=frames,
        invalid_lines=invalid,
        source_bytes=path.stat().st_size,
    )


def _node_staleness(frame: TelemetryFrame) -> int:
    if frame.ts is None:
        return 0
    nodes = frame.data.get("nodes")
    if not isinstance(nodes, Mapping):
        return 0
    stale = 0
    for node in nodes.values():
        if not isinstance(node, Mapping):
            continue
        updated = node.get("updated_at")
        if isinstance(updated, (int, float)):
            if frame.ts - float(updated) > NODE_STALE_AFTER_S:
                stale += 1
    return stale


class _Clock:
    def __init__(self) -> None:
        self.value = time.time()

    def set(self, value: float | None) -> float:
        if value is not None:
            self.value = max(self.value if self.value else value, value)
        return self.value

    def __call__(self) -> float:
        return self.value


def _action_ref(action: Any) -> str:
    value = getattr(action, "value", None)
    suffix = "" if value.__class__.__name__ == "object" else f"={value!r}"
    return f"{getattr(action, 'node_id', '')}.{getattr(action, 'action', '')}{suffix}"


def replay_day(
    day: DayReplay,
    *,
    policy_file: Path,
    competition: bool = False,
    min_send_gap_s: float = DEFAULT_MIN_GAP_S,
) -> dict[str, Any]:
    """Run rules and policy over one shift and return a reproducible report."""
    gate = CommandGate(
        policy_file,
        control_enabled=True,
        competition=competition,
    )
    clock = _Clock()
    with tempfile.TemporaryDirectory(prefix="dome-replay-") as tmp:
        state = StateStore(Path(tmp) / "state.json")
        rules = RuleEngine(state, clock=clock)
        report: dict[str, Any] = {
            "day": day.day,
            "source_bytes": day.source_bytes,
            "frames": len(day.frames),
            "invalid_lines": day.invalid_lines,
            "anomalies": {name: 0 for name in INJECTIONS},
            "stale_nodes": 0,
            "decisions": 0,
            "action_requests": 0,
            "policy_allowed": 0,
            "policy_blocked": 0,
            "auto_protective": 0,
            "confirmation_required": 0,
            "max_actions_per_frame": 0,
            "situations": {},
            "blocked": {},
            "actions": {},
            "examples": [],
        }
        last_send = 0.0
        for frame in day.frames:
            for name in frame.anomalies:
                report["anomalies"][name] = report["anomalies"].get(name, 0) + 1
            report["stale_nodes"] += _node_staleness(frame)
            clock.set(frame.ts)
            snapshot = Snapshot(frame.data, received_at=clock())
            result = rules.decide(snapshot, state.task)
            report["decisions"] += 1
            report["situations"][result.situation] = (
                report["situations"].get(result.situation, 0) + 1
            )
            report["max_actions_per_frame"] = max(
                report["max_actions_per_frame"], len(result.actions)
            )
            for action in result.actions:
                report["action_requests"] += 1
                key = _action_ref(action)
                report["actions"][key] = report["actions"].get(key, 0) + 1
                decision = gate.check(action)
                if decision.allowed:
                    report["policy_allowed"] += 1
                    send_at = max(clock(), last_send + min_send_gap_s)
                    last_send = send_at
                    protective = is_auto_protective(action)
                    report["auto_protective"] += int(protective)
                    report["confirmation_required"] += int(not protective)
                else:
                    report["policy_blocked"] += 1
                    report["blocked"][decision.reason] = (
                        report["blocked"].get(decision.reason, 0) + 1
                    )
                if len(report["examples"]) < 12:
                    report["examples"].append(
                        {
                            "day": frame.day,
                            "index": frame.index,
                            "ts": frame.ts,
                            "situation": result.situation,
                            "action": key,
                            "allowed": decision.allowed,
                            "reason": decision.reason,
                            "protective": is_auto_protective(action),
                        }
                    )
        return report


def replay(
    logs_dir: str | Path,
    *,
    policy_file: str | Path = "dome_policy.competition.json",
    competition: bool = True,
    inject: Iterable[str] = (),
    min_send_gap_s: float = DEFAULT_MIN_GAP_S,
    days: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(logs_dir)
    selected = sorted(p.name for p in root.iterdir() if (p / "telemetry.jsonl").is_file())
    if days:
        wanted = {str(day) for day in days}
        selected = [day for day in selected if day in wanted]
    reports = []
    for day in selected:
        loaded = load_day(root / day, inject=inject)
        reports.append(
            replay_day(
                loaded,
                policy_file=Path(policy_file),
                competition=competition,
                min_send_gap_s=min_send_gap_s,
            )
        )
    totals = {
        "days": len(reports),
        "frames": sum(item["frames"] for item in reports),
        "invalid_lines": sum(item["invalid_lines"] for item in reports),
        "decisions": sum(item["decisions"] for item in reports),
        "action_requests": sum(item["action_requests"] for item in reports),
        "policy_allowed": sum(item["policy_allowed"] for item in reports),
        "policy_blocked": sum(item["policy_blocked"] for item in reports),
        "auto_protective": sum(item["auto_protective"] for item in reports),
        "confirmation_required": sum(item["confirmation_required"] for item in reports),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "logs_dir": str(root),
        "policy_file": str(policy_file),
        "competition": competition,
        "injections": list(inject),
        "min_send_gap_s": min_send_gap_s,
        "totals": totals,
        "days": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dome-replay",
        description="Replay historical shifts through rules and the policy gate (offline).",
    )
    parser.add_argument("--logs-dir", default="Dome_RAG-main/Logs", help="directory with <day>/telemetry.jsonl")
    parser.add_argument("--policy", default="dome_policy.competition.json", help="policy file")
    parser.add_argument("--day", action="append", default=None, help="limit to one or more days")
    parser.add_argument(
        "--inject",
        default="",
        help=f"comma-separated scenarios: {', '.join(INJECTIONS)}",
    )
    parser.add_argument("--min-send-gap", type=float, default=DEFAULT_MIN_GAP_S)
    parser.add_argument("--allow-unsafe-policy", action="store_true", help="do not enforce the competition policy")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inject = [part.strip() for part in str(args.inject or "").split(",") if part.strip()]
    try:
        report = replay(
            args.logs_dir,
            policy_file=args.policy,
            competition=not args.allow_unsafe_policy,
            inject=inject,
            min_send_gap_s=args.min_send_gap,
            days=args.day,
        )
    except (OSError, ValueError, GateError) as exc:
        print(f"replay error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    if not args.quiet:
        totals = report["totals"]
        print(
            f"replay: {totals['days']} days, {totals['frames']} frames, "
            f"{totals['decisions']} decisions, {totals['action_requests']} actions"
        )
        print(
            f"policy: allowed={totals['policy_allowed']} "
            f"blocked={totals['policy_blocked']} "
            f"protective={totals['auto_protective']} "
            f"confirm={totals['confirmation_required']} "
            f"invalid={totals['invalid_lines']}"
        )
        for day in report["days"]:
            anomalies = {k: v for k, v in day["anomalies"].items() if v}
            print(
                f"  {day['day']}: frames={day['frames']} actions={day['action_requests']} "
                f"blocked={day['policy_blocked']} anomalies={anomalies or '{}'}"
            )
        if not args.out:
            print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
