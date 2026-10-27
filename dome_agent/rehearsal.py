"""One-command offline rehearsal.

It chains the checks that can be completed without the live dome:

1. release pre-flight;
2. RAG evaluation on both question sets;
3. historical replay (all shifts, plus injected anomalies);
4. mock-dome failure scenarios;
5. optional model probe;
6. audit archive with hashes.

The 90-minute real-time run remains a separate acceptance step.  This command
records how long it actually ran, so a short rehearsal cannot be mistaken for
the full one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import audit, release, replay as replay_mod, scenarios
from .config import project_root

REPO = project_root()
DEFAULT_LOGS = REPO / "Dome_RAG-main" / "Logs"


def run_rehearsal(
    *,
    logs_dir: str | Path = DEFAULT_LOGS,
    include_scenarios: bool = True,
    include_probe: bool = True,
    report_path: str | Path | None = None,
    archive_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ok": True,
        "steps": {},
        "duration_s": 0.0,
    }

    check = release.check_all(run_tests=False)
    result["steps"]["release"] = {"ok": check["ok"], "checks": check["checks"]}
    result["ok"] = result["ok"] and check["ok"]

    try:
        evaluation = release.rag_eval()
        result["steps"]["rag_eval"] = {
            "ok": evaluation["ok"],
            "corpus_manifest_sha256": evaluation["corpus_manifest_sha256"],
            "sets": {
                name: {
                    key: metrics.get(key)
                    for key in (
                        "n",
                        "answer_ok",
                        "strict_answer_ok",
                        "recall@5",
                        "abstention_recall",
                        "abstention_precision",
                        "p95_ms",
                    )
                }
                for name, metrics in evaluation["sets"].items()
            },
        }
        result["ok"] = result["ok"] and evaluation["ok"]
    except Exception as exc:
        result["steps"]["rag_eval"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["ok"] = False

    try:
        historical = replay_mod.replay(
            logs_dir,
            policy_file=REPO / "dome_policy.competition.json",
            competition=True,
        )
        anomalous = replay_mod.replay(
            logs_dir,
            policy_file=REPO / "dome_policy.competition.json",
            competition=True,
            inject=list(replay_mod.INJECTIONS),
            days=[historical["days"][-1]["day"]] if historical["days"] else None,
        )
        result["steps"]["replay"] = {
            "ok": historical["totals"]["frames"] > 0,
            "historical": historical["totals"],
            "injected": anomalous["totals"],
            "injected_anomalies": (
                anomalous["days"][0]["anomalies"] if anomalous["days"] else {}
            ),
        }
        result["ok"] = result["ok"] and result["steps"]["replay"]["ok"]
    except Exception as exc:
        result["steps"]["replay"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["ok"] = False

    if include_scenarios:
        try:
            scenario_report = scenarios.run_all()
            result["steps"]["scenarios"] = {
                "ok": scenario_report["ok"],
                "items": [
                    {
                        "scenario": item["scenario"],
                        "ok": item["ok"],
                        "checks": item["checks"],
                    }
                    for item in scenario_report["scenarios"]
                ],
            }
            result["ok"] = result["ok"] and scenario_report["ok"]
        except Exception as exc:
            result["steps"]["scenarios"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            result["ok"] = False

    if include_probe:
        try:
            result["steps"]["model_probe"] = release.probe_model()
        except Exception as exc:
            result["steps"]["model_probe"] = {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    result["duration_s"] = round(time.monotonic() - started, 2)
    target = Path(report_path) if report_path else REPO / "logs" / "rehearsal.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    result["report"] = str(target)

    try:
        archive = audit.write_archive(
            archive_path,
            log_dir=REPO / "logs",
            state_file=REPO / "logs" / "agent_state.json",
            policy_file=REPO / "dome_policy.competition.json",
            kb_dir=REPO / "dome_rag",
            extras=[target],
        )
        result["archive"] = str(archive)
    except OSError as exc:
        result["archive_error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dome-rehearsal",
        description="Run the full offline rehearsal and write a JSON report + audit archive.",
    )
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS))
    parser.add_argument("--no-scenarios", action="store_true", help="skip mock-dome failure scenarios")
    parser.add_argument("--no-probe", action="store_true", help="skip the DeepSeek model probe")
    parser.add_argument("--out", default="logs/rehearsal.json")
    parser.add_argument("--archive", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_rehearsal(
        logs_dir=args.logs_dir,
        include_scenarios=not args.no_scenarios,
        include_probe=not args.no_probe,
        report_path=args.out,
        archive_path=args.archive,
    )
    for name, step in report["steps"].items():
        print(f"{'PASS' if step.get('ok', step.get('status') in {'ok', 'skipped'}) else 'FAIL'} {name}")
    print(f"duration={report['duration_s']}s report={report['report']}")
    if report.get("archive"):
        print(f"archive={report['archive']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
