"""Pre-flight checks, RAG evaluation and a minimal model probe.

This module is deliberately operational: it answers "can this exact checkout
start a scored run?" without a live dome.  It does not replace the live
acceptance tests; it catches configuration, artifact and packaging mistakes
that are easy to miss after a refactor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import (
    COMPETITION_BASE_URL,
    COMPETITION_MODELS,
    AgentConfig,
    file_sha256,
    project_root,
)
from .gate import GateError, load_policy, validate_competition_policy

REPO = project_root()
ENTRYPOINTS = {
    "dome-agent": ("dome_agent.agent", "main"),
    "dome-panel": ("dome_control.cli", "main"),
    "dome-web": ("dome_web.dashboard_backend", "main"),
}
RAG_ARTIFACTS = (
    "dome_rag/data/cards.json",
    "dome_rag/build/kb.sqlite",
    "dome_rag/build/vectors.json",
    "dome_rag/build/card_vectors.json",
    "dome_rag/corpus_manifest.json",
)


def _truth(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on", "y"}


def _check(name: str, ok: bool, detail: str = "", *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": required,
        "detail": str(detail)[:500],
    }


def _entrypoint_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, (module_name, attr) in ENTRYPOINTS.items():
        try:
            module = importlib.import_module(module_name)
            target = getattr(module, attr)
            checks.append(_check(f"entrypoint:{name}", callable(target), f"{module_name}.{attr}"))
        except Exception as exc:
            checks.append(
                _check(f"entrypoint:{name}", False, f"{type(exc).__name__}: {exc}")
            )
    return checks


def _rag_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name in RAG_ARTIFACTS:
        path = REPO / name
        checks.append(
            _check(
                f"rag:{name}",
                path.is_file(),
                f"{path} ({path.stat().st_size if path.is_file() else 0} bytes)",
            )
        )
    manifest = REPO / "dome_rag/corpus_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            checks.append(
                _check(
                    "rag:manifest",
                    bool(data.get("corpus_hash")) and int((data.get("counts") or {}).get("documents") or 0) > 0,
                    f"corpus_hash={data.get('corpus_hash')} documents={(data.get('counts') or {}).get('documents')}",
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(_check("rag:manifest", False, f"{type(exc).__name__}: {exc}"))
    return checks


def _secret_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    gitignore = REPO / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    lines = {line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
    required_patterns = {
        ".env": {".env", ".env.*"},
        "logs/": {"logs", "logs/"},
        "build/": {"build/", "build"},
        "Keys/": {"**/Keys/", "Keys/", "keys/"},
    }
    for label, variants in required_patterns.items():
        checks.append(
            _check(
                f"gitignore:{label}",
                bool(lines & variants),
                f"{label} ignored",
            )
        )

    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(REPO),
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return checks + [_check("git:tracked", False, f"{type(exc).__name__}: {exc}", required=False)]
    if tracked.returncode != 0:
        return checks + [_check("git:tracked", False, tracked.stderr.decode("utf-8", "replace")[:300], required=False)]
    names = [item for item in tracked.stdout.decode("utf-8", "replace").split("\0") if item]
    risky: list[str] = []
    patterns = (
        re.compile(r"(?m)^[ \t]*DEEPSEEK_API_KEY[ \t]*=[ \t]*\S+"),
        re.compile(r"(?m)^[ \t]*DOME_KEY[ \t]*=[ \t]*\S+"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    )
    for name in names:
        path = REPO / name
        if not path.is_file() or path.stat().st_size > 256_000:
            continue
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".docx", ".sqlite"}:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(pattern.search(body) for pattern in patterns):
            risky.append(name)
    checks.append(
        _check(
            "git:no-secrets",
            not risky,
            "clean" if not risky else "possible secret in: " + ", ".join(risky[:8]),
        )
    )
    checks.append(
        _check(
            "git:env-not-tracked",
            ".env" not in names,
            ".env is not tracked",
        )
    )
    return checks


def _policy_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    enabled = _truth("AGENT_COMPETITION")
    policy_name = os.getenv("AGENT_POLICY_FILE", "dome_policy.json")
    policy = REPO / policy_name
    checks.append(_check("policy:exists", policy.is_file(), str(policy)))
    if not policy.is_file():
        return checks
    try:
        data = load_policy(policy)
    except GateError as exc:
        return checks + [_check("policy:load", False, str(exc))]
    errors = validate_competition_policy(data)
    checks.append(
        _check(
            "policy:competition-safe",
            not errors if enabled else True,
            ("safe" if not errors else "; ".join(errors))
            if enabled
            else "not enforced (development profile)",
            required=enabled,
        )
    )
    return checks


def _competition_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    enabled = _truth("AGENT_COMPETITION")
    checks.append(
        _check(
            "competition:enabled",
            True,
            "AGENT_COMPETITION=1" if enabled else "not enabled (development profile)",
            required=False,
        )
    )
    if not enabled:
        return checks
    mode = os.getenv("AGENT_MODE", "normal").strip().casefold()
    checks.append(_check("competition:mode", mode in {"normal", "auto"}, mode))
    checks.append(_check("competition:no-yolo", not _truth("AGENT_YOLO"), "AGENT_YOLO must be 0"))
    llm = _truth("AGENT_LLM", True)
    model = os.getenv("DEEPSEEK_MODEL", "").strip()
    if llm:
        checks.append(
            _check(
                "competition:model",
                model in COMPETITION_MODELS,
                f"{model!r} in {list(COMPETITION_MODELS)}",
            )
        )
        base = os.getenv("DEEPSEEK_BASE_URL", COMPETITION_BASE_URL).rstrip("/")
        checks.append(_check("competition:base-url", base == COMPETITION_BASE_URL, base))
    else:
        checks.append(_check("competition:rules-only", True, "AGENT_LLM=0"))
    try:
        config = AgentConfig.from_env(env_file=None)
        checks.append(_check("competition:config", True, f"version={config.version}"))
    except Exception as exc:
        checks.append(_check("competition:config", False, f"{type(exc).__name__}: {exc}"))
    return checks


def check_all(run_tests: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    py_ok = sys.version_info >= (3, 14)
    checks.append(
        _check(
            "python",
            py_ok,
            f"{platform.python_version()} (requires >=3.14)",
        )
    )
    checks.extend(_entrypoint_checks())
    checks.extend(_rag_checks())
    checks.extend(_policy_checks())
    checks.extend(_competition_checks())
    checks.extend(_secret_checks())
    if run_tests:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        checks.append(
            _check(
                "tests:unittest",
                completed.returncode == 0,
                " | ".join(tail),
            )
        )
    required = [item for item in checks if item["required"]]
    return {
        "ok": all(item["ok"] for item in required),
        "python": platform.python_version(),
        "repo": str(REPO),
        "checks": checks,
    }


def rag_eval(output: str | Path | None = None, k: int = 5) -> dict[str, Any]:
    from dome_rag.builder.evaluator import evaluate, load_questions
    from dome_rag.builder.retrieval import KB

    manifest = REPO / "dome_rag/corpus_manifest.json"
    report: dict[str, Any] = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus_manifest": str(manifest),
        "corpus_manifest_sha256": file_sha256(manifest),
        "k": k,
        "sets": {},
        "ok": True,
    }
    kb = KB()
    try:
        for name in ("questions.json", "questions_hard.json"):
            path = REPO / "dome_rag/eval" / name
            questions = load_questions(path)
            metrics = evaluate(kb, questions, k=k)
            summary = {key: value for key, value in metrics.items() if key != "rows"}
            report["sets"][name] = summary
            if metrics.get("answer_ok", 0) < 0.8:
                report["ok"] = False
    finally:
        kb.close()
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return report


def probe_model(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    import httpx

    base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", COMPETITION_BASE_URL)).rstrip("/")
    model = (model or os.getenv("DEEPSEEK_MODEL", "")).strip()
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "").strip()
    report: dict[str, Any] = {
        "base_url": base_url,
        "requested_model": model,
        "configured_models": list(COMPETITION_MODELS),
        "status": "skipped",
    }
    if not api_key:
        report["reason"] = "DEEPSEEK_API_KEY is empty; declare AGENT_LLM=0"
        return report
    if not model:
        report["status"] = "error"
        report["reason"] = "DEEPSEEK_MODEL is empty"
        return report
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 2,
        "temperature": 0,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        report["status"] = "error"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return report
    report["http_status"] = response.status_code
    try:
        data = response.json()
    except ValueError:
        report["status"] = "error"
        report["reason"] = "response is not JSON"
        return report
    actual = data.get("model")
    report["reported_model"] = actual
    report["ok"] = bool(response.status_code == 200 and actual)
    report["status"] = "ok" if report["ok"] else "error"
    report["matches_request"] = bool(actual == model)
    if not report["ok"]:
        report["reason"] = str(data.get("error") or data)[:500]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dome-release",
        description="Offline release checks, RAG evaluation and model probe.",
    )
    sub = parser.add_subparsers(dest="command")
    check = sub.add_parser("check", help="run pre-flight checks")
    check.add_argument("--tests", action="store_true", help="also run the unittest suite")
    check.add_argument("--json", action="store_true")

    eval_p = sub.add_parser("rag-eval", help="evaluate the RAG question sets")
    eval_p.add_argument("--out", default="dome_rag/build/reports/eval.json")
    eval_p.add_argument("--k", type=int, default=5)
    eval_p.add_argument("--json", action="store_true")

    probe = sub.add_parser("probe", help="minimal DeepSeek API probe")
    probe.add_argument("--base-url", default=None)
    probe.add_argument("--model", default=None)
    probe.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "check"
    if command == "check":
        report = check_all(run_tests=bool(getattr(args, "tests", False)))
        if getattr(args, "json", False):
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            for item in report["checks"]:
                mark = "PASS" if item["ok"] else ("FAIL" if item["required"] else "WARN")
                print(f"[{mark}] {item['name']}: {item['detail']}")
            print(f"release check: {'ok' if report['ok'] else 'failed'}")
        return 0 if report["ok"] else 1
    if command == "rag-eval":
        report = rag_eval(args.out, k=args.k)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            for name, metrics in report["sets"].items():
                print(
                    f"{name}: n={metrics['n']} answer_ok={metrics['answer_ok']:.3f} "
                    f"recall@5={metrics['recall@5']:.3f} "
                    f"abstention_recall={metrics['abstention_recall']:.3f} "
                    f"p95={metrics['p95_ms']:.1f} ms"
                )
            print(f"corpus_manifest_sha256={report['corpus_manifest_sha256']}")
            print(f"wrote {args.out}")
        return 0 if report["ok"] else 1
    if command == "probe":
        report = probe_model(base_url=args.base_url, model=args.model)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") or report.get("status") == "skipped" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
