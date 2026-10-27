"""Reproducible audit archive for a shift or a rehearsal.

The archive is intentionally small and deterministic enough for an appeal:
the newest agent JSONL, the restart state, the exact policy, the corpus
manifest and a hash manifest.  Secrets are never included; only a whitelist of
non-secret environment values is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ENV_WHITELIST = (
    "AGENT_COMPETITION",
    "AGENT_MODE",
    "AGENT_ALLOW_CONTROL",
    "AGENT_DRY_RUN",
    "AGENT_LLM",
    "AGENT_POLICY_FILE",
    "AGENT_LOG_DIR",
    "AGENT_STATE_FILE",
    "AGENT_KB_DIR",
    "AGENT_KB_ENABLED",
    "AGENT_KB_DB",
    "AGENT_RUN_SECONDS",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _newest(log_dir: Path) -> Path | None:
    logs = [path for path in log_dir.glob("agent_*.jsonl") if path.is_file()]
    return max(logs, key=lambda path: path.stat().st_mtime, default=None)


def _panel_logs(log_dir: Path) -> list[Path]:
    return sorted(path for path in log_dir.glob("panel_*.jsonl") if path.is_file())


def _safe_env() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ENV_WHITELIST:
        value = os.getenv(name)
        if value is not None:
            result[name] = value
    return result


def build_manifest(
    *,
    log_dir: str | Path = "logs",
    state_file: str | Path = "logs/agent_state.json",
    policy_file: str | Path = "dome_policy.competition.json",
    kb_dir: str | Path = "dome_rag",
    extras: Iterable[str | Path] = (),
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    log_dir = Path(log_dir).expanduser()
    state_file = Path(state_file).expanduser()
    policy_file = Path(policy_file).expanduser()
    kb_dir = Path(kb_dir).expanduser()

    members: list[tuple[Path, str]] = []
    agent_log = _newest(log_dir)
    if agent_log is not None:
        members.append((agent_log, "logs/agent.jsonl"))
    for index, path in enumerate(_panel_logs(log_dir), 1):
        members.append((path, f"logs/panel_{index}.jsonl"))
    if state_file.is_file():
        members.append((state_file, "state/agent_state.json"))
    if policy_file.is_file():
        members.append((policy_file, "config/dome_policy.json"))
    manifest_file = kb_dir / "corpus_manifest.json"
    if manifest_file.is_file():
        members.append((manifest_file, "rag/corpus_manifest.json"))
    db_file = kb_dir / "build" / "kb.sqlite"
    if db_file.is_file():
        # Hash only; the archive must stay small enough for an appeal upload.
        members.append((db_file, "rag/kb.sqlite.sha256.txt"))
    for extra in extras:
        path = Path(extra).expanduser()
        if path.is_file():
            members.append((path, f"extra/{path.name}"))

    files: dict[str, Any] = {}
    for path, arcname in members:
        digest = sha256_file(path)
        files[arcname] = {
            "source": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size if path.exists() else None,
        }
    manifest: dict[str, Any] = {
        "archive_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cwd": str(Path.cwd()),
        "git_head": _git_head(),
        "env": _safe_env(),
        "files": files,
        "policy_sha256": sha256_file(policy_file) if policy_file.is_file() else None,
        "corpus_manifest_sha256": (
            sha256_file(manifest_file) if manifest_file.is_file() else None
        ),
        "kb_sqlite_sha256": sha256_file(db_file) if db_file.is_file() else None,
        "python": sys.version.split()[0],
    }
    return manifest, members


def _git_head() -> str | None:
    try:
        head = Path(".git/HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head.startswith("ref: "):
        ref = Path(".git") / head[5:]
        try:
            return ref.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return head or None


def write_archive(
    output: str | Path | None = None,
    *,
    log_dir: str | Path = "logs",
    state_file: str | Path = "logs/agent_state.json",
    policy_file: str | Path = "dome_policy.competition.json",
    kb_dir: str | Path = "dome_rag",
    extras: Iterable[str | Path] = (),
) -> Path:
    manifest, members = build_manifest(
        log_dir=log_dir,
        state_file=state_file,
        policy_file=policy_file,
        kb_dir=kb_dir,
        extras=extras,
    )
    if output is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output = Path("logs") / f"dome-audit-{stamp}.tar.gz"
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(target, "w:gz") as archive:
        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, indent=2, default=str
        ).encode("utf-8")
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now().timestamp())
        archive.addfile(info, io.BytesIO(manifest_bytes))

        for path, arcname in members:
            if arcname.endswith(".sha256.txt"):
                text = (manifest["files"][arcname]["sha256"] or "") + "\n"
                payload = text.encode("utf-8")
                info = tarfile.TarInfo(arcname)
                info.size = len(payload)
                info.mtime = int(path.stat().st_mtime)
                archive.addfile(info, io.BytesIO(payload))
            else:
                archive.add(path, arcname=arcname, recursive=False)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dome-audit",
        description="Build a compact, hash-manifested audit archive (no secrets).",
    )
    parser.add_argument("--out", default=None, help="output .tar.gz path")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--state-file", default="logs/agent_state.json")
    parser.add_argument("--policy", default="dome_policy.competition.json")
    parser.add_argument("--kb-dir", default="dome_rag")
    parser.add_argument("--extra", action="append", default=None, help="extra file to include (repeatable)")
    parser.add_argument("--json", action="store_true", help="print the manifest instead of only the path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.json:
            manifest, _ = build_manifest(
                log_dir=args.log_dir,
                state_file=args.state_file,
                policy_file=args.policy,
                kb_dir=args.kb_dir,
                extras=args.extra or (),
            )
            print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
            return 0
        path = write_archive(
            args.out,
            log_dir=args.log_dir,
            state_file=args.state_file,
            policy_file=args.policy,
            kb_dir=args.kb_dir,
            extras=args.extra or (),
        )
    except OSError as exc:
        print(f"audit error: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
