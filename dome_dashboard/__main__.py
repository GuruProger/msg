"""CLI entry point for the read-only dashboard server."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import uvicorn


def _default_port() -> int:
    """Read ``DASHBOARD_PORT`` and fall back to 8080 when it is invalid."""
    raw = os.getenv("DASHBOARD_PORT", "8080")
    try:
        return int(raw)
    except ValueError:
        return 8080


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dashboard with uvicorn."""
    parser = argparse.ArgumentParser(
        prog="dome-dashboard",
        description="Read-only Dome observability dashboard.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("DASHBOARD_HOST", "127.0.0.1"),
        help="bind host (default: DASHBOARD_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_default_port(),
        help="bind port (default: DASHBOARD_PORT or 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="enable uvicorn auto-reload for development",
    )
    args = parser.parse_args(argv)
    uvicorn.run(
        "dome_dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
