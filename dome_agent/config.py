"""Configuration for the autonomous Dome agent.

Only environment variables (and an optional ``.env`` file) are used.  The
configuration object deliberately keeps secrets out of ``repr`` and exposes a
sanitized mapping that is safe to write to the agent log.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from dome_control.panel import load_dotenv, read_team_key

AGENT_VERSION = "0.2.0"
DEFAULT_NAME = "team3-agent"
DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return default if value is None else value


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().casefold()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = _env(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _env(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _path_env(name: str, default: Path) -> Path:
    raw = _env(name)
    return Path(raw).expanduser() if raw and raw.strip() else default


def _clean_url(value: str) -> str:
    return value.strip().rstrip("/")


def _load_key(file_name: str | None) -> str:
    if not file_name:
        return ""
    return read_team_key(file_name) or ""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Immutable runtime configuration.

    ``dome_key`` and ``deepseek_api_key`` are excluded from ``repr``.  They are
    also excluded from :meth:`safe_dict`, the only mapping written to logs.
    """

    dome_url: str
    dome_key: str = field(repr=False)
    dome_key_file: str | None = field(default=None, repr=False)
    agent_name: str = DEFAULT_NAME
    deepseek_api_key: str | None = field(default=None, repr=False)
    deepseek_base_url: str = DEFAULT_DEEPSEEK_URL
    deepseek_model: str = DEFAULT_MODEL
    deepseek_timeout: float = 30.0
    llm_enabled: bool = True
    llm_min_interval_s: float = 30.0
    llm_max_calls_per_hour: int = 60
    log_dir: Path = Path("logs")
    state_file: Path = Path("logs/agent_state.json")
    dry_run: bool = False
    control_enabled: bool = False
    mode: str = "normal"
    policy_file: Path = Path("dome_policy.json")
    telemetry_timeout_s: float = 10.0
    observation_interval_s: float = 45.0
    reconnect_delay_s: float = 5.0
    guest_recheck_s: float = 60.0
    min_send_gap_s: float = 0.35
    run_seconds: float = 0.0
    version: str = AGENT_VERSION

    def __post_init__(self) -> None:
        url = _clean_url(self.dome_url)
        parts = urlsplit(url)
        if parts.scheme.casefold() not in {"ws", "wss"} or not parts.netloc:
            raise ValueError(
                f"DOME_URL must be ws://host:port or wss://host, got {self.dome_url!r}"
            )
        if not self.dome_key.strip():
            raise ValueError("DOME_KEY is empty; set DOME_KEY or DOME_KEY_FILE")
        if not self.agent_name.strip():
            raise ValueError("AGENT_NAME must not be empty")
        if self.deepseek_timeout <= 0:
            raise ValueError("DEEPSEEK_TIMEOUT must be > 0")
        if self.telemetry_timeout_s <= 0:
            raise ValueError("AGENT_TELEMETRY_TIMEOUT must be > 0")
        if self.observation_interval_s < 5:
            raise ValueError("AGENT_OBSERVATION_INTERVAL must be >= 5")
        if self.min_send_gap_s < 0.2:
            # The API contract is at most five commands/requests per second.
            raise ValueError("AGENT_MIN_SEND_GAP must be >= 0.2 seconds")
        if self.llm_min_interval_s < 0:
            raise ValueError("AGENT_LLM_MIN_INTERVAL must be >= 0")
        if self.llm_max_calls_per_hour < 1:
            raise ValueError("AGENT_LLM_MAX_CALLS_PER_HOUR must be >= 1")
        if self.run_seconds < 0:
            raise ValueError("AGENT_RUN_SECONDS must be >= 0")
        mode = str(self.mode or "").strip().casefold()
        if mode not in {"normal", "auto", "yolo"}:
            raise ValueError(
                f"AGENT_MODE must be normal, auto or yolo, got {self.mode!r}"
            )
        object.__setattr__(self, "mode", mode)
        # YOLO is an explicit opt-in to unsafe control.  It also enables control;
        # otherwise the mode would be a no-op that still looks dangerous in logs.
        if self.yolo and not self.control_enabled:
            object.__setattr__(self, "control_enabled", True)
        object.__setattr__(self, "dome_url", url)
        object.__setattr__(self, "deepseek_base_url", _clean_url(self.deepseek_base_url))

    @property
    def yolo(self) -> bool:
        return self.mode == "yolo"

    @property
    def llm_available(self) -> bool:
        return bool(self.llm_enabled and self.deepseek_api_key)

    @property
    def state_parent(self) -> Path:
        parent = self.state_file.parent
        return parent if str(parent) else Path(".")

    @classmethod
    def from_env(
        cls,
        env_file: str | None = ".env",
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> "AgentConfig":
        """Build configuration from an optional ``.env`` and the environment.

        ``overrides`` is intended for tests and CLI flags.  Keys use the same
        names as the environment variables, for example ``{"dry_run": True}``.
        """
        load_dotenv(env_file)
        values: dict[str, Any] = {}
        overrides = dict(overrides or {})

        url = overrides.get("dome_url", _env("DOME_URL", ""))
        key_file = overrides.get("dome_key_file", _env("DOME_KEY_FILE", ""))
        key = overrides.get("dome_key", _env("DOME_KEY", ""))
        key_file_text = str(key_file).strip() if key_file else None
        key_text = str(key or "").strip()
        if not key_text and key_file_text:
            key_text = _load_key(key_file_text)

        log_dir = Path(
            overrides.get(
                "log_dir",
                _path_env("AGENT_LOG_DIR", Path("logs")),
            )
        ).expanduser()
        state_file = Path(
            overrides.get(
                "state_file",
                _path_env("AGENT_STATE_FILE", log_dir / "agent_state.json"),
            )
        ).expanduser()

        values.update(
            {
                "dome_url": str(url),
                "dome_key": key_text,
                "dome_key_file": key_file_text,
                "agent_name": overrides.get(
                    "agent_name",
                    (_env("AGENT_NAME", DEFAULT_NAME) or DEFAULT_NAME).strip(),
                ),
                "deepseek_api_key": overrides.get(
                    "deepseek_api_key",
                    (_env("DEEPSEEK_API_KEY", "") or "").strip() or None,
                ),
                "deepseek_base_url": str(
                    overrides.get(
                        "deepseek_base_url",
                        _env("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_URL),
                    )
                ),
                "deepseek_model": str(
                    overrides.get(
                        "deepseek_model",
                        _env("DEEPSEEK_MODEL", DEFAULT_MODEL),
                    )
                ).strip(),
                "deepseek_timeout": float(
                    overrides.get(
                        "deepseek_timeout",
                        _float_env("DEEPSEEK_TIMEOUT", 30.0, minimum=1.0, maximum=120.0),
                    )
                ),
                "llm_enabled": bool(
                    overrides.get("llm_enabled", _bool_env("AGENT_LLM", True))
                ),
                "llm_min_interval_s": float(
                    overrides.get(
                        "llm_min_interval_s",
                        _float_env("AGENT_LLM_MIN_INTERVAL", 30.0, minimum=0.0),
                    )
                ),
                "llm_max_calls_per_hour": int(
                    overrides.get(
                        "llm_max_calls_per_hour",
                        _int_env("AGENT_LLM_MAX_CALLS_PER_HOUR", 60, 1),
                    )
                ),
                "log_dir": log_dir,
                "state_file": state_file,
                "dry_run": bool(
                    overrides.get("dry_run", _bool_env("AGENT_DRY_RUN", False))
                ),
                "control_enabled": bool(
                    overrides.get(
                        "control_enabled",
                        _bool_env("AGENT_ALLOW_CONTROL", False),
                    )
                ),
                "mode": str(
                    overrides.get(
                        "mode",
                        _env(
                            "AGENT_MODE",
                            "yolo" if _bool_env("AGENT_YOLO", False) else "normal",
                        )
                        or ("yolo" if _bool_env("AGENT_YOLO", False) else "normal"),
                    )
                )
                .strip()
                .casefold(),
                "policy_file": Path(
                    overrides.get(
                        "policy_file",
                        _path_env("AGENT_POLICY_FILE", Path("dome_policy.json")),
                    )
                ).expanduser(),
                "telemetry_timeout_s": float(
                    overrides.get(
                        "telemetry_timeout_s",
                        _float_env("AGENT_TELEMETRY_TIMEOUT", 10.0, minimum=2.0, maximum=60.0),
                    )
                ),
                "observation_interval_s": float(
                    overrides.get(
                        "observation_interval_s",
                        _float_env(
                            "AGENT_OBSERVATION_INTERVAL", 45.0, minimum=5.0, maximum=600.0
                        ),
                    )
                ),
                "reconnect_delay_s": float(
                    overrides.get(
                        "reconnect_delay_s",
                        _float_env("AGENT_RECONNECT_DELAY", 5.0, minimum=1.0, maximum=120.0),
                    )
                ),
                "guest_recheck_s": float(
                    overrides.get(
                        "guest_recheck_s",
                        _float_env("AGENT_GUEST_RECHECK", 60.0, minimum=10.0, maximum=600.0),
                    )
                ),
                "min_send_gap_s": float(
                    overrides.get(
                        "min_send_gap_s",
                        _float_env("AGENT_MIN_SEND_GAP", 0.35, minimum=0.2, maximum=1.0),
                    )
                ),
                "run_seconds": float(
                    overrides.get(
                        "run_seconds",
                        _float_env("AGENT_RUN_SECONDS", 0.0, minimum=0.0),
                    )
                ),
                "version": str(overrides.get("version", AGENT_VERSION)),
            }
        )
        return cls(**values)

    def safe_dict(self) -> dict[str, Any]:
        """Configuration values allowed in the agent log (no secrets)."""
        return {
            "agent": self.agent_name,
            "version": self.version,
            "dome_url": self.dome_url,
            "deepseek_base_url": self.deepseek_base_url,
            "deepseek_model": self.deepseek_model,
            "llm_enabled": self.llm_available,
            "dry_run": self.dry_run,
            "control_enabled": self.control_enabled,
            "mode": self.mode,
            "yolo": self.yolo,
            "policy_file": str(self.policy_file),
            "log_dir": str(self.log_dir),
            "state_file": str(self.state_file),
            "telemetry_timeout_s": self.telemetry_timeout_s,
            "observation_interval_s": self.observation_interval_s,
            "reconnect_delay_s": self.reconnect_delay_s,
            "guest_recheck_s": self.guest_recheck_s,
            "min_send_gap_s": self.min_send_gap_s,
            "run_seconds": self.run_seconds,
        }

    def secrets(self) -> tuple[str, ...]:
        values = [self.dome_key, self.deepseek_api_key or ""]
        return tuple(value for value in values if value)

    def redact(self, text: str) -> str:
        result = text
        for secret in self.secrets():
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return result


_SECRET_RE = re.compile(r"(?i)^(?:.*_)?(?:key|token|secret|password)$")


def redact_mapping(value: Any, *, extra_secrets: tuple[str, ...] = ()) -> Any:
    """Recursively remove secret-looking fields and literal secret values."""
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _SECRET_RE.match(name):
                cleaned[name] = "[REDACTED]"
            else:
                cleaned[name] = redact_mapping(item, extra_secrets=extra_secrets)
        return cleaned
    if isinstance(value, list):
        return [redact_mapping(item, extra_secrets=extra_secrets) for item in value]
    if isinstance(value, tuple):
        return [redact_mapping(item, extra_secrets=extra_secrets) for item in value]
    if isinstance(value, str):
        for secret in extra_secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    return value
