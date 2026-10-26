"""Async client and CLI for the Dome WebSocket API (Team 3)."""

from .client import (
    MISSING,
    DomeAuthError,
    DomeClient,
    DomeConfigError,
    DomeConnectionError,
    DomeError,
    DomeSessionRevoked,
    DomeTimeout,
)
from .agent_api import DomeAgentAPI, tool_specs
from .domain import DomeValidationError

__all__ = [
    "MISSING",
    "DomeAuthError",
    "DomeClient",
    "DomeConfigError",
    "DomeConnectionError",
    "DomeError",
    "DomeSessionRevoked",
    "DomeTimeout",
    "DomeAgentAPI",
    "DomeValidationError",
    "tool_specs",
]
