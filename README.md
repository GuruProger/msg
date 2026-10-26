# Dome autonomous shift agent

One-command autonomous agent for the Dome WebSocket API. Deterministic rules own
all safety and production decisions; DeepSeek is advisory only and never sends
commands.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: DOME_URL and either DOME_KEY or DOME_KEY_FILE
python main.py
```

`python -m dome_agent` is equivalent. `dome-agent` is installed by
`pip install .`.

## Configuration

Required:

- `DOME_URL` — training or production WebSocket endpoint.
- `DOME_KEY` or `DOME_KEY_FILE` — team credential. Prefer a file outside the
  repository and keep `.env` at mode `0600`.

Optional:

- `DEEPSEEK_API_KEY` — enables advisory DeepSeek calls. Without it the agent
  runs rules-only.
- `AGENT_DRY_RUN=1` — analyze and log without sending commands or chat messages.
- `AGENT_RUN_SECONDS=N` — stop cleanly after N seconds; `0` runs indefinitely.
- `AGENT_LOG_DIR`, `AGENT_STATE_FILE` — log and restart-state locations.
- `AGENT_MIN_SEND_GAP=0.35` — enforces at most five commands/requests per second.

## Runtime behavior

- Connects, authenticates, and waits as `guest` until the team shift is active.
- Reconnects after network errors, `session_revoked`, or a closed socket.
- Reads telemetry and operator chat, applies deterministic rules, then executes
  validated actions through `DomeAgentAPI.control_device`.
- Verifies state-changing commands against the next telemetry frame.
- Persists the current task, chat cursor, cooldowns, and CO2 source override so a
  restart does not duplicate or forget work.
- Ignores operator chat addressed to another team or agent.
- Blocks CNC work when fume extraction is off, alarming, or below 50% airflow.
- Treats CO2 as a false alarm only when an isolated high sensor is contradicted
  by a reliable cluster; a real event keeps production paused.
- Refuses unknown machine hold reasons, bed collisions, and filament runout
  without operator action.

## Logs

Each run creates `logs/agent_<run_id>.jsonl` with flushed JSONL records:
`start`, `connection`, `observation`, `decision`, `action`, `verification`,
`message`, `llm_call`, `error`, and `stop`. Secrets are redacted before writing.
A normal SIGINT/SIGTERM shutdown always writes the final `stop` record.

## Tests

```bash
python -m unittest discover -s tests
python tests/acceptance_dome_cli.py
```

The suite includes a real subprocess run of the agent against
`tests/mock_dome_api.py`, including an operator print task and telemetry
verification.

## Layout

- `main.py` / `dome_agent/agent.py` — one-command entry point and loop.
- `dome_agent/rules.py` — deterministic safety and production rules.
- `dome_agent/analysis.py` — telemetry normalization, CO2 consensus, alerts.
- `dome_agent/chat.py` — operator parsing and completion guards.
- `dome_agent/llm.py` — bounded DeepSeek advisory client.
- `dome_agent/state.py` — restart-safe JSON state.
- `dome_agent/logging.py` — JSONL audit log.
- `dome_control/` — WebSocket client, validation, CLI, and agent API facade.
