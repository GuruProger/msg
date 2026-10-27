# Dome autonomous shift agent

One-command autonomous agent for the Dome WebSocket API. The LLM proposes or
executes actions depending on the selected mode; a fail-closed policy gate and
a hard denylist still protect the dome. There is no shell access: the model can
only call the limited Dome tool set.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: DOME_URL and either DOME_KEY or DOME_KEY_FILE
python main.py
```

The agent is **read-only by default**. To run the normal control mode, set
`AGENT_ALLOW_CONTROL=1` (or pass `--allow-control`) and `AGENT_MODE=normal`.
`python -m dome_agent` is equivalent. `dome-agent` is installed by
`pip install .`.

## Read-only dashboard

The dashboard is a local viewer over the agent JSONL log and persistent state.
It never sends Dome commands, writes chat messages, or mutates agent files.

```bash
python -m dome_dashboard
# or, after pip install .:
dome-dashboard --host 127.0.0.1 --port 8080
```

It reads `AGENT_LOG_DIR` and `AGENT_STATE_FILE` when set, otherwise `logs/` and
`logs/agent_state.json`. Override the bind address with `DASHBOARD_HOST` and
`DASHBOARD_PORT`. Secrets and secret-looking fields are removed before data is
returned; use `/api/demo` for an offline sample.

## Configuration

Required:

- `DOME_URL` — training or production WebSocket endpoint.
- `DOME_KEY` or `DOME_KEY_FILE` — team credential. Prefer a file outside the
  repository and keep `.env` at mode `0600`.

Optional:

- `DEEPSEEK_API_KEY` — enables the LLM controller. Without it the agent cannot
  execute LLM commands and reports that the LLM is unavailable.
- `AGENT_MODE=normal|auto|yolo` — operating mode (default `normal`).
  - `normal` — the LLM analyzes telemetry/chat and proposes actions; nothing
    is executed until the operator confirms in Dome chat. The approved
    protective exceptions (`line_off`, `stop_job`, `turn_off`, `set_fan=ON`,
    ventilation/fume `turn_on`, `set_mode=FRESH_AIR`) run automatically.
    Blocked proposals require an explicit force confirmation.
  - `auto` — deterministic rules only; legacy/fallback mode for tests or when
    the LLM is intentionally disabled. Still uses `dome_policy.json` and
    `AGENT_ALLOW_CONTROL`.
  - `yolo` — **UNSAFE**. The LLM executes its commands immediately; policy,
    cooldowns and telemetry verification are bypassed. Hard-denied actions
    remain blocked. Use only on a test dome.
- `AGENT_ALLOW_CONTROL=1` — master control switch for `normal` and `auto`.
  Without it those modes only observe and log. `yolo` implies it.
- `AGENT_YOLO=1` — backward-compatible alias for `AGENT_MODE=yolo`.
- `AGENT_POLICY_FILE=dome_policy.json` — fail-closed allowlist of node/action
  pairs and values for `normal`/`auto`. Missing or malformed policy prevents
  control mode startup.
- `AGENT_DRY_RUN=1` — analyze and log without sending commands or chat messages.
- `AGENT_RUN_SECONDS=N` — stop cleanly after N seconds; `0` runs indefinitely.
- `AGENT_LOG_DIR`, `AGENT_STATE_FILE` — log and restart-state locations.
- `AGENT_MIN_SEND_GAP=0.35` — enforces at most five commands/requests per second.
- `DASHBOARD_HOST`, `DASHBOARD_PORT` — dashboard bind address and port
  (defaults: `127.0.0.1:8080`).

## Runtime behavior

- Connects, authenticates, and waits as `guest` until the team shift is active.
- Reconnects after network errors, `session_revoked`, or a closed socket.
- Reads telemetry and operator chat. In `normal`, the LLM returns a structured
  proposal; the agent posts it to Dome chat and waits for confirmation.
- Confirmation phrases: `да`, `подтверждаю`, `выполняй`, `ок`, `давай`;
  cancellation: `отмена`, `не надо`, `стоп`; force: `да, принудительно` or
  `force`. A pending proposal expires after 5 minutes.
- In `normal`/`auto`, every control action passes through `CommandGate` and
  `dome_policy.json`; dangerous actions such as `block_interlock`,
  `set_thresholds`, and `unlock_door` are denied.
- In `yolo`, the LLM commands are executed through `DomeTool` immediately.
  The hard denylist (`block_interlock`, `block_automation`, `set_thresholds`,
  `unlock_door`, `remove_part`, `shutdown`, `reboot`) cannot be overridden.
- `normal`/`auto` use the API verification path; `yolo` sends with
  `verify=false` and logs `unsafe: true`.
- The only model tools are `control`, `query`, `telemetry`, `catalog`, and
  `send_message`. There is no shell/subprocess tool.
- Persists the current task, chat cursor, cooldowns, CO2 source override, and
  pre-send action intents so a restart reconciles unknown command outcomes.
- Ignores operator chat addressed to another team or agent.
- Blocks CNC work when fume extraction is off, alarming, or below 50% airflow.
- Treats CO2 as a false alarm only when an isolated high sensor is contradicted
  by a reliable cluster; a real event keeps production paused.
- Refuses unknown machine hold reasons, bed collisions, and filament runout
  without operator action.

## Logs

Each run creates `logs/agent_<run_id>.jsonl` with flushed JSONL records:
`start`, `connection`, `observation`, `decision`, `action`, `action_blocked`,
`action_reconcile`, `verification`, `message`, `llm_call`, `error`,
`action_queue`, and `stop`.
`decision.source` shows `rules` or `llm`; `action.source` shows `yolo`,
`confirmed`, `confirmed_force`, or `auto`. Secrets are redacted before writing.
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
- `dome_policy.json` — fail-closed allowlist for commands.
- `dome_agent/gate.py` — policy gate and unsafe YOLO bypass.
- `dome_agent/control.py` — structured LLM decisions, restricted `DomeTool`,
  hard denylist, and confirmation proposals.
- `dome_agent/rules.py` — deterministic safety and production rules.
- `dome_agent/analysis.py` — telemetry normalization, CO2 consensus, alerts.
- `dome_agent/chat.py` — operator parsing and completion guards.
- `dome_agent/llm.py` — bounded DeepSeek advisory client.
- `dome_agent/state.py` — restart-safe JSON state.
- `dome_agent/logging.py` — JSONL audit log.
- `dome_control/` — WebSocket client, validation, CLI, and agent API facade.
- `dome_dashboard/` — read-only FastAPI API and mission-control frontend.
