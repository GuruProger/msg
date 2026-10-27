# Справочник backend

## 1. Переменные окружения

Конфигурация читается из `.env` (если файл существует) и окружения. В `.env`
используется `os.environ.setdefault`: уже заданная переменная окружения имеет
приоритет над строкой из файла. Подстановок `${...}` нет; значения обрезаются
от кавычек только на этапе чтения простого `.env`.

| Переменная | По умолчанию | Допустимо | Назначение |
|---|---|---|---|
| `DOME_URL` | — (обязательна) | `ws://host:port`, `wss://host` | WebSocket endpoint |
| `DOME_KEY` | — | непустая строка | Ключ команды; приоритетнее key-файла |
| `DOME_KEY_FILE` | — | путь | Файл со строкой `team: <key>` |
| `DEEPSEEK_API_KEY` | пусто | строка | Включает LLM; без него LLM недоступна |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | URL | База API; `/chat/completions` добавляется автоматически |
| `DEEPSEEK_MODEL` | `deepseek-chat` | строка | Модель |
| `DEEPSEEK_TIMEOUT` | `30` | 1..120 | Таймаут HTTP-запроса LLM, s |
| `AGENT_NAME` | `team3-agent` | непустая строка | Имя агента в чате/логе |
| `AGENT_LOG_DIR` | `logs` | путь | Каталог JSONL |
| `AGENT_STATE_FILE` | `<log_dir>/agent_state.json` | путь | State-файл |
| `AGENT_ALLOW_CONTROL` | `0` | bool | Мастер-разрешение команд для normal/auto |
| `AGENT_MODE` | `normal` | `normal`/`auto`/`yolo` | Режим |
| `AGENT_YOLO` | `0` | bool | Алиас yolo, если `AGENT_MODE` не задан |
| `AGENT_POLICY_FILE` | `dome_policy.json` | путь | Policy allowlist |
| `AGENT_DRY_RUN` | `0` | bool | Не отправлять команды/сообщения |
| `AGENT_RUN_SECONDS` | `0` | ≥0 | Остановиться через N s; 0 — бесконечно |
| `AGENT_OBSERVATION_INTERVAL` | `45` | 5..600 | Период observation-записей, s |
| `AGENT_RECONNECT_DELAY` | `5` | 1..120 | Пауза между переподключениями, s |
| `AGENT_GUEST_RECHECK` | `60` | 10..600 | Проверка смены в guest, s |
| `AGENT_TELEMETRY_TIMEOUT` | `10` | 2..60 | Таймаут телеметрии/API, s |
| `AGENT_MIN_SEND_GAP` | `0.35` | 0.2..1.0 | Минимальный интервал запросов, s (≤5/s) |
| `AGENT_LLM` | `1` | bool | Разрешить LLM |
| `AGENT_LLM_MIN_INTERVAL` | `30` | ≥0 | Минимальный интервал LLM-вызовов, s |
| `AGENT_LLM_MAX_CALLS_PER_HOUR` | `60` | ≥1 | Часовой бюджет LLM |
| `AGENT_KB` / `AGENT_KB_ENABLED` | `1` | bool | Включить read-only базу знаний |
| `AGENT_KB_DIR` | `dome_rag` | путь | Каталог `dome_rag` (`data/`, `build/`) |
| `AGENT_KB_DB` | `<kb_dir>/build/kb.sqlite` | путь | SQLite-база знаний |
| `AGENT_KB_MAX_CARDS` | `6` | ≥1 | Максимум карточек в пакете |
| `AGENT_KB_MAX_CHARS` | `6000` | ≥800 | Лимит размера пакета, символов |

Bool-значения: `1/true/yes/on/y` и `0/false/no/off/n` (case-insensitive).
Пустая строка для bool означает default.

Валидация при старте:

- `DOME_URL` обязателен и должен быть `ws://`/`wss://`;
- ключ должен быть непустым после чтения key-файла;
- `AGENT_NAME` непустой;
- `DEEPSEEK_TIMEOUT > 0`;
- `AGENT_TELEMETRY_TIMEOUT > 0`;
- `AGENT_OBSERVATION_INTERVAL >= 5`;
- `AGENT_MIN_SEND_GAP >= 0.2`;
- `AGENT_LLM_MIN_INTERVAL >= 0`;
- `AGENT_LLM_MAX_CALLS_PER_HOUR >= 1`;
- `AGENT_RUN_SECONDS >= 0`;
- `AGENT_KB_MAX_CARDS >= 1`;
- `AGENT_KB_MAX_CHARS >= 800`;
- `AGENT_MODE` из трёх значений;
- `yolo` включает `control_enabled`, даже если `AGENT_ALLOW_CONTROL=0`.

`AgentConfig.safe_dict()` пишет в лог только безопасные поля: agent, version,
dome_url, deepseek_base_url, deepseek_model, `llm_enabled` (фактическая
доступность LLM: флаг и наличие ключа), dry_run, control_enabled, mode, yolo,
policy_file, log_dir, state_file, таймауты, reconnect/guest, min_send_gap,
run_seconds. Ключи не входят.

## 2. CLI агента

```bash
python main.py [options]
python -m dome_agent [options]
dome-agent [options]
```

| Флаг | Действие |
|---|---|
| `--env-file PATH` | `.env`; по умолчанию `.env`, отсутствие допустимо |
| `--url URL` | Переопределить `DOME_URL` |
| `--name NAME` | Переопределить `AGENT_NAME` |
| `--log-dir PATH` | Переопределить `AGENT_LOG_DIR` |
| `--state-file PATH` | Переопределить `AGENT_STATE_FILE` |
| `--dry-run` | Включить dry-run |
| `--allow-control` | Включить мастер-разрешение |
| `--mode {normal,auto,yolo}` | Режим |
| `--yolo` | Синоним `--mode yolo` |
| `--run-seconds N` | Остановиться через N s |

CLI-переопределения применяются после `.env`/окружения. Для bool-флагов нет
парного `--no-...`; снять ранее включённый флаг можно только через env. Если
переданы и `--mode`, и `--yolo`, побеждает `--yolo` (он применяется последним).

Коды возврата `main()`:

- `0` — штатное завершение;
- `2` — ошибка конфигурации (сообщение в stderr);
- `130` — `KeyboardInterrupt`.

## 3. Policy (`dome_policy.json`)

Схема:

```json
{
  "version": 1,
  "default": "deny",
  "deny_nodes": ["..."],
  "deny_actions": ["..."],
  "allow": {
    "node_id": {
      "actions": ["action"],
      "values": {"action": ["value1", "value2"]}
    },
    "another_node": ["action"]
  }
}
```

Правила:

- `default` обязан быть `"deny"`;
- `allow` обязан быть объектом;
- `deny_actions` и `deny_nodes` проверяются раньше allow;
- узел, которого нет в `allow`, запрещён;
- действие, которого нет в `actions`, запрещено;
- если для действия есть `values`, значение должно быть в списке;
- `force` в `DomeAgentAPI.resolve_control_action` может разрешить неизвестное
  действие каталога, но не отменяет `operator_controls` и не отменяет policy
  gate агента.

Текущий allowlist:

| Узел | Действия | Ограничения значений |
|---|---|---|
| `dome_automation_01` | `set_data_trust`, `set_primary_sensor`, `reset_interlock` | `reset_interlock`: только `IL_CO2`; `set_primary_sensor`: `climate_sensor_01..03` |
| `smart_panel_01` | `line_on`, `line_off`, `reset_protection` | линии 1..8 |
| `supply_ventilation_01` | `turn_on`, `set_mode` | `set_mode`: `FRESH_AIR` |
| `solar_inverter_01` | `set_fan`, `reset_error` | `set_fan`: `ON`/`OFF` |
| `dizel_1` | `turn_on` | — |
| `fume_extraction_01` | `turn_on` | — |
| `printer_3d_01` | `start_job`, `resume`, `home`, `reconnect` | — |
| `cnc_01` | `start_job`, `resume`, `home`, `reconnect` | — |

`deny_nodes`: `access_control_01`, `dome_sealing_01`, `fire_suppression_01`.

`deny_actions`: `block_automation`, `block_interlock`, `cancel_job`, `force`,
`reboot`, `remove_part`, `set_thresholds`, `shutdown`, `unlock_door`.

Hard denylist агента (`HARD_DENY_ACTIONS`) не зависит от policy:
`block_automation`, `block_interlock`, `set_thresholds`, `unlock_door`,
`remove_part`, `shutdown`, `reboot`.

## 4. JSONL-лог

### 4.1. Общие поля

| Поле | Тип | Описание |
|---|---|---|
| `ts` | ISO 8601 | Локальное время с миллисекундами |
| `run_id` | string | `YYYYMMDD-HHMMSS`, при коллизии `-2`, `-3` |
| `seq` | int | Монотонный номер в run, начиная с 1 |
| `kind` | string | Тип записи |
| `dome_time` | string | Игровое время, если известно |
| `ref` | int | Ссылка на `seq` связанной записи |

Имена полей с `key`/`token`/`secret`/`password` заменяются на `[REDACTED]`;
литеральные значения секретов из конфигурации вычищаются из строк.

### 4.2. Типы записей

| `kind` | Когда | Основные поля |
|---|---|---|
| `start` | Старт | `agent`, `version`, `models`, `config` |
| `stop` | Штатное завершение | `reason`, `stats` |
| `connection` | Auth/guest/disconnect/revoked | `event`, `role`, `team`, `details` |
| `observation` | Периодически/при изменении | `summary`, `metrics`, `anomalies`, `task` |
| `decision` | Решение правил или LLM | `situation`, `cause`, `confidence`, `evidence`, `plan`, `source`, `actions`/`commands`, `messages`, `task`, `alerts` |
| `action` | Отправка команды | `request`, `response`, `reason`, `dry_run`, `unsafe`, `latency_ms`, `ref`, `expect`, `source`, `force` |
| `action_blocked` | Действие запрещено | `request`, `reason`, `unsafe`, `ref`, `expect`, `source`, `forceable` |
| `action_queue` | Дренаж не завершился | `event=drain_timeout`, `pending`, `in_flight` |
| `action_reconcile` | Intent после рестарта | `action`, `request`, `expected`, `ref`, `status=unknown`, `started_at` |
| `verification` | Проверка эффекта | `ref`, `expected`, `observed`, `ok`, `error`, `waited_s`, `no_effect`, `unknown` |
| `message` | Чат | `direction`, `sender`/`role`/`team`, `to`, `text`, `message_id`, `dry_run`, `sent` |
| `llm_call` | Вызов LLM | `model`, `purpose`, `latency_ms`, `ok`, `error`, `summary`, `prompt_tokens`, `completion_tokens`, `context_chars` |
| `knowledge` | Запрос к базе знаний | `purpose` (`control`/`advice`), `question`, `verdict`, `confidence`, `sources`; при сбое — `event=unavailable`, `error` |
| `error` | Исключение/ошибка | `where`, `error`, `traceback` и контекст |

`decision.source`: `rules`, `llm`, `chat_epoch`.
`action.source` (может отсутствовать у действий rule engine): `yolo`,
`confirmed`, `confirmed_force`, `protective`, `before_telemetry`, `dependency`,
`role`.
`action.unsafe`: `true` для yolo/force. Набор полей `action`/`action_blocked`
зависит от пути: например, `latency_ms` пишется для действий rule engine, а
`forceable` — только для блокировок модельных команд.

### 4.3. Метрики observation

`observation.metrics` содержит `co2_ppm`, `co2_source`, `co2_false_alarm`,
`soc_pct`, `printer`, `cnc`, `alerts`, `task`. API журнала нормализует публичные
ключи `co2_ppm`, `soc_pct`, `printer`, `cnc`, `fume`, `power_w`; отсутствующие
значения становятся `null`.

## 5. State-файл

Путь: `AGENT_STATE_FILE` (по умолчанию `AGENT_LOG_DIR/agent_state.json`).

| Ключ | Формат | Назначение |
|---|---|---|
| `version` | int | Версия схемы, сейчас 1 |
| `last_message_id` | int | Последний обработанный id чата |
| `handled_message_ids` | list[int] | Последние 300 id |
| `task` | object/null | Текущее задание |
| `operator_requests` | object | Дедупликация сообщений, до 80 |
| `last_actions` | object | `key -> {at, success}`, до 200 |
| `failed_until` | object | `key -> epoch`, до 200 |
| `pending_actions` | object | `key -> {request, ref, expected, at}`, до 50 |
| `co2_override` | object/null | `{node_id, at, reason}` |
| `notes` | object | Зарезервировано |

`task` (из `OperatorTask.as_dict()`):

```json
{
  "message_id": 12,
  "text": "Напечатать bracket.gcode",
  "kind": "print",
  "created_at": 1710000000.0,
  "sender": "operator",
  "items": [
    {
      "kind": "print",
      "text": "Напечатать bracket.gcode",
      "file": "bracket.gcode",
      "line": null,
      "actionable": true,
      "status": "starting",
      "note": "start_job sent"
    }
  ]
}
```

Статусы пунктов: `pending`, `starting`, `started`, `homing`, `needs_info`,
`done`; могут добавляться `note`, `completed_at`, `completion`.

Свойства файла:

- режим `0600`; родительская директория создаётся;
- атомарная запись: temp + `fsync` + `os.replace` + `fsync` директории;
- при повреждении — копия `.corrupt-<ns>` и сброс к default;
- `pending_actions` не экспортируется API журнала;
- `updated_at` не хранится, API журнала подставляет mtime;
- параллельная работа двух агентов с одним state не поддерживается.

## 6. HTTP API журнала

Запуск: `python -m dome_web [--backend-root PATH] [--port PORT]`
(или `dome-web` после `pip install .`).

API — read-only. Все `/api/*` ответы получают `Cache-Control: no-store`.
Ошибки чтения лога/каталога возвращаются как HTTP 500; неверный `run_id` — 400.

### `GET /api/health`

Без параметров. Пример:

```json
{
  "status": "ok",
  "has_log": true,
  "has_state": true,
  "log_file": "agent_20261027-120000.jsonl",
  "state_file": "logs/agent_state.json",
  "last_record_age_s": 1.2,
  "records_loaded": 42
}
```

`status`:

- `waiting` — лога/записей нет;
- `ok` — есть запись не старше 120 s и это не `stop`;
- `stopped` — последняя запись `stop`;
- `stale` — последняя запись старше 120 s;
- `error` — ошибка чтения лога/каталога.

### `GET /api/summary`

Сводка по самому новому логу. Верхний уровень:

```text
generated_at, demo, health, run, state, observation, decision,
alerts, actions, verifications, messages, timeline, counts, metrics
```

- `run`: run_id, file, started_at, last_ts, stopped, stop_reason, agent, config,
  role, team;
- `state`: безопасное подмножество state;
- `observation`: последняя observation либо null;
- `decision`: последняя decision либо null;
- `alerts`: ошибки, неуспешные/заблокированные действия, аномалии наблюдений;
- `actions`: нормализованные action-записи с привязанной verification;
- `verifications`, `messages`;
- `timeline`: последние 60 записей;
- `counts`: records, decisions, actions, verifications, messages, errors;
- `metrics`: `co2_ppm`, `soc_pct`, `printer`, `cnc`, `fume`, `power_w`.

Секретные поля удаляются, литеральные секреты вычищаются из строк.

### `GET /api/logs`

Query-параметры:

| Параметр | Тип | По умолчанию | Ограничение |
|---|---|---|---|
| `kind` | string | нет | фильтр по `record.kind` |
| `limit` | int | 200 | 1..1000 |
| `run_id` | string | последний лог | `[A-Za-z0-9_.-]+` |

Поведение:

- `run_id` проверяется на traversal (`.`/`..` запрещены);
- ищется `agent_<run_id>.jsonl`, затем fallback по mtime;
- если лог не найден, возвращается пустой `records` (не 404);
- записи возвращаются в хронологическом порядке;
- ответ: `run_id`, `file`, `records`, `truncated`, `count`.

### `GET /api/state`

Возвращает только:

```text
task, last_message_id, co2_override, last_actions, failed_until, notes, updated_at
```

При отсутствии/повреждении state возвращает пустую форму, а не ошибку.

### `GET /api/demo`

Полная демонстрационная сводка без чтения файлов и сети. Полезно для проверки
UI/API без запущенного агента.

### `/` и `/static/*`

Отдают фронтенд, если он установлен. В backend-документации не описываются.
Если статики нет, `/` возвращает 503.

## 7. `dome_control.agent_api`

`DomeAgentAPI` — async-фасад для внешнего кода:

- `from_env()`, `connect()`, `close()`, `reconnect()`;
- `catalog()`, `list_devices()`, `get_device()`;
- `get_telemetry()`, `get_status()`;
- `control_device()`, `query_device()`;
- `read_messages()`, `post_message()`;
- `whoami()`, `ping()`;
- `call_tool(name, arguments)`.

`control_device`:

- ищет узел по `node_id` или `code_name` в живом `describe_nodes`;
- разрешает canonical action/alias; `operator_controls` запрещены даже при
  `force`;
- `force=True` разрешает неизвестное каталогу действие, но не отменяет
  operator-only;
- требует свежий before-кадр, иначе `blocked=true` без отправки;
- после `success` и `verify=true` получает after и `telemetry_diff`;
- `allow_without_before=True` разрешает отправку без before (используется для
  auto-protective), результат может иметь `unknown=true`.

`TOOL_SCHEMAS`/`tool_specs()` — 10 внешних JSON-схем:
`dome_get_telemetry`, `dome_get_status`, `dome_list_devices`,
`dome_get_device`, `dome_control_device`, `dome_query_device`,
`dome_read_messages`, `dome_post_message`, `dome_whoami`, `dome_ping`.
Это не runtime-набор LLM: у агента только `control`, `query`, `telemetry`,
`catalog`, `knowledge`, `send_message`.

## 8. Ошибки и исключения

| Исключение | Где | Смысл |
|---|---|---|
| `DomeError` | `dome_control.client` | Базовое |
| `DomeConfigError` | client | Неверный URL/конфигурация |
| `DomeAuthError` | client | Сервер отклонил ключ |
| `DomeTimeout` | client | Нет ответа/телеметрии |
| `DomeConnectionError` | client | Сокет закрыт/отправка не удалась |
| `DomeSessionRevoked` | client | Сервер отозвал сессию |
| `DomeValidationError` | domain | Узел/действие/запрос не разрешены каталогом |
| `GateError` | gate | Policy отсутствует/некорректна |

`DeepSeekAdvisor.advise/decide` не выбрасывают наружу HTTP/JSON-ошибки: они
логируют `llm_call` с `ok=false` и возвращают `None`.

## 9. Тесты

Основная команда:

```bash
python -m unittest discover -s tests
```

Снимок: **259 tests, OK** (2026-10-27, Python 3.14.8).

| Файл | Что покрывает |
|---|---|
| `tests/test_dome_agent.py` | config, chat, rules, CO2, safety, state, logging, LLM parsing, agent loop, verification, gate, control, cooldowns |
| `tests/test_agent_api.py` | facade, catalog validation, before/after, tool schemas |
| `tests/test_agent_api_integration.py` | интеграция facade с mock-сервером |
| `tests/test_agent_integration.py` | сквозной сценарий: агент читает задание Оператора и запускает печать через mock Dome |
| `tests/test_client_telemetry.py` | telemetry freshness, meeting clear, retry policy |
| `tests/test_domain.py` | catalog/alias/validation/diff |
| `tests/test_dashboard.py` | health/summary/logs/state/demo, sanitization, corrupt lines, cache |
| `tests/test_dome_rag.py` | сборка/поиск/ответы RAG, покрытие и карточки |
| `tests/test_integrated_knowledge.py` | связка `KnowledgeService` с агентом и `dome_rag` |
| `tests/test_dome_web.py`, `tests/test_dome_web_live.py` | backend `dome_web` и live-сценарии |
| `tests/test_frontend_backend.py` | контракт фронтенд↔backend |
| `tests/mock_dome_api.py` | локальный WebSocket mock (5 узлов в тестовом каталоге) |
| `tests/acceptance_dome_cli.py` | acceptance-сценарий CLI |
| `tests/fixtures/telemetry.jsonl` | базовая телеметрия |

Дополнительно:

```bash
python -m compileall -q dome_agent dome_control dome_web tests
python tests/acceptance_dome_cli.py
```

`ruff` в текущем окружении отсутствует. Если добавлен — запускать отдельно;
в CI/локально это дополнительный, а не заменяющий тесты шаг.

## 10. Карта исходников

| Файл | Ключевые сущности |
|---|---|
| `dome_agent/agent.py` | `DomeAgent`, `run_agent`, `main`, `build_parser` |
| `dome_agent/analysis.py` | `Snapshot`, `Co2Assessment`, `analyze_co2`, `normalize_interlocks`, `merge_alerts` |
| `dome_agent/rules.py` | `ActionSpec`, `RuleResult`, `OutgoingMessage`, `RuleEngine` |
| `dome_agent/chat.py` | `ChatMessage`, `OperatorTask`, `parse_operator_task`, completion guards |
| `dome_agent/control.py` | `HARD_DENY_ACTIONS`, `ModelCommand`, `ModelDecision`, `PendingProposal`, `DomeTool`, `is_auto_protective` |
| `dome_agent/gate.py` | `CommandGate`, `GateDecision`, `GateError` |
| `dome_agent/verification.py` | `ExpectedEffect`, `expected_effect`, `meaningful_diff` |
| `dome_agent/state.py` | `StateStore` |
| `dome_agent/logging.py` | `AgentLog`, `VALID_KINDS` |
| `dome_agent/config.py` | `AgentConfig`, `redact_mapping`, `AGENT_VERSION` |
| `dome_agent/llm.py` | `DeepSeekAdvisor`, `Advice`, `parse_advice` |
| `dome_agent/knowledge.py` | `KnowledgeService`, `_pack_refs`, read-only мост к `dome_rag` |
| `dome_control/client.py` | `DomeClient`, `retry_policy`, transport errors |
| `dome_control/agent_api.py` | `DomeAgentAPI`, `TOOL_SCHEMAS`, `tool_specs` |
| `dome_control/domain.py` | catalog/filter/alias/validation/diff |
| `dome_control/cli.py` | `dome-panel` CLI |
| `dome_control/panel.py` | форматирование, `.env`, key-файл, demo |
| `dome_web/journal_api.py` | `create_app`, API endpoints |
| `dome_policy.json` | allowlist/denylist |
| `.env.example` | шаблон конфигурации |
