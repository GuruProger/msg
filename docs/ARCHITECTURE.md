# Архитектура backend

## 1. Назначение

`dome_agent` — один долгоживущий процесс, который управляет сменой в куполе:

- подключается к Dome WebSocket API и аутентифицируется;
- читает телеметрию и общий чат `meeting_01`;
- принимает задания Оператора;
- применяет детерминированные правила безопасности и производства;
- в зависимости от режима либо выполняет действия, либо предлагает их
  Оператору, либо отдаёт решение LLM;
- проверяет физический эффект команд по телеметрии;
- ведёт полный JSONL-аудит и переживает перезапуск.

LLM не является источником истины для безопасности. В `normal` она предлагает
действия, в `yolo` — выполняет их немедленно, но и в том, и в другом случае
жёсткий denylist остаётся в силе. В `auto` команды определяет только rule engine.

## 2. Компоненты

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        python main.py / dome_agent                  │
│                                                                     │
│  AgentConfig ──► DomeAgent ──► RuleEngine ──► ActionSpec            │
│                      │              │                               │
│                      │              ├── Snapshot / analyze_co2      │
│                      │              └── StateStore (cooldowns)       │
│                      │                                              │
│                      ├── CommandGate (dome_policy.json)             │
│                      ├── DomeTool (LLM tools)                       │
│                      ├── DeepSeekAdvisor                          │
│                      └── AgentLog (JSONL)                          │
│                                                                     │
│  DomeAgentAPI ──► DomeClient ──► WebSocket Dome API                │
└─────────────────────────────────────────────────────────────────────┘

dome_web (отдельный процесс)
    FastAPI ──► JSONL-лог + agent_state.json
    только чтение, никаких команд в Dome
```

| Слой | Модуль | Ответственность |
|---|---|---|
| Точка входа | `main.py`, `dome_agent/__main__.py` | Запуск `dome_agent.agent.main` |
| Конфигурация | `dome_agent/config.py` | `.env`, env, валидация, redaction |
| Главный цикл | `dome_agent/agent.py` | Подключение, цикл смены, чат, очередь действий, подтверждения, shutdown |
| Правила | `dome_agent/rules.py` | Safety/production-решения, `ActionSpec`, сообщения |
| Анализ | `dome_agent/analysis.py` | Нормализация телеметрии, CO2 consensus, alerts, `Snapshot` |
| Чат | `dome_agent/chat.py` | Разбор заданий, фильтрация сообщений, completion guards |
| LLM | `dome_agent/llm.py` | DeepSeek `advise`/`decide`, бюджет вызовов, JSON-парсинг |
| Модельные команды | `dome_agent/control.py` | `ModelCommand`, `ModelDecision`, `PendingProposal`, `DomeTool`, hard denylist |
| Policy gate | `dome_agent/gate.py` | Fail-closed allowlist/denylist |
| Проверка эффекта | `dome_agent/verification.py` | `ExpectedEffect`, проверка телеметрии после команды |
| Состояние | `dome_agent/state.py` | Атомарный JSON-state, cooldowns, intent, chat cursor |
| Аудит | `dome_agent/logging.py` | JSONL, `seq`, flush, redaction |
| Dome API | `dome_control/agent_api.py` | Валидация каталога, before/after, tool schemas |
| Транспорт | `dome_control/client.py` | WebSocket, auth, request correlation, reconnect, rate limit |
| Домен | `dome_control/domain.py` | Alias-ы действий, поиск узлов, telemetry diff |
| Админ-CLI | `dome_control/cli.py`, `dome_control/panel.py` | Ручная диагностика и управление, не агент |
| Веб-панель | `dome_web/` | Read-only HTTP API журнала + локальный фронтенд |

## 3. Точки входа и зависимости

Установленные console scripts (`pyproject.toml`):

- `dome-agent` → `dome_agent.agent:main`
- `dome-panel` → `dome_control.cli:main`
- `dome-web` → `dome_web.dashboard_backend:main`

Эквиваленты без установки пакета:

```bash
python main.py
python -m dome_agent
python -m dome_control
python -m dome_web
```

Зависимости: `fastapi`, `httpx`, `uvicorn`, `websockets`; Python `>=3.14`.
`dome_agent` импортирует `dome_control`, лениво — `dome_rag` (только чтение
свидетельств через `KnowledgeService`), но не `dome_web`.
`dome_web` обслуживает локальную панель, импортирует агентские модули и
читает JSONL/state.

## 4. Жизненный цикл агента

```text
main()
  └─ AgentConfig.from_env()
       ├─ загрузка .env (setdefault, существующие env имеют приоритет)
       ├─ чтение DOME_KEY / DOME_KEY_FILE
       ├─ валидация URL, режима, таймаутов, min send gap
       └─ safe_dict() без секретов
  └─ asyncio.run(run_agent(config))
       ├─ установка SIGINT/SIGTERM
       └─ DomeAgent.run()
            ├─ start в JSONL
            ├─ state load_error, если файл был повреждён
            ├─ reconcile незавершённых action intent
            ├─ цикл:
            │    ├─ connect / auth
            │    ├─ role == team|admin ? cycle : wait guest
            │    ├─ telemetry → Snapshot → chat → rules → actions
            │    └─ sleep 1 s (или до run_seconds)
            └─ shutdown:
                 ├─ drain очередей действий (до 7 s)
                 ├─ отмена control/advice задач
                 ├─ закрытие HTTP-клиента LLM и Dome API
                 ├─ stop в JSONL
                 └─ close лога
```

Особенности:

- Роль `team` или `admin` даёт право на команды. Иначе агент работает как
  `guest`: ждёт `AGENT_GUEST_RECHECK` секунд, закрывает соединение и пробует
  снова. В `dry-run` проверка роли при выполнении действия пропускается.
- `AGENT_RUN_SECONDS > 0` ограничивает время работы; по истечении агент
  завершает цикл штатно и пишет `stop` с причиной `run_seconds`.
- При транспортной ошибке, `session_revoked` или закрытии сокета агент
  переподключается. Подключение (в том числе повторное) использует
  retry-политику `DomeClient.connect_with_retry`; после ошибки главный цикл
  дополнительно выдерживает `AGENT_RECONNECT_DELAY`.
- Штатный `SIGINT`/`SIGTERM` вызывает `request_stop`. Транспортные ошибки
  обрабатываются переподключением; неожиданное исключение внутри `_cycle`
  логируется, и агент продолжает смену. Исключение верхнего уровня завершает
  процесс с записью `error`/`stop`; молчаливого продолжения нет.

## 5. Один цикл смены

`DomeAgent._cycle()`:

1. Проверяет отложенную ошибку action worker.
2. Получает свежую телеметрию с `max_age=5 s`, строит `Snapshot` с учётом
   сохранённого CO2 override.
3. Обновляет dome time в логе.
4. Опрашивает чат (не чаще раза в 3 s).
5. Уведомляет Оператора о действиях с неизвестным исходом после рестарта.
6. Вызывает `RuleEngine.decide(snapshot, state.task)`.
7. Если правило изменило задание — сохраняет его.
8. Дальше режим определяет ветку:

| Режим | Что делает цикл |
|---|---|
| `yolo` | Логирует observation, передаёт решение LLM; действия правил **не выполняются**, они только контекст |
| `normal` | Сначала обрабатывает подтверждение/отмену ожидающего предложения; логирует решение; автоматически ставит в очередь только `is_auto_protective` действия правил; отправляет сообщения правил; запускает LLM для предложения |
| `auto` | Логирует решение; ставит в очередь **все** действия правил; отправляет сообщения; при доступной LLM запрашивает только advisory-текст |
| `dry-run` | Как `auto` по постановке действий в очередь (даже в `normal`), но команды и сообщения не отправляются; в `yolo` действия правил не ставятся, LLM-команды выполняются в dry-run |

9. `_maybe_observe` пишет `observation` не чаще `AGENT_OBSERVATION_INTERVAL`
   или при изменении ситуации/набора алертов.
10. Цикл засыпает на 1 s, прерываясь по stop/дедлайну.

Решение логируется не каждый кадр: при наличии действий/сообщений/изменения
задания, при смене ситуации, при изменении ключа алертов или по истечении
observation interval.

## 6. Режимы

| Режим | Кто выбирает действия | Выполнение | Проверка эффекта | LLM |
|---|---|---|---|---|
| `normal` (по умолчанию) | Rule engine + LLM-предложения | Только auto-protective действия правил; команды LLM — после подтверждения Оператора; прочие действия правил не исполняются | Да, для normal/auto | `decide`, proposal |
| `auto` | Rule engine | Все действия правил | Да | Только advisory `advise` |
| `yolo` | LLM | Немедленно | Нет (`verify=false`) | `decide`, исполнение |
| `dry-run` | Зависит от режима | Ничего не отправляется | Нет реального эффекта | Может вызываться; результат dry-run |

### `normal`

- Детерминированные safety-сообщения (пожар, дым, CO2 и т.п.) всё равно
  отправляются Оператору.
- Автоматически выполняются только действия из `is_auto_protective`:
  `smart_panel_01.line_off`, любой `turn_off`, `solar_inverter_01.set_fan=ON`,
  `supply_ventilation_01.turn_on`, `fume_extraction_01.turn_on`,
  `supply_ventilation_01.set_mode=FRESH_AIR`.
- LLM возвращает `ModelDecision`; команды становятся `PendingProposal`,
  публикуются в чат и ждут подтверждения.
- Подтверждение: `да`, `подтверждаю`, `выполняй`, `выполнить`, `ок`,
  `давай`, `yes`, `confirm`.
- Отмена: `отмена`, `отменить`, `не надо`, `нет`, `стоп`, `cancel`, `no`,
  `stop`.
- Принудительное подтверждение: `принудительно`, `force`, `форсируй`; при
  ожидании force-подтверждения обычное «да» также считается force.
- TTL предложения — 300 s. Просроченное предложение не выполняется.
- Пока одно предложение ожидает ответа, отличающееся новое предложение не
  подменяет его; агент просит сначала закрыть текущее. Идентичное предложение
  просто игнорируется.

### `auto`

- Единственный источник команд — `RuleEngine`.
- `dome_policy.json` и `AGENT_ALLOW_CONTROL` обязательны для фактической
  отправки.
- LLM, если настроена, получает контекст и может присылать диагностический
  текст, но не команды.
- Это режим для тестов и для работы без LLM.

### `yolo`

Опасный режим. Включается `AGENT_MODE=yolo` или `--mode yolo` (алиас
`--yolo`). `AGENT_ALLOW_CONTROL` подразумевается включённым.

- LLM — основной управляющий модуль; действия правил не исполняются.
- Policy gate, cooldowns правил и telemetry verification обходятся.
- Hard denylist остаётся: `block_automation`, `block_interlock`,
  `set_thresholds`, `unlock_door`, `remove_part`, `shutdown`, `reboot`.
- Сохраняются также проверки, не относящиеся к policy: узел должен существовать
  в живом каталоге; `operator_controls` по-прежнему запрещены даже при `force`;
  для не-protective команд нужна свежая before-телеметрия (иначе команда
  блокируется); действует lock «действие уже в полёте».
- Аутентификация, rate limit (`AGENT_MIN_SEND_GAP`), аудит и JSONL-логи
  сохраняются.
- Каждая команда логируется с `unsafe=true`; `verify=false`.
- Использовать только на тестовом куполе.

### `dry-run`

`AGENT_DRY_RUN=1` или `--dry-run`:

- не отправляет control-команды в Dome;
- не отправляет сообщения в чат (пишет `message` с `dry_run=true, sent=false`);
- не пишет action intent в state и не обновляет cooldown успеха/неуспеха;
- в `normal`/`auto` прогоняет все действия правил через очередь и policy gate,
  чтобы проверить логику;
- в `yolo` правила не исполняет; если LLM доступна, её команды уходят в
  `DomeTool` с `dry_run=true`.

## 7. Safety-цепочка

Порядок защиты от «модель что-то придумала» до физического действия:

```text
модель / правило
      │
      ▼
1. Hard denylist (только модельный путь) ──► block, forceable=false
      │
      ▼
2. CommandGate (policy + control switch) ──► block/forceable
      │
      ▼
3. Role check (team/admin; кроме dry-run)
      │
      ▼
4. Cooldown / recent failure / action lock / dependencies
      │
      ▼
5. Fresh before-telemetry (кроме auto-protective)
      │
      ▼
6. Отправка команды + JSONL action
      │
      ▼
7. ExpectedEffect по after/follow-up telemetry
      │
      ▼
8. State: success/failure cooldown, intent
      │
      ▼
9. Уведомление Оператора при failed/unknown
```

Цепочка не одинакова для всех путей:

- действия rule engine всегда проходят шаги 1–9 (шаг 1 для них не нужен, но
  gate и verification применяются);
- модельные команды в `normal`/`auto` проходят 1–9;
- в `yolo` для модельных команд пропускаются шаги 2 и 7 и cooldown-проверки
  шага 4, но hard denylist, lock, role/сессия, каталог, before-телеметрия для
  не-protective действий и аудит остаются;
- в `dry-run` команда не отправляется: before/after-телеметрия и проверка
  эффекта не выполняются, action-запись пишется с `dry_run=true`, а state и
  cooldown не меняются.

### 7.1. Hard denylist

`HARD_DENY_ACTIONS` в `dome_agent/control.py`:

```text
block_automation
block_interlock
set_thresholds
unlock_door
remove_part
shutdown
reboot
```

Он применяется к `ModelCommand` при парсинге и в `DomeTool.execute`, а также
повторно в `DomeAgent._execute_model_command`. `force` его не отменяет.
Правила не могут обойти policy gate; их действия проверяются тем же gate.

### 7.2. CommandGate

`dome_agent/gate.py` работает fail-closed:

- `dry_run` → разрешено, команда не отправляется;
- `yolo` → разрешено, `unsafe=true`;
- `control_enabled=false` → запрещено;
- action в `deny_actions` → запрещено;
- node в `deny_nodes` → запрещено;
- node отсутствует в `allow` → запрещено;
- action отсутствует в списке node → запрещено;
- если для action задан список `values`, значение должно в него входить.

Policy загружается только для `control_enabled=true` и не-`yolo`. Отсутствующий
или некорректный JSON, `default != "deny"` или отсутствие `allow` приводят к
`GateError` при старте.

### 7.3. Проверка эффекта

`dome_agent/verification.py` содержит явные предикаты для действий:
`line_on/off`, `reset_protection`, `start_job`, `resume`, `home`,
`turn_on/off`, `stop_job`, `reconnect`, `set_mode`, `set_primary_sensor`,
`reset_interlock`, `reset_error`, `set_fan`, `set_data_trust`.

- Для известных действий проверяется физическое состояние, а не только
  `success` ответа API.
- Метаданные ответа (`node_id`, `action`, `request_id`, `success`, `timestamp`,
  `accepted`, `verified` и т.п.) не считаются доказательством эффекта.
- `DomeAgentAPI.control_device` требует свежий before-кадр, если действие не
  auto-protective; иначе возвращает `blocked` без отправки.
- После команды `ExpectedEffect.check(after)`; если предикат вернул `None`
  (неизвестное действие), используется `meaningful_diff`, но только по
  не-метаданным.
- Если эффект не подтверждён сразу, агент ждёт новые кадры до `timeout_s`
  (по умолчанию 6 s), затем фиксирует `verification` с `ok=false`.
- `unknown` (например, команда ушла, но before-телеметрия недоступна для
  auto-protective) обрабатывается как неизвестный исход, а не как успех.

### 7.4. Cooldowns и повторные попытки

- Rule engine проверяет `StateStore.action_allowed(key, cooldown_s)` и
  `action_failed_recently(key)`; в `yolo` эти проверки для правил обходятся, но
  правила в `yolo` и не исполняются.
- После неуспешной проверки/ошибки действия ключ блокируется на
  `ACTION_FAILURE_COOLDOWN_S = 900 s`.
- Одинаковое действие в полёте защищено `_pending_action_keys` по
  `node:action:value`.
- Зависимости (`depends_on`) выполняются в топологическом порядке; цикл
  зависимостей блокируется, а не исполняется.
- Очередь — отдельная lane на `node_id`, поэтому команды одного узла
  сериализуются, а разные узлы могут выполняться параллельно.

### 7.5. Fail-closed в правилах

Правила не угадывают при неполных данных. Примеры:

- неизвестная причина `hold_reason` → resume запрещён;
- нет данных CO2/климата/дыма/пожарной автоматики → производство не
  запускается, resume не выполняется;
- нет `bed_clear` → стол считается неподтверждённым, запуск печати запрещён;
- нет номера линии/файла → задание переводится в `needs_info` и Оператору
  задаётся вопрос;
- малформированные interlocks нормализуются; неизвестное состояние трактуется
  как небезопасное;
- критический алерт без отдельной ветки восстановления всё равно блокирует
  производство и создаёт сообщение.

## 8. Rule engine

`RuleEngine.decide()` имеет два уровня:

1. `_safety()` — аварии и восстановление.
2. `_task()` — выполнение задания Оператора.
3. Если ничего не сработало — «Нет требующих вмешательства событий».

### 8.1. Порядок safety-веток

Фактический порядок в `_safety()`:

1. Пожарная автоматика: `alarm` или `state=DISCHARGED`.
2. Дым по датчикам.
3. Ложное срабатывание CO2: изолированный высокий датчик + два текущих
   безопасных датчика в той же зоне.
4. Реальный/неподтверждённый высокий CO2: вентиляция, но без resume.
5. Сброс `IL_CO2`, если CO2 нормализовался, а интерлок остался.
6. Resume станков, если причина удержания известна и устранена.
7. Инвертор: `error_code`; сброс только в сухую погоду, иначе ожидание.
8. Перегрев инвертора > 75 °C: включить вентилятор.
9. Фантомная нагрузка линии 8 ≥ 20 W при включённой линии.
10. Восстановление линии 3 (печать) или 4 (CNC), если она нужна заданию:
    сначала reset protection, затем line_on.
11. Низкий SOC АКБ (<20%) при дефиците/отсутствии сети: дизель, если
    autostart разрешён, дизель исправен, топливо качественное и >10%.
12. Оставшиеся критические алерты без отдельной ветки — сообщение Оператору.

Пожар и дым имеют абсолютный приоритет: автоматического возобновления
производства нет.

### 8.2. CO2

Константы в `analysis.py`:

- датчики: `climate_sensor_01..03` плюс любые `climate_sensor_*`;
- trip: 2000 ppm;
- «душно»: 1000 ppm;
- spread консенсуса: 200 ppm;
- outlier gap: 400 ppm;
- outlier ratio: 1.8;
- положительный `reading_delay_s` исключает показание из текущего кворума.

Логика:

- невалидные/offline/`UNRELIABLE`/`FAILED`/`FAULT`/`ERROR`/низкая confidence
  датчики не участвуют;
- delayed-показания сохраняются в отчёте, но не могут опровергнуть свежую
  аварию;
- ложное срабатывание: изолированный высокий датчик + минимум два текущих
  безопасных датчика в его зоне;
- реальное событие: минимум два высоких надёжных показания при консенсусе или
  все надёжные показания высокие;
- при текущем высоком и отсутствии false alarm выбирается максимальное высокое
  значение (safety-first);
- иначе используется сохранённый primary override, если он валиден и не
  suspect; затем значение, ближайшее к медиане; при отсутствии кластера —
  максимальное значение, чтобы не занизить опасность;
- после успешного `set_primary_sensor` агент сохраняет `co2_override` в state.

### 8.3. Alerts

`Snapshot.alerts()` собирает нормализованные алерты, сортирует по severity
(`critical` → `warning` → `info`) и кэширует на кадр. Примеры кодов:
`SMOKE`, `SMOKE_OFFLINE`, `FIRE`, `FIRE_NOT_READY`, `CO2_TRIP`,
`CO2_SENSOR_SUSPECT`, `CO2_UNCONFIRMED`, `CONDENSATE`, `CO`, `VOC`,
`RADIATION`, `CHEM`, `SEISMIC`, `FUME_ALARM`, `FUME_FLOW`, `BATTERY_LOW`,
`INVERTER_ERROR`, `MACHINE_ERROR`, `MACHINE_HOLD_ALARM`, `PANEL_LINE_ALARM`,
`NETWORK_DOWN`, `SEAL_LEAK`, `ACCESS`, `INTERLOCK` и другие.

Важно: алерт — это диагностика. Разрешение на действие всё равно даёт rule
engine + gate + verification.

### 8.4. Задания Оператора

`chat.parse_operator_task` разбирает сообщение на независимые clause-ы:

- `print` — 3D-печать, файл `.gcode`/`.gco`/и т.п.;
- `cnc` — фрезеровка, `.nc`/`.ngc`/`.tap` и т.п.;
- `line_on` / `line_off` — линия 1..8;
- `report`, `inspect`, `monitor`/`shift`;
- иначе — `needs_info`/неизвестное задание, агент просит уточнение.

Задание сохраняется в state, позиции получают статусы (`pending`, `starting`,
`started`, `homing`, `needs_info`, `done` и т.п.). Для print/CNC:

- offline → `reconnect` (priority 80);
- завершение проверяется `check_print_completion`/`check_cnc_completion`, а не
  только webhook;
- ошибка/ALARM → задание ждёт Оператора;
- нет файла/файла нет в каталоге → вопрос Оператору;
- уже RUNNING/PRINTING с тем же файлом → не запускать повторно;
- hold/ALARM → не запускать;
- перед стартом проверяются CO2, дым, пожарная готовность, климат, питание,
  качество воздуха, конденсат, вытяжка для CNC;
- для печати нужен `bed_clear=true` и `homed`; при неизвестном столе запуск
  запрещён;
- для линии 3/4 при необходимости сначала reset protection, затем line_on.

### 8.5. Приоритеты действий

Значения `ActionSpec.priority` в текущем коде включают:

| Приоритет | Примеры |
|---|---|
| 100 | CO2 trust / primary / reset interlock |
| 95, 94 | CO2 ventilation on / FRESH_AIR |
| 90 | reset IL_CO2 после нормализации |
| 85 | resume станка |
| 80 | reconnect, fan инвертора |
| 78 | запуск дизеля |
| 74, 72 | reset/line_on для линии задания |
| 70, 68, 67, 65 | reset линии, home, вытяжка, start_job |
| 60 | line_on/off, phantom line 8 |

Приоритет влияет на порядок независимых действий; зависимости всё равно
имеют приоритет через топологическую сортировку.

## 9. Чат и подтверждения

- `meeting_01` — общий чат. Агент принимает только сообщения от Оператора
  (`role=operator/admin` или имя содержит «оператор») и только адресованные
  ему/команде/всем; сообщения другой команде игнорируются.
- Опрос: раз в 3 s, `limit=50`; раз в 300 s — полная сверка. При первом запуске
  (cursor=0) backlog старше 5 минут пропускается, чтобы не выполнять старые
  задания.
- `last_message_id` и `handled_message_ids` хранятся в state; при очистке
  истории сервером (`meeting_cleared`) cursor сбрасывается.
- Входящее сообщение логируется, разбирается в задание, отмечается обработанным.
- Completion guards не доверяют одному `webhook_state`: проверяются state,
  progress, Z, paused, hold.
- Подтверждения предложений обрабатываются в `normal` до новых решений; в
  `yolo` подтверждения не нужны.

## 10. LLM-слой

`DeepSeekAdvisor`:

- использует только `DEEPSEEK_BASE_URL` (по умолчанию
  `https://api.deepseek.com`; путь `/chat/completions` добавляется
  автоматически);
- доступен, если `AGENT_LLM=1` и задан `DEEPSEEK_API_KEY`;
- `advise()` — диагностический текст для `auto`; ошибки/таймауты/плохой JSON
  возвращают `None`, агент продолжает работу;
- `decide()` — строгий JSON `ModelDecision` для `normal`/`yolo`;
- не более 6 команд за решение; неизвестные tool и hard-denied actions
  отбрасываются с warning;
- контекст обрезается до 14 000 символов;
- вызовы ограничены `AGENT_LLM_MIN_INTERVAL` и
  `AGENT_LLM_MAX_CALLS_PER_HOUR`; сообщение Оператора может форсировать вызов
  control, но часовой бюджет остаётся;
- в лог пишутся только метаданные (`llm_call`: model, purpose, latency, tokens,
  summary), без prompt-ов и ключей;
- нет shell/subprocess tool. Модель может вызвать только `control`, `query`,
  `telemetry`, `catalog`, `knowledge`, `send_message`.

Контекст `normal`/`yolo` включает: `mode`, `snapshot.brief()`,
`rule_decision`, task, recent chat, новые сообщения Оператора, pending,
hard-denied actions, auto-protective список, policy summary и compact catalog
(до 40 узлов).

`dome_control.agent_api.TOOL_SCHEMAS` — отдельный набор из 10 JSON-схем для
внешнего фасада. Runtime-инструменты агента — только пять перечисленных выше.

### 10.1. База знаний (RAG)

`KnowledgeService` (`dome_agent/knowledge.py`) — тонкий read-only мост к
`dome_rag`:

- лениво импортирует `dome_rag.builder.retrieval`/`answer` при первом запросе;
- читает `AGENT_KB_DB` (по умолчанию `<AGENT_KB_DIR>/build/kb.sqlite`) и
  карточки из `<AGENT_KB_DIR>/data`; отсутствие БД — не ошибка, а
  «свидетельств нет»;
- запросы сериализуются и выполняются в отдельном потоке
  (`asyncio.to_thread`), чтобы не блокировать event loop;
- `pack()` отдаёт модели компактный пакет (вердикт, confidence, facts,
  hypotheses, actions, sources, unknown) в пределах
  `AGENT_KB_MAX_CHARS`;
- пакет добавляется в контекст `decide`/`advise`; краткие ссылки
  (`RuleResult.rag_refs`) попадают в prompt. События `knowledge` пишут
  `purpose`, `question`, `verdict`, `confidence`, `sources`;
- ошибка базы логируется и не ломает цикл; policy gate, hard denylist,
  cooldowns и проверка эффекта от RAG не зависят.

## 11. Очередь и исполнение действий

`ActionSpec` содержит:

```text
key, node_id, action, value (NO_VALUE = без value), reason,
verify, priority, depends_on, expect (ExpectedEffect)
```

Путь действия:

1. `RuleEngine` создаёт action только если cooldown разрешает.
2. `_enqueue_actions` дедуплицирует по `key`, сортирует по зависимостям,
   проверяет preflight (цикл/неисполненная зависимость) и ставит в lane узла.
3. `_run_action` проверяет роль (кроме dry-run), ждёт зависимости, выполняет
   `_execute_protective` или `_execute_action`.
4. `_execute_action` проверяет policy, свежий before, recent failure, lock,
   сохраняет intent (`state.begin_action`) до отправки.
5. `DomeAgentAPI.control_device` валидирует каталог/alias, отправляет команду,
   получает after.
6. `_verify_action` проверяет `ExpectedEffect`, при необходимости ждёт новые
   кадры.
7. `_mark_action` фиксирует success/failure и cooldown; при неуспехе Оператору
   уходит сообщение.
8. `finish_action` удаляет intent.

Статусы: `queued`, `running`, `succeeded`, `failed`, `blocked`, `unknown`.
`ACTION_TERMINAL_STATUSES` завершает ожидание зависимостей. `running` для
повторного нажатия означает «уже в полёте».

### 11.1. Что переживает рестарт

Очередь и in-flight задачи **не** сохраняются. После рестарта агент:

- читает телеметрию заново и принимает новое решение;
- поднимает `pending_actions` из state и трактует их как `unknown`, блокируя
  повтор на 900 s;
- пишет `action_reconcile` и уведомляет Оператора.

## 12. State и логи

### 12.1. State

Файл по умолчанию `logs/agent_state.json`, режим `0600`, атомарная замена:

```text
version, last_message_id, handled_message_ids, task, operator_requests,
last_actions, failed_until, pending_actions, co2_override, notes
```

- `task` — текущее задание и статусы его пунктов;
- `last_message_id`/`handled_message_ids` — cursor чата (до 300 id);
- `last_actions` — последние 200 действий и success;
- `failed_until` — до 200 cooldown-ов после неуспеха;
- `pending_actions` — до 50 intent-ов «команда отправлена, эффект не подтверждён»;
- `operator_requests` — до 80 записей дедупликации сообщений;
- `co2_override` — выбранный primary-датчик;
- `notes` — зарезервировано.

При повреждении state сохраняется копия `agent_state.json.corrupt-<ns>` с
режимом `0600`, агент стартует с дефолтным state и пишет `error` с
`where=state_load`. Директория state создаётся при необходимости. `save()`
делает `fsync` temp-файла, `os.replace`, `fsync` директории (best effort).
`updated_at` в state не пишется; дашборд подставляет mtime файла.

### 12.2. JSONL

Файл: `AGENT_LOG_DIR/agent_<run_id>.jsonl`, `run_id` — локальное время
`YYYYMMDD-HHMMSS`, при коллизии добавляется `-2`, `-3`, ...

Каждая строка:

```json
{"ts": "2026-10-27T12:00:00.000+03:00", "run_id": "20261027-120000", "seq": 1, "kind": "start", "...": "..."}
```

- запись сбрасывается на диск после каждой строки;
- `seq` монотонно растёт в рамках run;
- `dome_time` добавляется, если известен игровой время;
- секреты заменяются на `[REDACTED]`, поля с именами `*_key`, `token`,
  `secret`, `password` очищаются;
- файл открывается с `0600`;
- штатное завершение всегда пишет `stop`, кроме аварийного kill -9.

Типы записей: `start`, `stop`, `connection`, `observation`, `decision`,
`action`, `action_blocked`, `action_queue`, `action_reconcile`,
`verification`, `message`, `llm_call`, `knowledge`, `error`. Подробные поля — в
[REFERENCE.md](REFERENCE.md).

## 13. Read-only API журнала

`dome_web` — отдельный процесс, не влияет на агента:

- `create_app()` читает `AGENT_LOG_DIR`/`AGENT_STATE_FILE` при каждом запросе;
- берёт самый новый `agent_*.jsonl` по mtime, читает хвост до 4 MB;
- игнорирует битые/частичные строки;
- `/api/health`, `/api/summary`, `/api/logs`, `/api/state`, `/api/demo`;
- `Cache-Control: no-store` для `/api/*`;
- state проходит whitelist и sanitization; секретные поля удаляются;
- `/api/logs` проверяет `run_id` регуляркой и не позволяет traversal;
- `/api/state` не отдаёт `pending_actions`, `handled_message_ids` и прочие
  внутренние поля;
- `/` и `/static` относятся к фронтенду и в этой документации не
  рассматриваются.

Health-статусы: `ok`, `waiting`, `stopped`, `stale` (последняя запись старше
120 s), `error`.

## 14. Инварианты

Изменения кода не должны нарушать эти свойства без явного пересмотра
документации и тестов:

1. Управление по умолчанию выключено; `AGENT_ALLOW_CONTROL=0` не отправляет
   control-команды.
2. `dome_policy.json` fail-closed: неизвестный узел/действие запрещены.
3. Hard denylist нельзя обойти `force`, подтверждением или `yolo`.
4. Safety-действия не зависят от доступности LLM.
5. Команда не считается успешной только по `success` API; нужен физический
   эффект или явное `unknown`.
6. Все действия и решения попадают в JSONL с `run_id`/`seq`.
7. Секреты не попадают в логи, state и HTTP-ответы панели.
8. State переживает рестарт; незавершённые команды не повторяются слепо.
9. Один процесс-агент владеет одним state-файлом; параллельный запуск двух
   агентов на одном state не поддерживается.
10. Нет shell/subprocess-инструмента у LLM.

## 15. Ограничения

- RAG — read-only источник свидетельств: пакет знаний попадает в контекст
  LLM и в события `knowledge`, но не является разрешением на команду. Вердикт
  базы нельзя повышать; `NO_RELEVANT_EVIDENCE` означает отказ. Несобранная
  база не блокирует запуск агента.
- Веб-панель не имеет аутентификации; по умолчанию слушает `127.0.0.1:8080`,
  её нельзя выставлять в сеть без отдельного прокси/аутентификации.
- Агент не является системой реального времени: цикл 1 s, телеметрия может
  быть stale, проверка эффекта — best effort с таймаутом.
- В `normal` без DeepSeek работают только auto-protective действия правил;
  остальные предложения не создаются.
- В `yolo` policy gate, cooldowns и проверка эффекта отключены для модельных
  команд. Каталог, hard denylist, `operator_controls`, lock и требование
  before-телеметрии для не-protective действий сохраняются; это осознанно
  опасный режим.
- Задание «стоп/отменить» не имеет отдельной безопасной ветки правил;
  отмена задания Оператором обрабатывается как новое указание/уточнение.
- Тесты используют mock Dome и фикстуры, а не реальный купол.
