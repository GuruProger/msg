# Эксплуатация backend

## 1. Требования

- Python 3.14+.
- Зависимости из `requirements.txt` / `pyproject.toml`: FastAPI, httpx,
  uvicorn, websockets.
- Сетевой доступ к Dome WebSocket API (`ws://` или `wss://`).
- Ключ команды Dome.
- Для LLM-режимов — ключ DeepSeek. Без него агент работает в rules-only
  варианте (см. ограничения ниже).

## 2. Установка

```bash
cd /path/to/msg
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Для установки console scripts:

```bash
pip install .
```

После этого доступны `dome-agent`, `dome-panel`, `dome-dashboard`.

## 3. Первичная настройка

```bash
cp .env.example .env
chmod 600 .env
```

Заполнить минимум:

```dotenv
DOME_URL=ws://<host>:8765
DOME_KEY_FILE=/path/outside/repo/team3.key
# или DOME_KEY=<key>
```

Формат key-файла: строка `team: <key>`. Key-файл предпочтительнее `.env`:
его можно держать вне репозитория и не копировать в логи/архивы.

`.env` не должен попадать в git. Секреты не печатать в issue/логи. Если
`DOME_KEY` и `DOME_KEY_FILE` заданы одновременно, используется `DOME_KEY`.
Если оба пусты — конфигурация завершится ошибкой.

Проверка конфигурации без запуска цикла:

```bash
python main.py --help
python -m dome_agent --help
```

## 4. Запуск

### 4.1. Одна команда

```bash
python main.py
```

Эквиваленты:

```bash
python -m dome_agent
dome-agent
```

По умолчанию:

- режим `normal`;
- управление выключено (`AGENT_ALLOW_CONTROL=0`);
- агент подключается, читает телеметрию, пишет state/логи и сообщения
  Оператору, но **не отправляет команды оборудованию**.

### 4.2. Нормальный управляющий режим

```bash
AGENT_ALLOW_CONTROL=1 AGENT_MODE=normal python main.py
# или
python main.py --allow-control --mode normal
```

`--allow-control` только включает мастер-разрешение. Policy всё равно
проверяется. `AGENT_ALLOW_CONTROL` не отменяет `dome_policy.json`.

### 4.3. Rules-only

```bash
AGENT_ALLOW_CONTROL=1 AGENT_MODE=auto python main.py
```

LLM может быть недоступна (`AGENT_LLM=0` или пустой `DEEPSEEK_API_KEY`) —
`auto` продолжит работать на правилах.

### 4.4. YOLO (только тестовый купол)

```bash
AGENT_ALLOW_CONTROL=1 AGENT_MODE=yolo python main.py
# или
python main.py --yolo
```

Будет предупреждение в stderr. В этом режиме LLM выполняет команды сразу,
policy/cooldowns/verification обходятся; hard denylist остаётся. При этом
каталог, `operator_controls`, lock и свежая before-телеметрия для
не-protective действий всё ещё проверяются. Не использовать на production.

### 4.5. Ограниченный прогон

```bash
python main.py --run-seconds 60
```

`0` (по умолчанию) — работать до сигнала остановки.

### 4.6. Dry-run

```bash
python main.py --dry-run
AGENT_DRY_RUN=1 python main.py
```

Ничего не отправляется в Dome и чат. В `normal`/`auto` действия всё равно
проходят через очередь/policy/логи, чтобы проверить поведение. В `yolo`
правила не исполняются, а LLM-команды уходят в dry-run-исполнитель.

## 5. Режимы и что ожидать

| Запуск | Команды | LLM | Подтверждение |
|---|---|---|---|
| без флагов | нет (control off) | `normal` может предлагать, но control выключен | — |
| `--allow-control --mode normal` | auto-protective сразу, остальное после «да» | предложения | да/нет/force |
| `--allow-control --mode auto` | все действия правил | только advisory | не требуется |
| `--mode yolo` / `--yolo` | LLM сразу | управляет | не требуется |
| `--dry-run` | нет | может вызываться | как в выбранном режиме |

`AGENT_YOLO=1` — совместимый алиас: он выбирает `yolo`, только если
`AGENT_MODE` не задан (пустой `AGENT_MODE` тоже считается незаданным). Если
заданы оба, приоритет у явного `AGENT_MODE`.

## 6. Policy

Файл по умолчанию: `dome_policy.json`, путь переопределяется
`AGENT_POLICY_FILE`.

```bash
python -m json.tool dome_policy.json >/dev/null && echo POLICY_JSON_OK
```

Требования:

- JSON-объект;
- `default` равен `"deny"`;
- есть объект `allow`;
- `deny_actions`/`deny_nodes` применяются раньше allow;
- значения `values` проверяются, если для действия задан список.

Текущий allowlist включает:

- `dome_automation_01`: `set_data_trust`, `set_primary_sensor`,
  `reset_interlock` (сброс только `IL_CO2`);
- `smart_panel_01`: `line_on`, `line_off`, `reset_protection` (линии 1–8);
- `supply_ventilation_01`: `turn_on`, `set_mode=FRESH_AIR`;
- `solar_inverter_01`: `set_fan` (`ON`/`OFF`), `reset_error`;
- `dizel_1`: `turn_on`;
- `fume_extraction_01`: `turn_on`;
- `printer_3d_01`, `cnc_01`: `start_job`, `resume`, `home`, `reconnect`.

Запрещённые узлы: `access_control_01`, `dome_sealing_01`,
`fire_suppression_01`. Запрещённые действия включают `block_automation`,
`block_interlock`, `cancel_job`, `force`, `reboot`, `remove_part`,
`set_thresholds`, `shutdown`, `unlock_door`.

Менять policy нужно осознанно: unknown node/action при `default=deny`
блокируются. После изменения policy перезапустить агента — gate загружает файл
при старте. Ошибка/отсутствие policy при включённом control останавливает
запуск.

## 7. Логи

Каталог: `AGENT_LOG_DIR` (по умолчанию `logs`). Файл:
`logs/agent_<run_id>.jsonl`.

Полезные команды:

```bash
ls -lt logs | head
tail -n 20 logs/agent_*.jsonl
grep '"kind": "action_blocked"' logs/agent_*.jsonl
grep '"kind": "verification"' logs/agent_*.jsonl | tail
```

Чтение через Python, если нет `jq`:

```bash
python - <<'PY'
import glob, json
path = sorted(glob.glob('logs/agent_*.jsonl'))[-1]
for line in open(path, encoding='utf-8'):
    record = json.loads(line)
    if record.get('kind') in {'action', 'action_blocked', 'verification', 'error'}:
        print(record.get('seq'), record.get('kind'), record.get('reason') or record.get('error') or record.get('expected'))
PY
```

Что искать:

- `decision` — что решили и почему (`source`: `rules`/`llm`);
- `action_blocked` — policy, role, cooldown, before telemetry, dependency;
- `verification` — подтвердился ли физический эффект;
- `action_reconcile` — команда была в полёте при рестарте;
- `llm_call` — стоимость/задержка/ошибки LLM, без prompt-ов;
- `stop` — причина завершения и статистика.

Секреты в логах заменяются. Если секрет всё же попал в строку — считать это
инцидентом, ротировать ключ и проверить redaction.

## 8. State и перезапуск

Файл: `AGENT_STATE_FILE` (по умолчанию `logs/agent_state.json`).

- Агент пишет state атомарно и с `0600`.
- Перезапуск безопасен: задание, chat cursor, cooldowns и CO2 override
  сохраняются.
- Очередь команд не сохраняется; после рестарта решение принимается заново.
- Если процесс умер после отправки команды, при старте появится
  `action_reconcile`; повтор этой команды блокируется на 15 минут, Оператор
  получает предупреждение.
- Повреждённый state не мешает старту: создаётся приватная копия
  `agent_state.json.corrupt-<ns>`, state сбрасывается, в лог пишется `error`.
- Не запускать два агента на одном state-файле.

Проверить состояние:

```bash
python -m json.tool logs/agent_state.json | head -60
```

Не редактировать state вручную при работающем агенте. Если нужно сбросить
задание — остановить агента, сделать backup, изменить/удалить state, запустить
снова.

## 9. Штатная остановка

- `Ctrl+C` → SIGINT; в systemd/Docker — SIGTERM.
- Агент перестаёт брать новые действия, до 7 s дренирует очередь, отменяет
  LLM-задачи, закрывает соединения и пишет `stop`.
- `SIGKILL`/`kill -9` не даёт штатного `stop`; возможны `pending_actions`,
  которые будут согласованы при следующем старте.

Проверка, что процесс завершился:

```bash
ps -ef | grep '[d]ome_agent'
tail -n 1 logs/agent_*.jsonl
```

## 10. Дашборд (только backend)

```bash
python -m dome_dashboard
# или
dome-dashboard --host 127.0.0.1 --port 8080
```

Переменные:

- `DASHBOARD_HOST` (по умолчанию `127.0.0.1`);
- `DASHBOARD_PORT` (по умолчанию `8080`);
- `AGENT_LOG_DIR`, `AGENT_STATE_FILE` — что читать.

Проверка:

```bash
curl -s http://127.0.0.1:8080/api/health
curl -s http://127.0.0.1:8080/api/summary | head
curl -s 'http://127.0.0.1:8080/api/logs?kind=action&limit=20'
```

Дашборд:

- только читает JSONL и state;
- не отправляет команды и не пишет в файлы агента;
- не имеет аутентификации, поэтому не выставлять в интернет;
- читает последние 4 MB лога; старые записи могут быть не видны в summary;
- `/api/state` отдаёт только безопасное подмножество state;
- `/api/demo` работает без файлов и сети.

Если дашборд не нужен, его можно не запускать: агент от него не зависит.

## 11. Административный CLI

`dome_control` — ручной инструмент, не автономный агент:

```bash
python -m dome_control whoami
python -m dome_control doctor
python -m dome_control status
python -m dome_control telemetry
python -m dome_control catalog
python -m dome_control control smart_panel_01 line_on 3 --yes
python -m dome_control query meeting_01 read_messages
python -m dome_control chat
python -m dome_control say "текст"
```

Общие опции: `--url`, `--key`, `--key-file`, `--env-file`, `--timeout`,
`--json`, `--log-file`, `--log-telemetry`, `--verbose`, `--quiet`,
`--demo`.

CLI также валидирует действия по живому каталогу и умеет `--demo` на
телеметрии. Для production-смены использовать `dome_agent`, а не CLI.

## 12. Диагностика

| Симптом | Вероятная причина | Что проверить |
|---|---|---|
| `agent configuration error: DOME_URL...` | URL пуст/не `ws://`/`wss://` | `.env`, `--url` |
| `DOME_KEY is empty` | нет ключа | `DOME_KEY`/`DOME_KEY_FILE`, формат `team: ...` |
| `GateError: policy file not found` | нет `dome_policy.json` | путь `AGENT_POLICY_FILE`, рабочая директория |
| `GateError: policy...` | policy не denial-by-default/нет allow | `python -m json.tool`, `default=deny` |
| `authentication failed` | неверный ключ/URL | `python -m dome_control whoami` |
| Долго `guest` | смена команды не активна | лог `connection`, `AGENT_GUEST_RECHECK` |
| Нет команд | `AGENT_ALLOW_CONTROL=0` | лог `action_blocked`, флаг control |
| `not in the policy allowlist` | узел/действие не разрешены | policy, `allow` |
| `fresh before-telemetry unavailable` | узел отсутствует/телеметрия stale | `status`, `telemetry`, каталог |
| `verification ok=false` | команда не дала физического эффекта | `verification.expected/observed`, повторить вручную после проверки |
| `LLM is not configured` | нет ключа или `AGENT_LLM=0` | `DEEPSEEK_API_KEY`, `AGENT_LLM` |
| `LLM не ответила` | таймаут/бюджет/битый JSON | `llm_call` в логе, `DEEPSEEK_TIMEOUT` |
| Предложение не выполняется | ждёт «да»/force или истекло 5 минут | чат, `PendingProposal` в контексте |
| Дашборд `waiting` | нет лога/записи | `AGENT_LOG_DIR`, запущен ли агент |
| Дашборд `stale` | последняя запись старше 120 s | процесс агента, сеть |
| Порт занят | другой процесс | `ss -ltnp`, `DASHBOARD_PORT` |
| State повреждён | crash/ручное редактирование | `.corrupt-*`, backup, перезапуск |

Для воспроизведения без реального купола:

```bash
python tests/mock_dome_api.py --help
python -m unittest discover -s tests
```

## 13. Проверка перед включением управления

1. Тестовый купол или подтверждённое окно на production.
2. `dome_policy.json` проверен и соответствует задаче.
3. `AGENT_ALLOW_CONTROL=1` выставлен осознанно.
4. `AGENT_MODE=normal` (или `auto`) выбран; `yolo` — только для теста.
5. Секреты в `.env`/key-файле, права `0600`, `.env` не в git.
6. Есть свежая телеметрия и активная смена (`role=team`).
7. Понятно, кто подтверждает предложения в чате.
8. Проверены логи `action_blocked`/`verification` после первого действия.
9. Известно, как остановить процесс и где state/логи.
10. Для `normal` проверено наличие DeepSeek или принято, что доступны только
    auto-protective действия.

## 14. Тесты и проверка изменений

```bash
python -m unittest discover -s tests
python -m compileall -q dome_agent dome_control dome_dashboard tests
```

Опционально:

```bash
python tests/acceptance_dome_cli.py
```

`ruff` в текущем окружении не установлен; если он появится, запускать
`ruff check dome_agent dome_control tests` как дополнительный шаг.
