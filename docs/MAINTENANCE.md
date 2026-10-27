# Поддержка документации

Документация должна меняться вместе с кодом. Этот файл — процедура для
следующих изменений: пользователь присылает diff/патч, мы обновляем
затронутые разделы и фиксируем новый снимок.

## 1. Принципы

1. Источник истины — текущий код, а не эта документация.
2. Документация описывает backend: `dome_agent`, `dome_control`,
   `dome_dashboard/app.py` и read-only мост `dome_agent/knowledge.py`.
   Фронтенд `dome_dashboard/static` и внутреннее устройство `dome_rag` —
   вне scope, если пользователь явно не попросит обратное.
3. Не документировать поведение, которого нет в коде. Если код и README
   расходятся — фиксировать фактическое поведение и отдельно отметить
   расхождение.
4. Не копировать значения секретов, ключи, токены и содержимое `.env` в docs,
   логи, примеры или commit message.
5. Имена переменных, endpoint-ов, полей JSON и классов писать как в коде;
   пояснения — по-русски.

## 2. Куда смотреть при изменении

| Что изменилось | Обновить в первую очередь |
|---|---|
| `dome_agent/config.py`, `.env.example`, `pyproject.toml` | `REFERENCE.md` (env/CLI), `OPERATIONS.md` (запуск), `README.md` снимок |
| `dome_agent/agent.py` (цикл, режимы, shutdown, очередь) | `ARCHITECTURE.md` (жизненный цикл, режимы, action pipeline) |
| `dome_agent/rules.py` | `ARCHITECTURE.md` (порядок safety/task, приоритеты) |
| `dome_agent/analysis.py` | `ARCHITECTURE.md` (CO2, alerts, Snapshot) |
| `dome_agent/chat.py` | `ARCHITECTURE.md` (чат/задания), `REFERENCE.md` (state.task) |
| `dome_agent/control.py`, `gate.py` | `ARCHITECTURE.md` (safety), `REFERENCE.md` (policy/hard deny) |
| `dome_agent/verification.py` | `ARCHITECTURE.md` (проверка эффекта) |
| `dome_agent/state.py` | `REFERENCE.md` (state), `ARCHITECTURE.md` (рестарт) |
| `dome_agent/logging.py` | `REFERENCE.md` (JSONL), `ARCHITECTURE.md` (аудит) |
| `dome_agent/llm.py` | `ARCHITECTURE.md` (LLM), `REFERENCE.md` (env LLM) |
| `dome_control/client.py`, `agent_api.py`, `domain.py` | `ARCHITECTURE.md` (транспорт), `REFERENCE.md` (API/ошибки) |
| `dome_dashboard/app.py` | `REFERENCE.md` (HTTP API), `OPERATIONS.md` (дашборд) |
| `dome_control/cli.py`, `panel.py` | `OPERATIONS.md` (админ-CLI), `REFERENCE.md` |
| `dome_policy.json` | `REFERENCE.md` (таблица policy), `OPERATIONS.md` |
| `tests/` | `REFERENCE.md` (тесты), при изменении инвариантов — `ARCHITECTURE.md` |
| `README.md` | Ссылка на docs и снимок |

## 3. Процедура после получения изменений

1. **Зафиксировать состояние**
   ```bash
   git status --short
   git diff --stat
   git log --oneline -3
   ```
   Проверить, что изменено: только backend или также фронтенд/`dome_rag`.
2. **Прочитать изменённый код**, а не только diff. Для новых функций найти
   вызывающие места и тесты.
3. **Сопоставить с документацией** по таблице выше. Если изменился инвариант
   (режим, safety, формат лога, state, endpoint), обновить не только пример, но
   и пояснение/таблицу.
4. **Обновить снимок** в `docs/README.md`: ветка, commit, дата, Python,
   локальные незакоммиченные изменения. Если рабочее дерево dirty — указать
   это явно.
5. **Проверить примеры**:
   ```bash
   python main.py --help
   python -m dome_control --help
   python -m dome_dashboard --help
   ```
   Все флаги/команды в docs должны существовать.
6. **Проверить ссылки и имена**:
   ```bash
   grep -R "AGENT_" docs README.md | sort -u
   grep -R "/api/" docs | sort -u
   ```
   Сверить с `dome_agent/config.py`, `dome_dashboard/app.py`.
7. **Прогнать тесты и компиляцию**:
   ```bash
   python -m unittest discover -s tests
   python -m compileall -q dome_agent dome_control dome_dashboard tests
   ```
   Если тесты изменились количественно — обновить цифру в `REFERENCE.md`.
8. **Проверить отсутствие секретов**:
   ```bash
   git diff -- docs README.md
   grep -RniE "api[_-]?key|token|secret|password|DOME_KEY=" docs README.md
   ```
   Совпадения имён переменных допустимы; значения — нет.
9. **Проверить diff документации** и только после этого завершать задачу.

## 4. Что считать изменением поведения

Обновлять документацию обязательно, если:

- добавлен/удалён режим или флаг;
- изменился default env;
- изменился порядок safety-веток или hard denylist;
- изменился policy allowlist/denylist;
- изменился формат `state` или JSONL;
- добавлен/удалён endpoint или поле ответа;
- изменился способ проверки эффекта;
- изменился формат задания/подтверждения;
- изменился контур зависимостей (например, `dome_rag` начал
  импортироваться агентом);
- изменился способ запуска/имя entry point;
- изменились тесты, подтверждающие инвариант.

Недостаточно обновить только пример, если изменилась семантика. Например,
если `normal` начнёт исполнять все действия правил, нельзя оставлять таблицу
режимов старой.

## 5. Инварианты, которые нельзя потерять при обновлении

Перед коммитом документации проверить, что она всё ещё утверждает:

1. Управление по умолчанию выключено.
2. Policy fail-closed, неизвестные узлы/действия запрещены.
3. Hard denylist не обходится `force`/`yolo`.
4. Safety-правила не зависят от LLM.
5. Успех команды подтверждается телеметрией, а не только ответом API.
6. JSONL содержит `run_id`/`seq`, пишется с flush и redaction.
7. State атомарный, переживает рестарт, intent-ы согласуются.
8. Дашборд только читает и не имеет аутентификации.
9. LLM не имеет shell/subprocess.
10. `dome_rag` описан только как read-only источник свидетельств; его
    внутреннее устройство вне scope, пока оно не влияет на backend-контракт.

Если изменение намеренно ломает инвариант — это нужно явно указать в docs и
в ответе пользователю, а не молча переписать текст.

## 6. Стиль

- Заголовки и пояснения — по-русски.
- Код, имена файлов, переменных, endpoint-ов — в backticks.
- Таблицы — для перечислений; длинные поля JSON — в блоках кода.
- Примеры команд — POSIX shell, без секретов.
- Не обещать «полную безопасность»: описывать конкретные слои и их границы.
- Для потенциально опасных режимов (`yolo`) явно писать «только тестовый
  купол».
- Ссылки между docs — относительные, чтобы работали в Git-просмотрщике.

## 7. Быстрый чек-лист

- [ ] Изменённый код прочитан, включая вызовы и тесты.
- [ ] Обновлены затронутые разделы по таблице.
- [ ] Примеры команд/флагов существуют.
- [ ] Имена env/endpoint/полей совпадают с кодом.
- [ ] Снимок в `docs/README.md` обновлён.
- [ ] Тесты и `compileall` проходят.
- [ ] Секретов в docs нет.
- [ ] `README.md` ссылается на docs.
- [ ] В ответе пользователю перечислены обновлённые файлы и проверки.
