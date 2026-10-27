# Backend-документация Dome

Документация описывает backend-часть проекта: автономного агента, транспортный
слой, read-only API дашборда и read-only мост к базе знаний. Фронтенд
`dome_dashboard/static` и внутреннее устройство пакета `dome_rag` в этот контур
не входят: агент использует RAG только как источник свидетельств.

## Снимок, для которого написана документация

| Параметр | Значение |
|---|---|
| Ветка | `main` |
| Коммит | `7f5c111` |
| Дата снимка | 2026-10-27 |
| Python | 3.14.8 |
| Обновление | интеграция `dome_agent/knowledge.py` с `dome_rag`, пакет `dome_web`, обновлённые control/gate/dashboard/policy и тесты |
| Вне scope | внутреннее устройство `dome_rag`, `dome_web` и фронтенд `dome_dashboard/static` |

Документация отражает **рабочее дерево**, а не только последний коммит. При
следующем изменении кода нужно обновить затронутые разделы и снимок в этой
таблице — процедура описана в [MAINTENANCE.md](MAINTENANCE.md).

## Состав документации

| Файл | Назначение |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Компоненты, поток данных, жизненный цикл агента, режимы, safety-цепочка, правила, LLM, state и логи |
| [OPERATIONS.md](OPERATIONS.md) | Установка, запуск, `.env`, режимы, policy, dry-run, логи, восстановление, дашборд, troubleshooting |
| [REFERENCE.md](REFERENCE.md) | Справочник по переменным окружения, CLI, policy, форматам логов/state и HTTP API |
| [MAINTENANCE.md](MAINTENANCE.md) | Как поддерживать документацию в актуальном состоянии после изменений кода |

## Границы backend-контура

```text
main.py / python -m dome_agent
        │
        ▼
dome_agent/            автономный агент: цикл, правила, LLM, safety, state, JSONL
        │
        ├── dome_control/agent_api.py   async-фасад над WebSocket API
        ├── dome_control/client.py      WebSocket-транспорт и корреляция запросов
        └── dome_control/domain.py      каталог, alias-ы, валидация, telemetry diff

dome_dashboard/        read-only FastAPI над JSONL-логом и state-файлом
dome_control/cli.py    административный CLI (не автономный агент)
```

Важно:

- `dome_agent` лениво импортирует `dome_rag` через `dome_agent/knowledge.py`;
  база знаний доступна только для чтения и не влияет на policy gate, hard
  denylist и выполнение команд. Если база не собрана, агент работает без неё.
- `dome_dashboard.app` только читает лог и state. Он не отправляет команды в Dome и
  не изменяет файлы агента.
- По умолчанию управляющие команды запрещены (`AGENT_ALLOW_CONTROL=0`). Агент при
  этом всё равно читает телеметрию, пишет state/логи и может отправлять сообщения
  оператору; «read-only» относится к командам оборудования, а не к чату.
- Секреты хранятся в `.env` или key-файле. В документации и логах их значений быть
  не должно.

## С чего начать

- Нужно запустить агента: [OPERATIONS.md](OPERATIONS.md).
- Нужно понять, почему действие было выполнено или заблокировано:
  [ARCHITECTURE.md](ARCHITECTURE.md), раздел «Safety-цепочка».
- Нужны точные имена переменных, полей логов или endpoint-ов:
  [REFERENCE.md](REFERENCE.md).
- Пришли изменения кода: [MAINTENANCE.md](MAINTENANCE.md).
