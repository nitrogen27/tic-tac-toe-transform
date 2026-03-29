# Shared Agent Instructions

Правила и контекст общий для всех агентов в проекте.

---

## Репозиторий

```
Repo:    https://github.com/nitrogen27/tic-tac-toe-transform.git
Branch:  feature/gomoku-platform-v3
Dir:     C:\gitlab\ml\tic-tac-toe-transform\
Shell:   bash (Unix-синтаксис, даже на Windows)
```

## Перед началом работы

Всегда читать:
1. `.claude/rules/reference.md` — полный технический референс
2. `.claude/memory/tasks-completed.md` — что уже сделано
3. Файлы которые планируешь изменять — перед редактированием

## Правила редактирования файлов

1. Читать файл перед изменением (`Read` tool)
2. Использовать `Edit` для точечных изменений, `Write` только для новых файлов
3. Не создавать файлы без явной необходимости
4. Не добавлять комментарии в код без запроса
5. Не рефакторить то, что не связано с задачей

## Git-конвенции

```bash
# Staged только нужные файлы (не git add -A)
git add apps/api/src/gomoku_api/services/engine_adapter.py
git add apps/api/tests/test_engine_router.py

# Commit message формат:
# feat: Phase N — краткое описание (1 строка)
# Опционально: 3–5 bullet points что изменено

git push origin feature/gomoku-platform-v3
```

## Не коммитить

- `__pycache__/`, `*.pyc`, `*.egg-info/`
- `node_modules/`, `dist/`, `.venv/`
- `engine-core/build/`
- `.env` файлы

## Тесты перед коммитом

```bash
# Backend
cd apps/api && pytest tests/ -q

# Frontend
cd apps/web && npm run test -- --run

# C++ engine
cd engine-core/build && ctest --output-on-failure
```

## Стиль ответов

- Отвечать по-русски
- Коротко — факты, имена файлов, команды
- Сначала результат, потом объяснение если нужно
- Не пересказывать что было сделано — показать diff/результат

## Структура проекта

Смотри: `.claude/rules/reference.md#monorepo-layout`

## Контракты API

Смотри: `.claude/memory/integration-contracts.md`

## Архитектурные решения

Смотри: `.claude/memory/architecture-decisions.md`
