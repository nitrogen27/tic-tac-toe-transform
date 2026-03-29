# Architecture Decisions

Ключевые решения принятые в проекте и их обоснование.

---

## AD-001: Plain Makefile вместо nx/turborepo

**Решение:** Единый `Makefile` в корне репозитория.

**Почему:** Polyglot стек (C++, Python, TypeScript) — ни nx, ни turborepo не дают
нативной поддержки CMake. Makefile понятен любому разработчику без обучения.

---

## AD-002: JSON Schema как единый источник типов

**Решение:** `packages/shared/schemas/*.schema.json` → codegen в TS и Python.

**Почему:** Три языка (C++, Python, TypeScript) должны использовать одни и те же
контракты. JSON Schema — нейтральный формат. Codegen предотвращает расхождение типов.

---

## AD-003: CMake + FetchContent без vcpkg

**Решение:** Зависимости C++ (nlohmann-json, GTest, benchmark) через FetchContent.

**Почему:** vcpkg не был доступен в среде разработки. FetchContent работает из коробки,
не требует установки. vcpkg поддерживается как опциональный fallback через `vcpkg.json`.

---

## AD-004: CLI subprocess bridge вместо FFI/gRPC

**Решение:** FastAPI запускает `gomoku-engine` как subprocess, JSON через stdin/stdout.

**Почему:** Проще всего для начального этапа. gRPC добавил бы сложность сборки C++.
Python→C++ FFI (ctypes/pybind11) усложняет деплой. Subprocess проще изолировать и тестировать.

**Таймаут:** 30 секунд на subprocess.communicate(). Fallback — центральный ход.

---

## AD-005: Все размеры доски → padding до 16×16 в нейросети

**Решение:** Входной тензор фиксированный `[B, 6, 16, 16]` для любого boardSize (7–16).

**Почему:** ONNX экспорт требует фиксированных размеров. Padding нулями не влияет на
качество т.к. plane 5 (board_size_mask) явно маркирует активную область.

---

## AD-006: D4 симметрия для аугментации данных

**Решение:** 8 трансформаций (4 поворота × 2 зеркала) применяются к position+policy.

**Почему:** Гомоку симметрично по D4 группе → 8× увеличение обучающей выборки без
потери корректности. Policy вектор трансформируется той же функцией, что и доска.

---

## AD-007: setTimeout(4ms) вместо microtask для MCTS батчинга (legacy)

**Решение:** `flushTimer = setTimeout(flushEvalQueue, 4)` вместо `Promise.resolve().then()`

**Почему:** Microtask flush срабатывает до того, как накапливается батч. 4ms задержка
позволяет накопить 96+ позиций → один forward pass GPU вместо 96 маленьких.
Это главное изменение давшее +40W GPU мощности в legacy стеке.

---

## AD-008: MAX_SAFE_BATCH=256 в legacy TF.js

**Решение:** Жёсткий кап batch size на 256, даже если UI присылает 1024.

**Почему:** RTX 3060 Laptop 6GB в WDDM режиме имеет ~3.5GB свободной VRAM
(~2.5GB занято Windows GUI процессами). При batch>256 с TF_FORCE_GPU_ALLOW_GROWTH=false
возникал OOM. Кап в коде сервера защищает независимо от настроек UI.

---

## AD-009: Legacy стек остаётся нетронутым

**Решение:** `server/` и `client/` не изменяются в ходе разработки V3 (кроме GPU-fix).

**Почему:** Legacy — работающий продукт. Новая платформа строится параллельно
на отдельной ветке `feature/gomoku-platform-v3`. Миграция — отдельный этап.

---

## AD-010: useReducer + Context вместо Zustand/Redux

**Решение:** `gameStore.tsx` — React Context + useReducer.

**Почему:** Состояние игры простое и хорошо структурированное. Нет async side effects
в store (async только в хуках). Встроенный React, нет зависимостей. Достаточно для MVP.
