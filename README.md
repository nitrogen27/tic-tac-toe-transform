# Tic-Tac-Toe Transformer Fullstack (M2 Optimized)

Полнофункциональное приложение для игры в крестики-нолики с использованием Transformer модели на TensorFlow.js.

## Особенности

- 🤖 **Transformer модель** на TensorFlow.js для обучения игры
- 🚀 **Оптимизировано для Apple M2** процессоров
- 🎮 **Два режима игры**: против модели или против алгоритма minimax
- 🤖 **Автоматический режим**: модель играет против бота
- 🔄 **WebSocket** коммуникация для обучения и предсказаний
- 📊 **Визуальный прогресс** обучения с детальной статистикой

## Технологии

- **Frontend**: Vue.js 3, Vite
- **Backend**: Node.js, WebSocket (ws)
- **ML**: TensorFlow.js (Node.js backend)
- **Оптимизация**: Worker threads для генерации датасета

## Установка

### Вариант 1: Docker (рекомендуется) 🐳

Самый простой способ запустить проект без проблем с зависимостями:

## Быстрый старт

### ⚠️ Важно: GPU vs CPU режим

**`npm start`** запускает проект **локально** (CPU режим) - GPU недоступен без Docker.

**Для GPU режима** используйте:
```bash
npm run docker:up
# или
npm run start:gpu
```

### Локальный запуск (CPU режим)

```bash
# Установка зависимостей (первый раз)
npm install

# Запуск всего проекта локально (CPU)
npm start

# Остановка всех процессов и контейнеров
npm stop

# Перезапуск проекта
npm restart
```

Это запустит:
- Сервер на порту 8080 (WebSocket) - **CPU режим**  
- Клиент на порту 5173 (Vite dev server)

**Примечание:** 
- `npm start` - локальный запуск, **только CPU**
- `npm run docker:up` или `npm run start:gpu` - запуск в Docker с **GPU поддержкой**
- `npm stop` останавливает все Docker контейнеры и локальные процессы проекта
- `npm restart` останавливает всё и запускает заново

### Docker запуск

```bash
# Убедитесь, что Docker Desktop запущен
docker ps

# GPU версия (требует NVIDIA GPU и nvidia-container-toolkit)
docker-compose -f docker-compose.gpu.yml up --build

# Или CPU версия
docker-compose up --build
```

После запуска:
- Сервер: `ws://localhost:8080` (WebSocket)
- Клиент: `http://localhost:5173`

Подробнее: [QUICK_DOCKER_START.md](QUICK_DOCKER_START.md) или [DOCKER_SETUP.md](DOCKER_SETUP.md)

## Проверка GPU

При запуске сервера проверьте логи:
```
[TFJS] Using tfjs-node-gpu backend (CUDA support)
[TFJS] Backend: tensorflow (gpu)
[TrainTTT3] GPU acceleration: ENABLED ✓
```

Если GPU недоступен:
```
[TFJS] WARNING: Backend is not GPU! Check NVIDIA/CUDA/cuDNN installation.
[TrainTTT3] GPU acceleration: DISABLED ✗
```

**Все операции TensorFlow.js автоматически выполняются на GPU при использовании `@tensorflow/tfjs-node-gpu`** - тензоры, model.fit(), model.predict() и все операции размещаются на GPU автоматически.

### Вариант 2: Локальная установка

#### Требования

- **Node.js v18.x или v20.x** (LTS версии)
- **Visual Studio Build Tools 2022** с компонентом "Desktop development with C++" (для Windows)
- **CUDA Toolkit** (опционально, для GPU/CUDA поддержки)

```bash
# Установка зависимостей
npm install

# Запуск сервера и клиента одновременно
npm start

# Или отдельно:
npm run server  # Сервер на ws://localhost:8080
npm run client  # Клиент на http://localhost:5173
```

**Примечание:** Если возникают проблемы с установкой TensorFlow.js на Windows, используйте Docker (Вариант 1).

### Поддержка CPU и GPU

Проект автоматически определяет и использует:
- **CPU (x86_64)** - через `@tensorflow/tfjs-node`
- **CUDA GPU** - через `@tensorflow/tfjs-node-gpu` (если доступен)

При запуске сервера проверьте логи для определения используемого backend.

## Использование

1. **Обучение модели**: Нажмите кнопку "Обучить" для обучения модели с нуля
2. **Очистка модели**: Используйте "Очистить модель" чтобы удалить сохраненную модель
3. **Режим игры**: Выберите между "Модель" и "Алгоритм" для выбора противника
4. **Автоматическая игра**: Включите режим "Автоматическая игра" для наблюдения за игрой модели против бота

## Структура проекта

```
.
├── client/          # Vue.js фронтенд
│   ├── src/
│   │   ├── App.vue  # Основной компонент
│   │   └── main.js
│   └── package.json
├── server/          # Node.js бэкенд
│   ├── src/
│   │   ├── model_transformer.mjs  # Transformer модель
│   │   ├── dataset.mjs            # Генерация датасета
│   │   ├── tic_tac_toe.mjs         # Логика игры и minimax
│   │   └── tf.mjs                  # TensorFlow.js настройка
│   ├── server.mjs   # WebSocket сервер
│   └── service.mjs  # Бизнес-логика
└── package.json
```

## Оптимизации для M2

- Параллельная генерация датасета через worker threads
- Alpha-beta pruning в алгоритме minimax
- Оптимизированный batch size и TensorFlow.js настройки
- Нативные биндинги TensorFlow для ускорения

## Лицензия

Private
