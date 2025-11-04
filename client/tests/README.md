# Тесты клиента

## Структура тестов

```
tests/
├── setup.js              # Настройка тестового окружения
├── unit/                 # Unit тесты
│   ├── game-logic.test.js      # Тесты игровой логики
│   └── websocket-utils.test.js # Тесты утилит WebSocket
├── integration/          # Интеграционные тесты
│   ├── websocket.test.js       # Тесты WebSocket соединения
│   └── rest-api.test.js        # Тесты REST API
└── e2e/                 # End-to-end тесты
    └── game-flow.test.js       # Полный цикл игры
```

## Запуск тестов

```bash
# Все тесты
npm test

# С UI интерфейсом
npm run test:ui

# С покрытием кода
npm run test:coverage

# Только unit тесты
npm test -- tests/unit

# Только интеграционные тесты
npm test -- tests/integration

# Только E2E тесты
npm test -- tests/e2e
```

## Требования для интеграционных тестов

Интеграционные тесты требуют запущенного сервера:

```bash
# В одном терминале - запустить сервер
cd server
npm start

# В другом терминале - запустить тесты
cd client
npm test
```

## Что тестируется

### Unit тесты
- ✅ Игровая логика (getWinner, legalMoves)
- ✅ Утилиты WebSocket (parseMessage, sendMessage)
- ✅ Обработка различных состояний игры

### Интеграционные тесты
- ✅ Подключение к WebSocket серверу
- ✅ Обмен сообщениями (ping/pong, predict, train, pv.infer, health)
- ✅ Policy+Value inference через WebSocket
- ✅ Health check через WebSocket
- ✅ Обработка ошибок

### E2E тесты
- ✅ Полный цикл игры
- ✅ Отслеживание истории игр
- ✅ Сброс состояния игры

## Примечания

- Интеграционные тесты автоматически пропускаются, если сервер недоступен
- Для тестов, требующих сервер, используйте переменную окружения `TEST_SERVER_URL`
- E2E тесты могут быть медленными из-за реальных сетевых запросов

## Покрытие кода

После запуска `npm run test:coverage` отчет будет доступен в `coverage/` директории.
