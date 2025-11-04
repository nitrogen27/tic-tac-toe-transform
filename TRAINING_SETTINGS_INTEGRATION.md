# Интеграция настроек обучения между UI и сервером

## ✅ Выполнено

### 1. Настройки основного обучения в UI
Добавлены настройки в `client/src/App.vue`:
- **Количество эпох** (`mainTrainingEpochs`): 1-10, по умолчанию 2
- **Размер батча** (`mainTrainingBatchSize`): 128-4096, по умолчанию 1024

### 2. Настройки дообучения в UI
Обновлены настройки в `client/src/App.vue`:
- **Количество эпох** (`trainingEpochs`): 1-10, по умолчанию 1
- **Размер батча** (`incrementalBatchSize`): 32-1024, по умолчанию 256
- **Вариаций паттерна на ошибку** (`patternsPerError`): 10-2000, по умолчанию 1000

### 3. Связь с сервером
- Все настройки передаются через WebSocket в соответствующих payload
- Сервер валидирует и ограничивает значения:
  - Основное обучение: epochs 1-10, batchSize 128-4096
  - Дообучение: epochs 1-10, batchSize 32-1024, patternsPerError 10-2000

### 4. Интеграция в функции
- `startTrain()` использует `mainTrainingEpochs` и `mainTrainingBatchSize`
- `trainOnGames()` использует `trainingEpochs`, `incrementalBatchSize` и `patternsPerError`
- `finishGame()` использует `patternsPerError` и `incrementalBatchSize` для фонового обучения

### 5. Тесты
Созданы и пройдены тесты:
- `test_training_settings.test.mjs` - проверка связи настроек дообучения
- `test_main_training_settings.test.mjs` - проверка валидации настроек

## Результаты тестов

```
✓ tests/test_training_settings.test.mjs (6 tests)
  ✓ should use epochs from payload in trainOnGames
  ✓ should use default values when not provided
  ✓ should use incremental batch size from settings
  ✓ should handle different pattern counts per error

✓ tests/test_main_training_settings.test.mjs (4 tests)
  ✓ should limit epochs to 10 for main training
  ✓ should accept batch size in valid range
  ✓ should validate training settings from UI
  ✓ should validate incremental training settings from UI

Test Files: 2 passed (2)
Tests: 10 passed (10)
```

## Структура настроек

### Основное обучение (train_ttt3)
```javascript
{
  epochs: mainTrainingEpochs.value,    // 1-10
  batchSize: mainTrainingBatchSize.value, // 128-4096
  earlyStop: true
}
```

### Дообучение (train_on_games)
```javascript
{
  epochs: trainingEpochs.value,        // 1-10
  batchSize: incrementalBatchSize.value, // 32-1024
  patternsPerError: patternsPerError.value, // 10-2000
  focusOnErrors: true
}
```

### Фоновое обучение (finish_game)
```javascript
{
  patternsPerError: patternsPerError.value, // 10-2000
  autoTrain: autoTrainAfterGame.value,       // boolean
  incrementalBatchSize: incrementalBatchSize.value // 32-1024
}
```

## Валидация на сервере

### server/server.mjs
- Основное обучение: ограничивает epochs до 10, batchSize до 128-4096
- Дообучение: валидирует все параметры в допустимых диапазонах
- Фоновое обучение: использует переданные настройки

### server/service.mjs
- `trainOnGames()` принимает и использует все настройки из payload
- `finishGame()` передает `incrementalBatchSize` в `startBackgroundTraining()`
- `startBackgroundTraining()` использует переданный `incrementalBatchSize`

## Использование

1. **Основное обучение**: Настройте `mainTrainingEpochs` и `mainTrainingBatchSize` в UI, нажмите "Обучить с нуля"
2. **Дообучение**: Настройте `trainingEpochs`, `incrementalBatchSize` и `patternsPerError` в UI, нажмите "Дообучить на играх"
3. **Фоновое обучение**: Включите `autoTrainAfterGame`, настройки будут использоваться автоматически после каждой игры

## Тесты

Все тесты проходят успешно:
```bash
docker exec tic-tac-toe-server bash -c "cd /app/server && npx vitest run tests/test_training_settings.test.mjs tests/test_main_training_settings.test.mjs"
```

