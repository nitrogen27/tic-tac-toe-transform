# Transformer для крестики-нолики 3×3

Полная реализация PV Transformer модели с обучением и оценкой качества.

## Структура файлов

1. **config.mjs** - Конфигурация (SEED, TRANSFORMER_CFG, TRAIN)
2. **game_ttt3.mjs** - Игровая логика (доска, ходы, кодирование)
3. **ttt3_minimax.mjs** - Minimax с мемоизацией и генератор датасета
4. **model_pv_transformer_seq.mjs** - PV Transformer модель (seq 9)
5. **train_ttt3_transformer.mjs** - Обучение с ранним стопом
6. **safety.mjs** - Safety-правила (выигрыш/блокировка)
7. **eval_ttt3.mjs** - Оценка качества модели

## Использование

### Обучение модели

```bash
cd server
node src/train_ttt3_transformer.mjs
```

Обучение:
- Использует teacher minimax для генерации датасета
- Ранний стоп при достижении accuracy ≥ 99.9% и MAE ≤ 1e-3
- Сохраняет модель в `saved/ttt3_transformer/`

### Оценка качества

```bash
cd server
node src/eval_ttt3.mjs
```

Оценка включает:
- Юнит-тесты базовых тактик
- 1000 игр против minimax (критерий: 0 поражений)
- Sanity-чек против случайного (должна выигрывать >> 50%)

## Конфигурация

```javascript
export const BOARD_N = 3;
export const SEED = 42;
export const TRANSFORMER_CFG = {
  dModel: 128,
  numLayers: 4,
  heads: 4,
  dropout: 0.0
};
export const TRAIN = {
  batchSize: 512,
  epochs: 2, // Оптимизировано для быстрого обучения
  lr: 5e-4,
  weightValue: 0.5  // Вес value loss
};
```

## Архитектура модели

- **Вход**: [B, 9, 3] (3 плоскости: my/opponent/empty)
- **Embedding**: Linear до dModel + Positional embedding
- **Transformer блоки**: LN → MHA → residual → LN → FFN → residual
- **Policy head**: Dense(9) → логиты (маскирование снаружи)
- **Value head**: GlobalAvgPool → MLP → tanh

## Safety-правила

Модель использует safety-правила перед выбором хода:
1. Мгновенный выигрыш (если есть)
2. Блокировка выигрыша оппонента (если есть)
3. Argmax от policy (иначе)

## Критерии успеха

- ✅ 0 поражений против minimax (1000 игр)
- ✅ Доля ничьих ≈ 100%
- ✅ Accuracy оптимальных ходов ≥ 99.9%
- ✅ MAE value ≤ 1e-3

## Пример использования модели

```javascript
import { forward } from './eval_ttt3.mjs';
import { safePick } from './safety.mjs';

const { policy, value } = await forward(model, board, player);
const move = safePick(board, player, policy);
```
