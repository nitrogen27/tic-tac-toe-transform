// Общая конфигурация игры и модели
export const BOARD_N = 3;               // для масштабирования поменяйте на 10
export const SEED = 42;                 // Seed для воспроизводимости
export const VOCAB_SIZE = 3;            // значения клетки: 0=пусто, 1=мой, 2=оппонент (или 3-категории)
export const TRANSFORMER_CFG = {
  dModel: 192,                          // Размерность модели (оптимизировано для баланса загрузки GPU и памяти)
  numLayers: 5,                         // Количество слоев (оптимизировано для баланса)
  heads: 6,                             // Количество голов внимания (оптимизировано для баланса)
  dropout: 0.0                          // Dropout (0.0 = без dropout)
};
export const TRAIN = {
  batchSize: 4096,                      // Размер батча (оптимизировано для быстрого обучения)
  epochs: 1,                            // Количество эпох (оптимизировано для быстрого обучения)
  lr: 5e-4,                             // Learning rate
  weightValue: 0.5                       // Вес value loss относительно policy loss
};
export const PV_RESNET = {
  filters: 64,                          // Количество фильтров в ResNet
  blocks: 8                              // Количество ResNet блоков
};
