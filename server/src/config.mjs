// Общая конфигурация игры и модели
export const BOARD_N = 3;               // для масштабирования поменяйте на 10
export const SEED = 42;                 // Seed для воспроизводимости
export const VOCAB_SIZE = 3;            // значения клетки: 0=пусто, 1=мой, 2=оппонент (или 3-категории)
export const TRANSFORMER_CFG = {
  dModel: 128,                          // Размерность модели
  numLayers: 4,                         // Количество слоев
  heads: 4,                             // Количество голов внимания
  dropout: 0.0                          // Dropout (0.0 = без dropout)
};
export const TRAIN = {
  batchSize: 512,                       // Размер батча
  epochs: 10,                            // Количество эпох
  lr: 5e-4,                             // Learning rate
  weightValue: 0.5                       // Вес value loss относительно policy loss
};
export const PV_RESNET = {
  filters: 64,                          // Количество фильтров в ResNet
  blocks: 8                              // Количество ResNet блоков
};
