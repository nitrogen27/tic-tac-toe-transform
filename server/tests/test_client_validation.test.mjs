// Тесты для проверки валидации настроек на клиенте
import { describe, it, expect } from 'vitest';

describe('Client-side Validation', () => {
  // Имитируем функции валидации из клиента
  function validateMainTrainingEpochs(epochs) {
    if (epochs < 1) return { valid: false, error: 'Количество эпох должно быть от 1 до 10' };
    if (epochs > 10) return { valid: false, error: 'Количество эпох должно быть от 1 до 10' };
    if (!Number.isInteger(epochs)) return { valid: false, error: 'Количество эпох должно быть целым числом' };
    return { valid: true };
  }

  function validateMainTrainingBatchSize(batchSize) {
    if (batchSize < 128) return { valid: false, error: 'Размер батча должен быть от 128 до 4096' };
    if (batchSize > 4096) return { valid: false, error: 'Размер батча должен быть от 128 до 4096' };
    if (!Number.isInteger(batchSize)) return { valid: false, error: 'Размер батча должен быть целым числом' };
    if (batchSize % 128 !== 0) return { valid: false, error: 'Размер батча должен быть кратен 128' };
    return { valid: true };
  }

  function validateTrainingEpochs(epochs) {
    if (epochs < 1) return { valid: false, error: 'Количество эпох должно быть от 1 до 10' };
    if (epochs > 10) return { valid: false, error: 'Количество эпох должно быть от 1 до 10' };
    if (!Number.isInteger(epochs)) return { valid: false, error: 'Количество эпох должно быть целым числом' };
    return { valid: true };
  }

  function validateIncrementalBatchSize(batchSize) {
    if (batchSize < 32) return { valid: false, error: 'Размер батча должен быть от 32 до 1024' };
    if (batchSize > 1024) return { valid: false, error: 'Размер батча должен быть от 32 до 1024' };
    if (!Number.isInteger(batchSize)) return { valid: false, error: 'Размер батча должен быть целым числом' };
    if (batchSize % 32 !== 0) return { valid: false, error: 'Размер батча должен быть кратен 32' };
    return { valid: true };
  }

  function validatePatternsPerError(patterns) {
    if (patterns < 10) return { valid: false, error: 'Вариаций паттерна должно быть от 10 до 2000' };
    if (patterns > 2000) return { valid: false, error: 'Вариаций паттерна должно быть от 10 до 2000' };
    if (!Number.isInteger(patterns)) return { valid: false, error: 'Вариаций паттерна должно быть целым числом' };
    if (patterns % 10 !== 0) return { valid: false, error: 'Вариаций паттерна должно быть кратно 10' };
    return { valid: true };
  }

  describe('Main Training Epochs Validation', () => {
    it('should accept valid epochs (1-10)', () => {
      expect(validateMainTrainingEpochs(1).valid).toBe(true);
      expect(validateMainTrainingEpochs(2).valid).toBe(true);
      expect(validateMainTrainingEpochs(5).valid).toBe(true);
      expect(validateMainTrainingEpochs(10).valid).toBe(true);
    });

    it('should reject epochs below 1', () => {
      expect(validateMainTrainingEpochs(0).valid).toBe(false);
      expect(validateMainTrainingEpochs(-1).valid).toBe(false);
    });

    it('should reject epochs above 10', () => {
      expect(validateMainTrainingEpochs(11).valid).toBe(false);
      expect(validateMainTrainingEpochs(20).valid).toBe(false);
    });

    it('should reject non-integer epochs', () => {
      expect(validateMainTrainingEpochs(2.5).valid).toBe(false);
      expect(validateMainTrainingEpochs(1.1).valid).toBe(false);
    });
  });

  describe('Main Training Batch Size Validation', () => {
    it('should accept valid batch sizes (128-4096, multiple of 128)', () => {
      expect(validateMainTrainingBatchSize(128).valid).toBe(true);
      expect(validateMainTrainingBatchSize(512).valid).toBe(true);
      expect(validateMainTrainingBatchSize(1024).valid).toBe(true);
      expect(validateMainTrainingBatchSize(2048).valid).toBe(true);
      expect(validateMainTrainingBatchSize(4096).valid).toBe(true);
    });

    it('should reject batch sizes below 128', () => {
      expect(validateMainTrainingBatchSize(64).valid).toBe(false);
      expect(validateMainTrainingBatchSize(0).valid).toBe(false);
    });

    it('should reject batch sizes above 4096', () => {
      expect(validateMainTrainingBatchSize(4097).valid).toBe(false);
      expect(validateMainTrainingBatchSize(8192).valid).toBe(false);
    });

    it('should reject batch sizes not multiple of 128', () => {
      expect(validateMainTrainingBatchSize(129).valid).toBe(false);
      expect(validateMainTrainingBatchSize(513).valid).toBe(false);
      expect(validateMainTrainingBatchSize(1000).valid).toBe(false);
    });
  });

  describe('Incremental Training Epochs Validation', () => {
    it('should accept valid epochs (1-10)', () => {
      expect(validateTrainingEpochs(1).valid).toBe(true);
      expect(validateTrainingEpochs(5).valid).toBe(true);
      expect(validateTrainingEpochs(10).valid).toBe(true);
    });

    it('should reject invalid epochs', () => {
      expect(validateTrainingEpochs(0).valid).toBe(false);
      expect(validateTrainingEpochs(11).valid).toBe(false);
      expect(validateTrainingEpochs(2.5).valid).toBe(false);
    });
  });

  describe('Incremental Batch Size Validation', () => {
    it('should accept valid batch sizes (32-1024, multiple of 32)', () => {
      expect(validateIncrementalBatchSize(32).valid).toBe(true);
      expect(validateIncrementalBatchSize(64).valid).toBe(true);
      expect(validateIncrementalBatchSize(256).valid).toBe(true);
      expect(validateIncrementalBatchSize(512).valid).toBe(true);
      expect(validateIncrementalBatchSize(1024).valid).toBe(true);
    });

    it('should reject batch sizes below 32', () => {
      expect(validateIncrementalBatchSize(16).valid).toBe(false);
      expect(validateIncrementalBatchSize(0).valid).toBe(false);
    });

    it('should reject batch sizes above 1024', () => {
      expect(validateIncrementalBatchSize(1025).valid).toBe(false);
      expect(validateIncrementalBatchSize(2048).valid).toBe(false);
    });

    it('should reject batch sizes not multiple of 32', () => {
      expect(validateIncrementalBatchSize(33).valid).toBe(false);
      expect(validateIncrementalBatchSize(100).valid).toBe(false);
      expect(validateIncrementalBatchSize(513).valid).toBe(false);
    });
  });

  describe('Patterns Per Error Validation', () => {
    it('should accept valid patterns (10-2000, multiple of 10)', () => {
      expect(validatePatternsPerError(10).valid).toBe(true);
      expect(validatePatternsPerError(100).valid).toBe(true);
      expect(validatePatternsPerError(1000).valid).toBe(true);
      expect(validatePatternsPerError(2000).valid).toBe(true);
    });

    it('should reject patterns below 10', () => {
      expect(validatePatternsPerError(9).valid).toBe(false);
      expect(validatePatternsPerError(0).valid).toBe(false);
    });

    it('should reject patterns above 2000', () => {
      expect(validatePatternsPerError(2001).valid).toBe(false);
      expect(validatePatternsPerError(5000).valid).toBe(false);
    });

    it('should reject patterns not multiple of 10', () => {
      expect(validatePatternsPerError(11).valid).toBe(false);
      expect(validatePatternsPerError(105).valid).toBe(false);
      expect(validatePatternsPerError(1001).valid).toBe(false);
    });
  });

  describe('Complete Settings Validation', () => {
    it('should validate complete main training settings', () => {
      const mainEpochs = validateMainTrainingEpochs(2);
      const mainBatch = validateMainTrainingBatchSize(1024);
      expect(mainEpochs.valid && mainBatch.valid).toBe(true);
    });

    it('should validate complete incremental training settings', () => {
      const incEpochs = validateTrainingEpochs(1);
      const incBatch = validateIncrementalBatchSize(256);
      const patterns = validatePatternsPerError(1000);
      expect(incEpochs.valid && incBatch.valid && patterns.valid).toBe(true);
    });

    it('should reject invalid complete settings', () => {
      const mainEpochs = validateMainTrainingEpochs(20); // Invalid
      const mainBatch = validateMainTrainingBatchSize(100); // Invalid
      expect(mainEpochs.valid && mainBatch.valid).toBe(false);
    });
  });
});

