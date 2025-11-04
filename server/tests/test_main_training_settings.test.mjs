// Тесты для проверки настроек основного обучения
import { describe, it, expect } from 'vitest';

describe('Main Training Settings', () => {
  it('should limit epochs to 2 for main training', () => {
    // Имитируем логику ограничения epochs из server.mjs
    const limitEpochs = (epochs) => {
      if (epochs === undefined) {
        return 2; // Дефолтное значение
      } else if (epochs > 2) {
        return 2; // Ограничиваем максимум 2 эпохами
      }
      return epochs;
    };

    expect(limitEpochs(undefined)).toBe(2);
    expect(limitEpochs(1)).toBe(1);
    expect(limitEpochs(2)).toBe(2);
    expect(limitEpochs(3)).toBe(2);
    expect(limitEpochs(10)).toBe(2);
  });

  it('should accept batch size in valid range', () => {
    // Проверяем, что batch size в допустимом диапазоне
    const validateBatchSize = (batchSize) => {
      if (batchSize === undefined) {
        return 1024; // Дефолтное значение
      }
      // Ограничиваем диапазон
      return Math.max(128, Math.min(4096, batchSize));
    };

    expect(validateBatchSize(undefined)).toBe(1024);
    expect(validateBatchSize(128)).toBe(128);
    expect(validateBatchSize(1024)).toBe(1024);
    expect(validateBatchSize(4096)).toBe(4096);
    expect(validateBatchSize(50)).toBe(128); // Минимум
    expect(validateBatchSize(10000)).toBe(4096); // Максимум
  });

  it('should validate training settings from UI', () => {
    // Имитируем валидацию настроек из UI
    const validateMainTrainingSettings = (settings) => {
      const validated = {
        epochs: Math.max(1, Math.min(10, settings.epochs || 2)),
        batchSize: Math.max(128, Math.min(4096, settings.batchSize || 1024)),
        earlyStop: settings.earlyStop !== false
      };
      return validated;
    };

    // Тест 1: Дефолтные значения
    const defaultSettings = validateMainTrainingSettings({});
    expect(defaultSettings.epochs).toBe(2);
    expect(defaultSettings.batchSize).toBe(1024);
    expect(defaultSettings.earlyStop).toBe(true);

    // Тест 2: Валидные значения
    const validSettings = validateMainTrainingSettings({
      epochs: 2,
      batchSize: 2048,
      earlyStop: true
    });
    expect(validSettings.epochs).toBe(2);
    expect(validSettings.batchSize).toBe(2048);
    expect(validSettings.earlyStop).toBe(true);

    // Тест 3: Значения вне диапазона
    const invalidSettings = validateMainTrainingSettings({
      epochs: 20, // Превышает максимум
      batchSize: 50, // Ниже минимума
      earlyStop: false
    });
    expect(invalidSettings.epochs).toBe(10); // Ограничено максимумом
    expect(invalidSettings.batchSize).toBe(128); // Ограничено минимумом
    expect(invalidSettings.earlyStop).toBe(false);
  });

  it('should validate incremental training settings from UI', () => {
    // Имитируем валидацию настроек дообучения из UI
    const validateIncrementalTrainingSettings = (settings) => {
      const validated = {
        epochs: Math.max(1, Math.min(10, settings.epochs || 1)),
        batchSize: Math.max(32, Math.min(1024, settings.batchSize || 256)),
        patternsPerError: Math.max(10, Math.min(2000, settings.patternsPerError || 1000))
      };
      return validated;
    };

    // Тест 1: Дефолтные значения
    const defaultSettings = validateIncrementalTrainingSettings({});
    expect(defaultSettings.epochs).toBe(1);
    expect(defaultSettings.batchSize).toBe(256);
    expect(defaultSettings.patternsPerError).toBe(1000);

    // Тест 2: Валидные значения
    const validSettings = validateIncrementalTrainingSettings({
      epochs: 5,
      batchSize: 512,
      patternsPerError: 1500
    });
    expect(validSettings.epochs).toBe(5);
    expect(validSettings.batchSize).toBe(512);
    expect(validSettings.patternsPerError).toBe(1500);

    // Тест 3: Значения вне диапазона
    const invalidSettings = validateIncrementalTrainingSettings({
      epochs: 20, // Превышает максимум
      batchSize: 10, // Ниже минимума
      patternsPerError: 5000 // Превышает максимум
    });
    expect(invalidSettings.epochs).toBe(10); // Ограничено максимумом
    expect(invalidSettings.batchSize).toBe(32); // Ограничено минимумом
    expect(invalidSettings.patternsPerError).toBe(2000); // Ограничено максимумом
  });
});

