// Тест для проверки одной конфигурации
import { describe, it, expect } from 'vitest';
import { testTrainingConfig } from './test_optimal_params.test.mjs';

describe('Single Configuration Test', () => {
  it('should test epochs=5 batchSize=2048', async () => {
    const config = { epochs: 5, batchSize: 2048 };
    const result = await testTrainingConfig(config);
    
    console.log('\n[Test] ========================================');
    console.log('[Test] Configuration:', config);
    console.log('[Test] Quality score:', result.evaluation.quality.toFixed(3));
    console.log('[Test] Success rate:', (result.evaluation.successRate * 100).toFixed(1) + '%');
    console.log('[Test] Win rate:', (result.evaluation.winRate * 100).toFixed(1) + '%');
    console.log('[Test] Draw rate:', (result.evaluation.drawRate * 100).toFixed(1) + '%');
    console.log('[Test] Final accuracy:', result.finalAccuracy + '%');
    console.log('[Test] ========================================\n');
    
    // Проверяем, что модель играет без ошибок
    expect(result.evaluation.invalid).toBe(0);
    expect(result.evaluation.successRate).toBeGreaterThan(0.5); // Хотя бы 50% успешных игр
  }, 600000); // 10 минут
});

