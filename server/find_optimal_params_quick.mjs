// Быстрый поиск оптимальных параметров (только ключевые конфигурации)
import { testTrainingConfig } from './tests/test_optimal_params.test.mjs';

const configs = [
  { epochs: 1, batchSize: 4096 },
  { epochs: 2, batchSize: 4096 },
  { epochs: 3, batchSize: 2048 },
  { epochs: 5, batchSize: 2048 },
];

async function main() {
  console.log('[OptimalParams] Starting QUICK parameter search...\n');
  const results = [];
  
  for (let i = 0; i < configs.length; i++) {
    const config = configs[i];
    console.log(`\n[OptimalParams] Testing config ${i + 1}/${configs.length}: epochs=${config.epochs}, batchSize=${config.batchSize}`);
    console.log('[OptimalParams] ========================================');
    
    try {
      const result = await testTrainingConfig(config);
      results.push(result);
      
      console.log(`[OptimalParams] ✓ Completed: quality=${result.evaluation.quality.toFixed(3)}, success=${result.evaluation.successRate.toFixed(3)}`);
    } catch (e) {
      console.error(`[OptimalParams] ✗ Error:`, e.message);
    }
  }
  
  // Сортируем по качеству
  results.sort((a, b) => b.evaluation.quality - a.evaluation.quality);
  
  console.log('\n\n[OptimalParams] ========================================');
  console.log('[OptimalParams] === OPTIMAL PARAMETERS SEARCH RESULTS ===');
  console.log('[OptimalParams] ========================================\n');
  
  results.forEach((r, i) => {
    const { wins, draws, losses, invalid } = r.evaluation;
    console.log(`${i + 1}. epochs=${r.config.epochs}, batchSize=${r.config.batchSize}`);
    console.log(`   Quality: ${r.evaluation.quality.toFixed(3)}`);
    console.log(`   Success: ${r.evaluation.successRate.toFixed(1)}% (wins: ${wins}, draws: ${draws}, losses: ${losses}, invalid: ${invalid})`);
    console.log(`   Final accuracy: ${r.finalAccuracy}%`);
    console.log('');
  });
  
  // Лучшая конфигурация
  const best = results[0];
  console.log('[OptimalParams] ========================================');
  console.log('[OptimalParams] === RECOMMENDED CONFIGURATION ===');
  console.log('[OptimalParams] ========================================');
  console.log(`epochs: ${best.config.epochs}`);
  console.log(`batchSize: ${best.config.batchSize}`);
  console.log(`Quality score: ${best.evaluation.quality.toFixed(3)}`);
  console.log(`Success rate: ${(best.evaluation.successRate * 100).toFixed(1)}%`);
  console.log(`Win rate: ${(best.evaluation.winRate * 100).toFixed(1)}%`);
  console.log(`Draw rate: ${(best.evaluation.drawRate * 100).toFixed(1)}%`);
  console.log(`Final training accuracy: ${best.finalAccuracy}%`);
  console.log('[OptimalParams] ========================================\n');
}

main().catch(console.error);

