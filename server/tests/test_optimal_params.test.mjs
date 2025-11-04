// Тесты для подбора оптимальных параметров обучения
import { describe, it, expect } from 'vitest';
import { trainTTT3WithProgress } from '../src/train_ttt3_transformer_service.mjs';
import { predictMove, clearModel, reloadTTT3Model } from '../service.mjs';
import { teacherBestMove, legalMoves, getWinner, emptyBoard } from '../src/tic_tac_toe.mjs';

// Играет одну игру и возвращает результат
async function playSingleGame() {
  const board = emptyBoard();
  let current = 1;
  let moves = 0;
  
  while (getWinner(board) === null && moves < 20) {
    const legal = legalMoves(board);
    if (legal.length === 0) break;
    
    let move;
    if (current === 1) {
      const result = await predictMove({ board: [...board], current: 1, mode: 'model' });
      move = result.move;
      if (move < 0 || move >= 9 || board[move] !== 0) {
        return { winner: 2, reason: 'invalid' };
      }
    } else {
      move = teacherBestMove(board, 2);
    }
    
    board[move] = current;
    current = current === 1 ? 2 : 1;
    moves++;
  }
  
  return { winner: getWinner(board) || 0, moves };
}

// Оценивает качество модели на N играх
async function evaluateModel(nGames = 10) {
  let wins = 0, draws = 0, losses = 0, invalid = 0;
  
  for (let i = 0; i < nGames; i++) {
    const result = await playSingleGame();
    if (result.winner === 1) wins++;
    else if (result.winner === 2) {
      if (result.reason === 'invalid') invalid++;
      else losses++;
    }
    else draws++;
  }
  
  return {
    wins,
    draws,
    losses,
    invalid,
    winRate: wins / nGames,
    drawRate: draws / nGames,
    lossRate: losses / nGames,
    successRate: (wins + draws) / nGames,
    quality: (wins + draws - invalid * 2) / nGames // Штраф за недопустимые ходы
  };
}

// Тестирует конфигурацию обучения
async function testTrainingConfig(config) {
  console.log(`[Test] Testing config: epochs=${config.epochs}, batchSize=${config.batchSize}`);
  
  // Очищаем модель перед обучением
  await clearModel();
  reloadTTT3Model();
  
  // Обучаем модель
  const trainingEvents = [];
  await trainTTT3WithProgress(
    (ev) => {
      trainingEvents.push(ev);
      if (ev.type === 'train.progress') {
        console.log(`[Test] Training progress: epoch ${ev.payload.epoch}/${ev.payload.epochs}, loss=${ev.payload.loss}, acc=${ev.payload.acc}`);
      }
    },
    config
  );
  
  // Ждем немного, чтобы модель загрузилась
  await new Promise(r => setTimeout(r, 1000));
  
  // Оцениваем качество
  const evaluation = await evaluateModel(10);
  console.log(`[Test] Evaluation results:`, evaluation);
  
  return {
    config,
    evaluation,
    finalAccuracy: trainingEvents
      .filter(e => e.type === 'train.progress')
      .map(e => parseFloat(e.payload.acc || 0))
      .pop() || 0
  };
}

describe('Optimal Parameters Search', () => {
  const configs = [
    // Быстрые конфигурации (1-2 эпохи)
    { epochs: 1, batchSize: 4096 },
    { epochs: 2, batchSize: 4096 },
    { epochs: 2, batchSize: 2048 },
    
    // Средние конфигурации (3-5 эпох)
    { epochs: 3, batchSize: 2048 },
    { epochs: 5, batchSize: 2048 },
    
    // Долгие конфигурации (больше эпох)
    { epochs: 5, batchSize: 1024 },
    { epochs: 10, batchSize: 1024 },
  ];
  
  const results = [];
  
  // Пропускаем тесты по умолчанию, чтобы не запускать их автоматически
  // Они будут запущены вручную через отдельный скрипт
  
  it.skip('should find optimal parameters', async () => {
    for (const config of configs) {
      try {
        const result = await testTrainingConfig(config);
        results.push(result);
        
        // Сохраняем промежуточные результаты
        console.log(`[Test] Config ${JSON.stringify(config)}: quality=${result.evaluation.quality.toFixed(3)}, successRate=${result.evaluation.successRate.toFixed(3)}`);
      } catch (e) {
        console.error(`[Test] Error testing config ${JSON.stringify(config)}:`, e);
      }
    }
    
    // Сортируем по качеству
    results.sort((a, b) => b.evaluation.quality - a.evaluation.quality);
    
    console.log('\n[Test] === OPTIMAL PARAMETERS SEARCH RESULTS ===');
    results.forEach((r, i) => {
      console.log(`${i + 1}. epochs=${r.config.epochs}, batchSize=${r.config.batchSize}: quality=${r.evaluation.quality.toFixed(3)}, success=${r.evaluation.successRate.toFixed(3)}, wins=${r.evaluation.wins}/10`);
    });
    
    // Лучшая конфигурация
    const best = results[0];
    console.log(`\n[Test] BEST CONFIG: epochs=${best.config.epochs}, batchSize=${best.config.batchSize}`);
    console.log(`[Test] Quality: ${best.evaluation.quality.toFixed(3)}`);
    console.log(`[Test] Success rate: ${best.evaluation.successRate.toFixed(3)}`);
    console.log(`[Test] Win rate: ${best.evaluation.winRate.toFixed(3)}`);
    console.log(`[Test] Draw rate: ${best.evaluation.drawRate.toFixed(3)}`);
    console.log(`[Test] Final accuracy: ${best.finalAccuracy}%`);
    
    expect(results.length).toBeGreaterThan(0);
  }, 1800000); // 30 минут на все тесты
});

// Экспортируем функции для использования в отдельном скрипте
export { testTrainingConfig, evaluateModel, playSingleGame };

