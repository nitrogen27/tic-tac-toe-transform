import WebSocket from 'ws';
const ws = new WebSocket('ws://localhost:8080');
ws.on('open', () => {
  ws.send(JSON.stringify({ type: 'clear_model', payload: { variant: 'ttt5' } }));
  setTimeout(() => {
    ws.send(JSON.stringify({ type: 'train_ttt5', payload: { epochs: 5, batchSize: 64, earlyStop: true } }));
    console.log('Training started');
  }, 2000);
});
ws.on('message', (data) => {
  const msg = JSON.parse(data);
  if (msg.type === 'train.done') {
    console.log('TRAINING DONE');
    ws.close();
    process.exit(0);
  }
  if (msg.type === 'train.progress' && msg.payload) {
    const p = msg.payload;
    // Only log epoch-level summaries
    if (p.epoch !== undefined && p.totalEpochs && p.loss !== undefined && typeof p.loss === 'number') {
      console.log(`[${p.phase || '?'}] Epoch ${p.epoch}/${p.totalEpochs} Loss=${p.loss.toFixed(4)} Acc=${(p.accuracy||0).toFixed(1)}%`);
    }
    // Log game progress every 10 games
    if (p.game !== undefined && p.totalGames && p.game % 10 === 0) {
      console.log(`[${p.phase}] Game ${p.game}/${p.totalGames}`);
    }
  }
  if (msg.type === 'train.error') {
    console.error('TRAIN ERROR:', msg.payload);
    ws.close();
    process.exit(1);
  }
});
ws.on('error', (e) => { console.error('WS error:', e.message); process.exit(1); });
