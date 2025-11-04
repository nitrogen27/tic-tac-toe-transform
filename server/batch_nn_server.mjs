import tfpkg from './src/tf.mjs';
const tf = tfpkg;

export async function startBatchServer({ loadModelFn, maxBatch=2048, tickMs=5 } = {}){
  const model = await loadModelFn();
  const q = [];
  let running = true;

  async function loop(){
    while(running){
      const start = Date.now();
      const batch = q.splice(0, maxBatch);
      if (batch.length){
        const xs = batch.map(r => r.xHW3);
        const ms = batch.map(r => r.maskFlat);
        const x = tf.concat(xs, 0);
        const m = tf.concat(ms, 0);
        try{
          const [logits, v] = model.apply(x);
          const masked = logits.add(m.mul(-1).add(1).mul(-1e9));
          const probs = tf.softmax(masked);
          const pa = await probs.array();
          const va = await v.array();
          for (let i=0;i<batch.length;i++) batch[i].resolve({ policy: pa[i], value: va[i][0] });
          logits.dispose(); v.dispose(); masked.dispose(); probs.dispose();
        }catch(e){
          batch.forEach(r => r.reject(e));
        } finally {
          x.dispose(); m.dispose();
          xs.forEach(t=>t.dispose()); ms.forEach(t=>t.dispose());
        }
      }
      const spent = Date.now()-start;
      await new Promise(r => setTimeout(r, Math.max(0, tickMs-spent)));
    }
  }
  loop();
  return { infer({ xHW3, maskFlat }){ return new Promise((resolve,reject)=> q.push({ resolve,reject,xHW3,maskFlat })); }, stop(){ running=false; } };
}
