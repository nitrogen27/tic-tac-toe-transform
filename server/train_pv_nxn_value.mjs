import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { buildPVModel, trainStep } from './src/model_pv_resnet_nxn.mjs';
import { batchGeneratorValue } from './src/dataset_gomoku10_value.mjs';
import { BOARD_N } from './src/config.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main(){
  const N = BOARD_N;
  const model = buildPVModel({ inC:3, filters: (N<=5?32:64), blocks: (N<=5?4:8) });
  const opt = tf.train.adam(1e-3);
  const gen = batchGeneratorValue({ N, batchSize: 512, steps: 1000 });
  let step=0;
  for await (const batch of gen){
    const loss = trainStep({ model, optimizer: opt, ...batch });
    if (step % 50 === 0){
      const v = await loss.data();
      console.log(`step ${step} total_loss=${v[0].toFixed(4)}`);
    }
    batch.xHW3.dispose(); batch.maskFlat.dispose(); batch.yPolicy.dispose(); batch.yValue.dispose(); loss.dispose();
    step++;
  }
  const saveDir = path.join(__dirname,'saved_pv','N'+N+'_value');
  await model.save('file://'+saveDir);
  console.log('Saved PV(+value) model to', saveDir);
}

main().catch(e=>{ console.error(e); process.exit(1); });
