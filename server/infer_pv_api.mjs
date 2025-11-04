import express from 'express';
import bodyParser from 'body-parser';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fs from 'fs/promises';
import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import { BOARD_N } from './src/config.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(bodyParser.json());

let model;
async function ensureModel(){
  if (!model){
    // Пробуем сначала PV модель, потом обычную
    const savedPvPath = path.join(__dirname, 'saved_pv', `N${BOARD_N}_value`, 'model.json');
    const savedPath = path.join(__dirname, 'saved', 'model.json');
    
    let url;
    try {
      await fs.access(savedPvPath);
      url = 'file://' + savedPvPath;
      console.log('[PV API] Found PV model, using:', url);
    } catch {
      url = 'file://' + savedPath;
      console.log('[PV API] Using standard model:', url);
    }
    
    try {
      model = await tf.loadLayersModel(url);
      console.log('[PV API] Model loaded successfully');
    } catch (e) {
      console.error('[PV API] Failed to load model:', e.message);
      throw new Error(`Model not found. Please train a model first. Expected at: ${url}`);
    }
  }
  return model;
}

app.post('/pv/infer', async (req,res)=>{
  try{
    const N = req.body.N || BOARD_N;
    const board = req.body.board; // length N*N, values {-1,0,1}
    const player = req.body.player || 1;
    const L = N*N;
    const xs = new Float32Array(1*L*3);
    const ms = new Float32Array(1*L);
    for (let i=0;i<L;i++){
      const base=i*3, v=board[i];
      xs[base+0]=(v===player)?1:0;
      xs[base+1]=(v!==0&&v!==player)?1:0;
      xs[base+2]=(v===0)?1:0;
      ms[i]=(v===0)?1:0;
    }
    const x = tf.tensor4d(xs,[1,N,N,3]);
    const m = tf.tensor2d(ms,[1,L]);
    const mdl = await ensureModel();
    const [logits, v] = mdl.apply(x);
    const masked = logits.add(m.mul(-1).add(1).mul(-1e9));
    const probs = tf.softmax(masked);
    const pa = Array.from((await probs.data()));
    const val = (await v.data())[0];
    x.dispose(); m.dispose(); logits.dispose(); masked.dispose(); probs.dispose(); v.dispose();
    res.json({ policy: pa, value: val });
  }catch(e){ console.error(e); res.status(500).json({ error: String(e) }); }
});

const port = process.env.PORT || 4001;
app.listen(port, ()=> console.log('PV infer API on :'+port));
