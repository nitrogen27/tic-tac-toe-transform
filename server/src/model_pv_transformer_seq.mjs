// PV Transformer для последовательности из 9 токенов (крестики-нолики 3×3)
// Вход: [B, 9, inDim=3] где inDim=3 (my/op/empty)
import tfpkg from './tf.mjs';
const tf = tfpkg;
import { TRANSFORMER_CFG } from './config.mjs';

// Синусоидальное позиционное кодирование для 9 позиций
function sinusoidalPositionalEmbedding(seqLen, dModel) {
  const embeddings = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const emb = new Array(dModel).fill(0);
    for (let i = 0; i < dModel; i += 2) {
      const div = Math.pow(10000, (2 * i) / dModel);
      emb[i] = Math.sin(pos / div);
      if (i + 1 < dModel) {
        emb[i + 1] = Math.cos(pos / div);
      }
    }
    embeddings.push(emb);
  }
  return tf.tensor2d(embeddings, [seqLen, dModel]);
}

// Multi-Head Self-Attention блок
// Реализация собственного MHA, так как tf.layers.multiHeadAttention недоступен в tfjs-node
function mhaBlock(x, dModel, numHeads, dropout, namePrefix = 'mha') {
  const keyDim = dModel / numHeads;
  
  // Проекции для Q, K, V с уникальными именами
  const qProj = tf.layers.dense({ units: dModel, useBias: false, name: `${namePrefix}_q` });
  const kProj = tf.layers.dense({ units: dModel, useBias: false, name: `${namePrefix}_k` });
  const vProj = tf.layers.dense({ units: dModel, useBias: false, name: `${namePrefix}_v` });
  
  // Применяем проекции
  const q = qProj.apply(x);
  const k = kProj.apply(x);
  const v = vProj.apply(x);
  
  // Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
  // Используем прямые тензорные операции вместо lambda (который недоступен в tfjs-node)
  
  // Создаем кастомный слой для attention
  class ScaledDotProductAttention extends tf.layers.Layer {
    constructor(config) {
      super(config);
      this.scale = config.scale;
    }
    
    call(inputs) {
      return tf.tidy(() => {
        const [q, k, v] = inputs;
        // Q @ K^T
        const scores = tf.matMul(q, k, false, true);
        // Scale
        const scaled = scores.mul(this.scale);
        // Softmax
        const attn = tf.softmax(scaled, -1);
        // Apply to V
        const output = tf.matMul(attn, v);
        return output;
      });
    }
    
    computeOutputShape(inputShapes) {
      return inputShapes[0]; // [B, seqLen, dModel]
    }
  }
  
  const scale = 1.0 / Math.sqrt(keyDim);
  const attentionLayer = new ScaledDotProductAttention({ scale, name: `${namePrefix}_attention` });
  const attnOutput = attentionLayer.apply([q, k, v]);
  
  // Output projection
  const outProj = tf.layers.dense({ units: dModel, useBias: false, name: `${namePrefix}_out` });
  const output = outProj.apply(attnOutput);
  
  return output;
}

// Feed-Forward Network
function ffnBlock(x, dModel, dropout) {
  let h = tf.layers.dense({
    units: dModel * 4,
    activation: 'relu',
    kernelInitializer: 'glorotUniform'
  }).apply(x);
  if (dropout > 0) {
    h = tf.layers.dropout({ rate: dropout }).apply(h);
  }
  h = tf.layers.dense({
    units: dModel,
    kernelInitializer: 'glorotUniform'
  }).apply(h);
  return h;
}

// Transformer блок: LN → MHA → residual → LN → FFN → residual
function transformerBlock(x, dModel, numHeads, dropout, name) {
  // LayerNorm перед MHA
  const ln1 = tf.layers.layerNormalization({ epsilon: 1e-6, name: `${name}_ln1` }).apply(x);
  const attn = mhaBlock(ln1, dModel, numHeads, dropout, `${name}_mha`);
  const x1 = tf.layers.add({ name: `${name}_add1` }).apply([x, attn]);
  
  // LayerNorm перед FFN
  const ln2 = tf.layers.layerNormalization({ epsilon: 1e-6, name: `${name}_ln2` }).apply(x1);
  const ffn = ffnBlock(ln2, dModel, dropout);
  const x2 = tf.layers.add({ name: `${name}_add2` }).apply([x1, ffn]);
  
  return x2;
}

// Построение PV Transformer модели
export function buildPVTransformerSeq({ 
  dModel = TRANSFORMER_CFG.dModel, 
  numLayers = TRANSFORMER_CFG.numLayers, 
  heads = TRANSFORMER_CFG.heads,
  dropout = TRANSFORMER_CFG.dropout,
  seqLen = 9,
  inDim = 3
} = {}) {
  // Вход: [B, 9, 3]
  const inp = tf.input({ shape: [seqLen, inDim], name: 'input' });
  
  // Embedding: линейный до dModel
  const tokenEmbedding = tf.layers.dense({
    units: dModel,
    useBias: true,
    kernelInitializer: 'glorotUniform',
    name: 'token_embedding'
  }).apply(inp);
  
  // Позиционное кодирование: используем trainable embedding
  // Создаем второй вход для позиций (0..8)
  const posIdx = tf.input({ shape: [seqLen], dtype: 'int32', name: 'pos_idx' });
  const posEmbedding = tf.layers.embedding({
    inputDim: seqLen,
    outputDim: dModel,
    name: 'positional_embedding'
  }).apply(posIdx);
  
  // Складываем token и positional embeddings
  let x = tf.layers.add({ name: 'add_embeddings' }).apply([tokenEmbedding, posEmbedding]);
  
  // Transformer блоки
  for (let i = 0; i < numLayers; i++) {
    x = transformerBlock(x, dModel, heads, dropout, `transformer_${i}`);
  }
  
  // Policy head: применяем dense к каждому токену и суммируем или берем глобальный pooling
  // Используем GlobalMaxPooling1d для получения агрегированного представления
  const policyPool = tf.layers.globalMaxPooling1d({ name: 'policy_pool' }).apply(x);
  const policyLogits = tf.layers.dense({
    units: seqLen, // 9
    useBias: true,
    kernelInitializer: 'glorotUniform',
    name: 'policy_logits'
  }).apply(policyPool);
  
  // Value head: GlobalAveragePooling по токенам → MLP → tanh
  let value = tf.layers.globalAveragePooling1d({ name: 'value_pool' }).apply(x);
  value = tf.layers.dense({
    units: dModel,
    activation: 'relu',
    kernelInitializer: 'glorotUniform',
    name: 'value_mlp1'
  }).apply(value);
  value = tf.layers.dense({
    units: 1,
    activation: 'tanh',
    kernelInitializer: 'glorotUniform',
    name: 'value_output'
  }).apply(value);
  
  // Модель возвращает [policyLogits, value]
  // Маскирование нелегальных ходов будет применено снаружи
  return tf.model({ inputs: [inp, posIdx], outputs: [policyLogits, value], name: 'pv_transformer_seq' });
}

// Вспомогательная функция для маскирования логитов
export function maskLogits(logits, mask) {
  // mask: [B, 9] где 1.0 = легальный ход, 0.0 = нелегальный
  const penalty = mask.mul(-1).add(1).mul(-1e9); // нелегальные → -1e9
  return logits.add(penalty);
}
