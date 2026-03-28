// PV Transformer для последовательности из 9 токенов (крестики-нолики 3×3)
// Вход: [B, 9, inDim=3] где inDim=3 (my/op/empty)
// Настоящий Multi-Head Attention с reshape на головы
import tfpkg from './tf.mjs';
const tf = tfpkg;
import { TRANSFORMER_CFG } from './config.mjs';

// ===== Кастомный слой: настоящий Multi-Head Self-Attention =====
// Проецирует Q,K,V → reshape на головы → scaled dot-product per head → concat
class MultiHeadAttentionLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.numHeads = config.numHeads;
    this.keyDim = config.keyDim;
    this.dModel = this.numHeads * this.keyDim;
  }

  build(inputShape) {
    // inputShape в functional API:
    //   single input: [null, seqLen, dModel] — inputShape[0] is null (not array)
    //   array input: [[null, seqLen, dModel]] — inputShape[0] is array
    let shape = inputShape;
    if (Array.isArray(shape[0])) {
      shape = shape[0]; // unwrap nested array
    }
    // shape = [null, seqLen, dModel]
    const inputDim = shape[shape.length - 1];

    // Проекции Q, K, V и выходная проекция — trainable weights
    this.wQ = this.addWeight('wQ', [inputDim, this.dModel], 'float32', tf.initializers.glorotUniform());
    this.wK = this.addWeight('wK', [inputDim, this.dModel], 'float32', tf.initializers.glorotUniform());
    this.wV = this.addWeight('wV', [inputDim, this.dModel], 'float32', tf.initializers.glorotUniform());
    this.wO = this.addWeight('wO', [this.dModel, this.dModel], 'float32', tf.initializers.glorotUniform());

    this.built = true;
  }

  call(inputs) {
    // НЕ используем tf.tidy() — он уничтожает промежуточные тензоры,
    // которые нужны для backward pass (вычисления градиентов).
    // Фреймворк сам управляет lifecycle тензоров при model.fit() и predict().
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    // x shape: [B, seqLen, dModel]

    const seqLen = x.shape[1];

    // Проекции: [B, seqLen, dModel] @ [dModel, dModel] → [B, seqLen, dModel]
    const wQ = this.wQ.read();
    const wK = this.wK.read();
    const wV = this.wV.read();
    const wO = this.wO.read();

    // Для 3D тензоров: reshape → matMul → reshape back
    const xFlat = x.reshape([-1, this.dModel]); // [B*seqLen, dModel]
    let q = tf.matMul(xFlat, wQ).reshape([-1, seqLen, this.dModel]);
    let k = tf.matMul(xFlat, wK).reshape([-1, seqLen, this.dModel]);
    let v = tf.matMul(xFlat, wV).reshape([-1, seqLen, this.dModel]);

    // Reshape to multi-head: [B, seqLen, numHeads, keyDim]
    q = q.reshape([-1, seqLen, this.numHeads, this.keyDim]);
    k = k.reshape([-1, seqLen, this.numHeads, this.keyDim]);
    v = v.reshape([-1, seqLen, this.numHeads, this.keyDim]);

    // Transpose to [B, numHeads, seqLen, keyDim]
    q = q.transpose([0, 2, 1, 3]);
    k = k.transpose([0, 2, 1, 3]);
    v = v.transpose([0, 2, 1, 3]);

    // Scaled dot-product attention per head
    const scale = 1.0 / Math.sqrt(this.keyDim);
    const scores = tf.matMul(q, k, false, true).mul(scale);
    const attn = tf.softmax(scores, -1);

    // Apply attention to values: [B, numHeads, seqLen, keyDim]
    let out = tf.matMul(attn, v);

    // Transpose back: [B, seqLen, numHeads, keyDim]
    out = out.transpose([0, 2, 1, 3]);

    // Reshape to [B, seqLen, dModel]
    out = out.reshape([-1, seqLen, this.dModel]);

    // Output projection: [B*seqLen, dModel] @ [dModel, dModel]
    const outFlat = out.reshape([-1, this.dModel]);
    const result = tf.matMul(outFlat, wO).reshape([-1, seqLen, this.dModel]);

    return result;
  }

  computeOutputShape(inputShape) {
    // single input: [null, seqLen, dModel]
    // array input: [[null, seqLen, dModel]]
    if (Array.isArray(inputShape[0])) {
      return inputShape[0]; // unwrap nested
    }
    return inputShape; // already the shape
  }

  getConfig() {
    const config = super.getConfig();
    config.numHeads = this.numHeads;
    config.keyDim = this.keyDim;
    return config;
  }

  static get className() {
    return 'MultiHeadAttentionLayer';
  }
}

// Регистрируем кастомный слой для десериализации
try {
  tf.serialization.registerClass(MultiHeadAttentionLayer);
} catch (e) {
  if (!e.message.includes('already registered')) {
    console.warn('[MHA] Could not register MultiHeadAttentionLayer:', e.message);
  }
}

// Multi-Head Self-Attention блок (обертка для функционального API)
function mhaBlock(x, dModel, numHeads, dropout, namePrefix = 'mha') {
  const keyDim = dModel / numHeads;

  const mha = new MultiHeadAttentionLayer({
    numHeads,
    keyDim,
    name: `${namePrefix}_mha`
  });

  let output = mha.apply(x);

  if (dropout > 0) {
    output = tf.layers.dropout({ rate: dropout, name: `${namePrefix}_drop` }).apply(output);
  }

  return output;
}

// Feed-Forward Network
function ffnBlock(x, dModel, dropout, namePrefix = 'ffn') {
  let h = tf.layers.dense({
    units: dModel * 4,
    activation: 'relu',
    kernelInitializer: 'glorotUniform',
    name: `${namePrefix}_expand`
  }).apply(x);
  if (dropout > 0) {
    h = tf.layers.dropout({ rate: dropout, name: `${namePrefix}_drop` }).apply(h);
  }
  h = tf.layers.dense({
    units: dModel,
    kernelInitializer: 'glorotUniform',
    name: `${namePrefix}_contract`
  }).apply(h);
  return h;
}

// Transformer блок: LN → MHA → residual → LN → FFN → residual (Pre-LN)
function transformerBlock(x, dModel, numHeads, dropout, name) {
  // LayerNorm перед MHA
  const ln1 = tf.layers.layerNormalization({ epsilon: 1e-6, name: `${name}_ln1` }).apply(x);
  const attn = mhaBlock(ln1, dModel, numHeads, dropout, `${name}_mha`);
  const x1 = tf.layers.add({ name: `${name}_add1` }).apply([x, attn]);

  // LayerNorm перед FFN
  const ln2 = tf.layers.layerNormalization({ epsilon: 1e-6, name: `${name}_ln2` }).apply(x1);
  const ffn = ffnBlock(ln2, dModel, dropout, `${name}_ffn`);
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

  // Token embedding: линейная проекция 3 → dModel
  const tokenEmbedding = tf.layers.dense({
    units: dModel,
    useBias: true,
    kernelInitializer: 'glorotUniform',
    name: 'token_embedding'
  }).apply(inp);

  // Позиционное кодирование: trainable embedding
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

  // Финальный LayerNorm
  x = tf.layers.layerNormalization({ epsilon: 1e-6, name: 'final_ln' }).apply(x);

  // ===== Policy head: per-token scoring (сохраняет позиционную информацию) =====
  // Dense(1) применяется к каждому токену → [B, 9, 1] → reshape → [B, 9]
  const policyPerToken = tf.layers.dense({
    units: 1,
    useBias: true,
    kernelInitializer: 'glorotUniform',
    name: 'policy_per_token'
  }).apply(x); // [B, 9, 1]

  const policyLogits = tf.layers.reshape({
    targetShape: [seqLen],
    name: 'policy_logits'
  }).apply(policyPerToken); // [B, 9]

  // ===== Value head: GlobalAveragePooling → MLP → tanh =====
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

  const model = tf.model({ inputs: [inp, posIdx], outputs: [policyLogits, value], name: 'pv_transformer_seq' });

  // Логируем архитектуру
  const totalParams = model.countParams();
  console.log(`[PVTransformer] Built model: dModel=${dModel}, layers=${numLayers}, heads=${heads}, params=${totalParams}`);

  return model;
}

// Вспомогательная функция для маскирования логитов
export function maskLogits(logits, mask) {
  // mask: [B, 9] где 1.0 = легальный ход, 0.0 = нелегальный
  const penalty = mask.mul(-1).add(1).mul(-1e9); // нелегальные → -1e9
  return logits.add(penalty);
}
