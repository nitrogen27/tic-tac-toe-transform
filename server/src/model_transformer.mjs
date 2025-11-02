import tfpkg from './tf.mjs';
const tf = tfpkg;

export function buildModel({
  dModel = 64,
  numLayers = 2,
  seqLen = 9,
  vocabSize = 3,
} = {}) {
  const cells = tf.input({ shape: [seqLen], dtype: 'int32', name: 'cells' });
  const posIdx = tf.input({ shape: [seqLen], dtype: 'int32', name: 'pos_idx' });

  const tokEmb = tf.layers.embedding({ inputDim: vocabSize, outputDim: dModel, name: 'tok_embedding' }).apply(cells);
  const posEmb = tf.layers.embedding({ inputDim: seqLen, outputDim: dModel, name: 'pos_embedding' }).apply(posIdx);
  let x = tf.layers.add({ name: 'sum_embed' }).apply([tokEmb, posEmb]);

  for (let i=0;i<numLayers;i++) {
    const n = tf.layers.layerNormalization({ epsilon:1e-6, name:`ln_${i+1}_pre` }).apply(x);
    const f1 = tf.layers.dense({ units: dModel*4, activation:'relu', name:`ff_${i+1}_in` }).apply(n);
    const f2 = tf.layers.dense({ units: dModel, name:`ff_${i+1}_out` }).apply(f1);
    x = tf.layers.add({ name:`res_${i+1}` }).apply([x, f2]);
  }
  x = tf.layers.layerNormalization({ epsilon:1e-6, name:'ln_final' }).apply(x);

  const logitsPerPos = tf.layers.dense({ units:1, name:'poslogits' }).apply(x);
  const logits = tf.layers.reshape({ targetShape:[seqLen], name:'reshape_logits'}).apply(logitsPerPos);
  const probs = tf.layers.activation({ activation:'softmax', name:'probs' }).apply(logits);

  const model = tf.model({ inputs:[cells, posIdx], outputs: probs, name:'t3_transformer' });
  // Оптимизация для M2: Adam optimizer с настройками для быстрой конвергенции
  const optimizer = tf.train.adam(1e-3, 0.9, 0.999, 1e-8);
  model.compile({ 
    optimizer: optimizer, 
    loss: 'categoricalCrossentropy', 
    metrics: ['accuracy']
  });
  return model;
}
