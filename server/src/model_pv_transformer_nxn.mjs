import tfpkg from './tf.mjs';
const tf = tfpkg;

// Simple Transformer block
// Реализация собственного MHA, так как tf.layers.multiHeadAttention недоступен в tfjs-node
function mha(x, dModel, numHeads){
  const keyDim = dModel / numHeads;
  
  // Проекции для Q, K, V
  const qProj = tf.layers.dense({ units: dModel, useBias: false });
  const kProj = tf.layers.dense({ units: dModel, useBias: false });
  const vProj = tf.layers.dense({ units: dModel, useBias: false });
  
  const q = qProj.apply(x);
  const k = kProj.apply(x);
  const v = vProj.apply(x);
  
  // Scaled dot-product attention - используем кастомный слой
  class SimpleAttention extends tf.layers.Layer {
    constructor(config) {
      super(config);
      this.scale = config.scale;
    }
    
    call(inputs) {
      return tf.tidy(() => {
        const [q, k, v] = inputs;
        const scores = tf.matMul(q, k, false, true);
        const scaled = scores.mul(this.scale);
        const attn = tf.softmax(scaled, -1);
        return tf.matMul(attn, v);
      });
    }
    
    computeOutputShape(inputShapes) {
      return inputShapes[0];
    }
  }
  
  const scale = 1.0 / Math.sqrt(keyDim);
  const attentionLayer = new SimpleAttention({ scale });
  const out = attentionLayer.apply([q, k, v]);
  
  const outProj = tf.layers.dense({ units: dModel, useBias: false });
  return outProj.apply(out);
}
function ffn(x, dModel){
  let h = tf.layers.dense({units: dModel*4, activation:'relu'}).apply(x);
  h = tf.layers.dense({units: dModel}).apply(h);
  return h;
}
function addNorm(x, y){
  const add = tf.layers.add().apply([x,y]);
  return tf.layers.layerNormalization({epsilon:1e-5}).apply(add);
}

// 2D sinusoidal embeddings for positions
function pos2dEmbeddingHW(H, W, dModel){
  // produce [H*W, dModel]
  const L = H*W;
  const out = [];
  const half = dModel>>1;
  for (let r=0;r<H;r++){
    for (let c=0;c<W;c++){
      const v = new Array(dModel).fill(0);
      for (let i=0;i<half;i++){
        const div = Math.pow(10000, (2*i)/dModel);
        v[2*i]   = Math.sin(r/div);
        v[2*i+1] = Math.cos(c/div);
      }
      out.push(v);
    }
  }
  return tf.tensor2d(out, [L, dModel]);
}

// Build PV Transformer for dynamic N×N: we accept dynamic H,W using conv1x1 + flatten tokens
export function buildPVTransformer({ inC=3, dModel=128, numLayers=4, numHeads=4 }){
  const inp = tf.input({shape:[null,null,inC]}); // [B,H,W,C]
  // 1x1 conv to dModel channels -> [B,H,W,dModel]
  let h = tf.layers.conv2d({filters:dModel, kernelSize:1, padding:'valid'}).apply(inp);
  const shape = h.shape; // [null,H,W,dModel]; H,W may be dynamic at graph build time in tfjs, but we can still reshape flatten spatial
  // Flatten spatial to tokens
  const toTokens = tf.layers.reshape({targetShape:[-1, dModel]}).apply(h); // [B*H*W?, dModel] not supported. So do merge dims: use tf.layers.permute? tfjs restricts - use pooling
  // Workaround: we keep conv-only transformer (depthwise conv as proxy) due to tfjs token reshape issues.
  // Fallback to a "convformer-lite": stacks of (1x1 + depthwise 3x3 + 1x1) mimicking attention receptive field.
  // If you want true MHSA, better train/export from PyTorch->ONNX->tfjs or freeze HW at compile time.

  // Convformer-lite blocks
  function convformer(x){
    // DW conv (token mixing)
    let y = tf.layers.depthwiseConv2d({kernelSize:3, padding:'same'}).apply(x);
    y = tf.layers.activation({activation:'relu'}).apply(y);
    y = tf.layers.conv2d({filters:dModel, kernelSize:1}).apply(y);
    return y;
  }
  for (let i=0;i<numLayers;i++){
    const y = convformer(h);
    h = tf.layers.add().apply([h,y]);
    h = tf.layers.layerNormalization({epsilon:1e-5}).apply(h);
  }

  // Heads
  // Policy: 1x1 -> 1, flatten
  const p = tf.layers.conv2d({filters:1, kernelSize:1, useBias:true}).apply(h);
  const policyLogits = tf.layers.flatten().apply(p);
  // Value: GAP + MLP + tanh
  let v = tf.layers.globalAveragePooling2d().apply(h);
  v = tf.layers.dense({units: dModel, activation:'relu'}).apply(v);
  const value = tf.layers.dense({units:1, activation:'tanh'}).apply(v);

  return tf.model({inputs: inp, outputs: [policyLogits, value]});
}

export function maskedLogits(logits, maskFlat){
  const penalty = maskFlat.mul(-1).add(1).mul(-1e9);
  return logits.add(penalty);
}
