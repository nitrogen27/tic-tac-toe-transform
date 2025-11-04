// Policy+Value ResNet для динамического N×N
// Маскирование нелегальных ходов, лосс CE + 0.5*MSE
import tfpkg from './tf.mjs';
const tf = tfpkg;

// ResNet блок
function resBlock(x, filters){
  const shortcut = x;
  let h = tf.layers.conv2d({filters, kernelSize:3, padding:'same', activation:'relu'}).apply(x);
  h = tf.layers.conv2d({filters, kernelSize:3, padding:'same'}).apply(h);
  h = tf.layers.add().apply([shortcut, h]);
  h = tf.layers.activation({activation:'relu'}).apply(h);
  return h;
}

// Policy+Value ResNet модель
export function buildPVModel({ inC=3, filters=64, blocks=8 } = {}){
  const inp = tf.input({shape:[null, null, inC]}); // [B,H,W,C] динамический размер
  
  // Начальный свёрточный слой
  let h = tf.layers.conv2d({filters, kernelSize:3, padding:'same', activation:'relu'}).apply(inp);
  
  // ResNet блоки
  for (let i=0; i<blocks; i++){
    h = resBlock(h, filters);
  }
  
  // Policy head: 1x1 conv -> flatten
  const policyConv = tf.layers.conv2d({filters:1, kernelSize:1, useBias:true}).apply(h);
  const policyLogits = tf.layers.flatten().apply(policyConv);
  
  // Value head: Global Average Pooling -> MLP -> tanh
  let value = tf.layers.globalAveragePooling2d().apply(h);
  value = tf.layers.dense({units: filters, activation:'relu'}).apply(value);
  value = tf.layers.dense({units:1, activation:'tanh'}).apply(value);
  
  return tf.model({inputs: inp, outputs: [policyLogits, value]});
}

// Маскирование нелегальных ходов
export function maskedLogits(logits, maskFlat){
  // maskFlat: [B, L] где 1 = легальный ход, 0 = нелегальный
  const penalty = maskFlat.mul(-1).add(1).mul(-1e9); // нелегальные -> -1e9
  return logits.add(penalty);
}

// Функция обучения (один шаг)
export function trainStep({ model, optimizer, xHW3, maskFlat, yPolicy, yValue }){
  return tf.tidy(() => {
    // Вычисляем loss и градиенты через optimizer.minimize
    const loss = optimizer.minimize(() => {
      const [logits, value] = model.apply(xHW3);
      
      // Маскирование policy logits
      const maskedLogs = maskedLogits(logits, maskFlat);
      const policyProbs = tf.softmax(maskedLogs);
      
      // Policy loss: categorical crossentropy
      const policyLoss = tf.losses.categoricalCrossentropy(yPolicy, policyProbs);
      
      // Value loss: MSE
      const valueLoss = tf.losses.meanSquaredError(yValue, value);
      
      // Total loss: CE + 0.5*MSE
      return policyLoss.add(valueLoss.mul(0.5));
    }, true);
    
    return loss;
  });
}
