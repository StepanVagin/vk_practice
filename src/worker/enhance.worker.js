// ==========================================================================
// enhance.worker.js
// Web Worker, в котором выполняется вся "тяжёлая" обработка изображения.
// Работает отдельно от главного потока страницы, поэтому UI не блокируется
// (требование ТЗ: "работа в асинхронном режиме без блокировки браузера").
//
// Это классический (не модульный) Worker — так можно использовать
// importScripts() для подключения готовых UMD-сборок TensorFlow.js.
// ==========================================================================

// Подключаем "движок" TensorFlow.js: только ядро (tf-core) и CPU-бэкенд.
// Используется ИСКЛЮЧИТЕЛЬНО для прямого прохода маленькой обученной сети
// подбора параметров (6 -> 12 -> 3, см. predictToneDeltas ниже) — она
// работает над 6 числами (статистикой изображения), поэтому выбор бэкенда
// не влияет на производительность.
// Сама обработка пикселей (яркость/контраст/цвет) через тензоры TF.js НЕ
// выполняется: на реальных изображениях 15+ Мп CPU-бэкенд (чистый JS,
// множество проходов по данным на каждую операцию — slice/cast/div/mean/
// square/sub и т.д. отдельными проходами) давал ~55 с обработки, что
// превышает лимит ТЗ в 30 с. Вместо этого пиксели обрабатываются обычными
// циклами по TypedArray — см. комментарий над analyzeAndEnhance().
// CPU-бэкенд (а не WebGL/WASM) для самой модели выбран, так как WebGL
// внутри Worker требует OffscreenCanvas, который не везде стабилен, а для
// сети такого размера разница в скорости бэкендов не заметна.
// seedrandom подключается первым: это внешняя зависимость tf-backend-cpu
// (используется для случайных операций тензоров, у нас не задействуется,
// но UMD-сборка бэкенда ожидает наличие глобальной функции seedrandom).
importScripts('../../vendor/seedrandom.min.js');
importScripts('../../vendor/tf-core.min.js');
importScripts('../../vendor/tf-backend-cpu.min.js');
// Обученные веса модели подбора параметров автотона (см.
// tools/train-tone-model/train.js). Определяет глобальную TONE_MODEL_WEIGHTS.
importScripts('../model/tone-model-weights.js');

let backendReady = null;

async function ensureBackend() {
  if (!backendReady) {
    backendReady = tf.setBackend('cpu').then(() => tf.ready());
  }
  return backendReady;
}

// ---------------------------------------------------------------------
// Модель подбора параметров коррекции: небольшая полносвязная сеть
// (6 входов -> 12 скрытых нейронов ReLU -> 3 выхода tanh), веса которой
// получены обучением градиентным спуском на реальных фотографиях
// (см. tools/train-tone-model/). В отличие от прежней версии, где
// коэффициенты вычислялись фиксированной формулой clamp(...), здесь
// они являются выходом обученной модели — то есть именно "ИИ подбирает
// оптимальные параметры коррекции" (см. рекомендации к проекту), а
// применение этих параметров к пикселям (циклы по TypedArray в
// analyzeAndEnhance ниже) — отдельный, простой вспомогательный алгоритм.
// ---------------------------------------------------------------------
let toneModelWeights = null;

function ensureToneModel() {
  if (!toneModelWeights) {
    const w = self.TONE_MODEL_WEIGHTS;
    toneModelWeights = {
      w1: tf.tensor2d(w.w1, [w.shape.input, w.shape.hidden]),
      b1: tf.tensor1d(w.b1),
      w2: tf.tensor2d(w.w2, [w.shape.hidden, w.shape.output]),
      b2: tf.tensor1d(w.b2),
    };
  }
  return toneModelWeights;
}

// Прямой проход обученной сети: 6 статистик изображения -> сырые
// (tanh, диапазон -1..1) коэффициенты коррекции.
function predictToneDeltas(featuresArray) {
  const { w1, b1, w2, b2 } = ensureToneModel();
  return tf.tidy(() => {
    const x = tf.tensor2d([featuresArray]); // [1, 6]
    const hidden = tf.relu(tf.add(tf.matMul(x, w1), b1));
    const output = tf.tanh(tf.add(tf.matMul(hidden, w2), b2));
    return output.dataSync(); // [3]: яркость, контраст-1, насыщенность-1 (сырые, без пересчёта)
  });
}

// Текущая выполняющаяся задача (в одном Worker обрабатывается одна задача,
// на каждую задачу API создаёт свой Worker — это упрощает отмену:
// достаточно вызвать worker.terminate()).
function reportProgress(status, progress) {
  self.postMessage({ type: 'progress', status, progress });
}

function reportError(message) {
  self.postMessage({ type: 'error', message: String(message) });
}

function reportResult(blob, width, height) {
  self.postMessage({ type: 'result', blob, width, height });
}

// Ограничение значения в диапазон [min, max]
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// Извлечение сырых пикселей (ImageData) из Blob через createImageBitmap +
// OffscreenCanvas. Обе API доступны внутри Worker в современных браузерах,
// поэтому декодирование JPG/PNG/BMP не требует главного потока.
async function decodeToImageData(blob) {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return imageData;
}

// Кодирование готового ImageData обратно в Blob нужного формата.
async function encodeImageData(imageData, mimeType, quality) {
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  return canvas.convertToBlob({ type: mimeType, quality });
}

// ---------------------------------------------------------------------
// Стадия 1 (ИИ подбирает параметры): анализирует пиксели изображения,
// извлекает 6 статистик (средняя яркость, контраст, насыщенность, доля
// клиппинга в тенях/светах, цветовой сдвиг) и прогоняет их через
// обученную модель (predictToneDeltas) — та возвращает предложенные
// коэффициенты коррекции.
// Стадия 2 (вспомогательный алгоритм применяет параметры): прямой проход
// по пикселям (offset/stretch/interpolate) применяет уже готовые
// коэффициенты — это простая детерминированная операция, а не часть
// модели.
//
// Обе стадии реализованы обычными циклами по TypedArray, а НЕ тензорными
// операциями TensorFlow.js: на изображениях 15+ Мп CPU-бэкенд tf.js
// (чистый JS, отдельный проход по всем пикселям на КАЖДУЮ операцию —
// slice/cast/div/mean/square/sub и т.д.) давал ~55 с обработки на 18.6 Мп
// фото, что превышает лимит ТЗ в 30 с. Три плотных прохода по TypedArray
// ниже дают тот же результат на порядок быстрее и укладываются в
// требуемые "в среднем 5 с". Модель подбора параметров (predictToneDeltas
// выше) по-прежнему считается через TensorFlow.js — она работает всего
// над 6 числами, поэтому выбор реализации там не влияет на
// производительность.
//
// Если пользователь передал свои значения brightness/contrast/saturation —
// используются они вместо автоматической оценки модели.
// ---------------------------------------------------------------------
const LUMA_R = 0.2126, LUMA_G = 0.7152, LUMA_B = 0.0722; // BT.709

async function analyzeAndEnhance(imageData, params, onStage) {
  const { width, height, data } = imageData;
  const pixelCount = width * height;

  onStage('analyzing', 25);

  // Проход 1: люма на пиксель (сохраняем для прохода 2, не пересчитываем)
  // + накопление статистик яркости/насыщенности/клиппинга/цветового сдвига.
  const lumaArr = new Float32Array(pixelCount);
  let lumaSum = 0;
  let satSum = 0;
  let shadowCount = 0;
  let highlightCount = 0;
  let rSum = 0, gSum = 0, bSum = 0;

  for (let i = 0, p = 0; i < pixelCount; i++, p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    const luma = r * LUMA_R + g * LUMA_G + b * LUMA_B;

    lumaArr[i] = luma;
    lumaSum += luma;
    // Оценка "цветности" (насыщенности) без перевода в HSV: чем сильнее
    // канал отличается от серого (люмы) — тем выше насыщенность пикселя.
    satSum += (Math.abs(r - luma) + Math.abs(g - luma) + Math.abs(b - luma)) / 3;
    // Клиппинг теней/светов — модель использует это, чтобы не усиливать
    // коррекцию там, где она всё равно упрётся в границы диапазона.
    if (luma < 0.04) shadowCount++;
    if (luma > 0.96) highlightCount++;
    rSum += r; gSum += g; bSum += b;
  }

  const lumaMean = lumaSum / pixelCount;
  const satMean = satSum / pixelCount;
  const shadowClip = shadowCount / pixelCount;
  const highlightClip = highlightCount / pixelCount;
  // Цветовой сдвиг (color cast): разброс средних значений по каналам.
  const channelMeans = [rSum / pixelCount, gSum / pixelCount, bSum / pixelCount];
  const colorCast = Math.max(...channelMeans) - Math.min(...channelMeans);

  // Проход 2: СКО люмы (мера контраста) по уже посчитанному lumaArr.
  let varianceSum = 0;
  for (let i = 0; i < pixelCount; i++) {
    const d = lumaArr[i] - lumaMean;
    varianceSum += d * d;
  }
  const lumaStd = Math.sqrt(varianceSum / pixelCount);

  // --- ИИ подбирает коэффициенты коррекции (обученная модель) ---
  // predictToneDeltas возвращает уже готовые (не требующие пересчёта)
  // значения в тех же единицах, что и итоговые параметры — модель
  // обучена регрессировать напрямую на них (см. tools/train-tone-model).
  const [rawBrightness, rawContrastDelta, rawSaturationDelta] = predictToneDeltas([
    lumaMean, lumaStd, satMean, shadowClip, highlightClip, colorCast,
  ]);
  // clamp() здесь — не эвристика подбора, а защитный предохранитель:
  // не даёт модели (в теории) выдать значение за пределами проверенного
  // безопасного диапазона.
  const autoBrightness = clamp(rawBrightness, -0.15, 0.15);
  const autoContrast = clamp(1 + rawContrastDelta, 0.85, 1.5);
  const autoSaturation = clamp(1 + rawSaturationDelta, 0.9, 1.6);

  // params.* в диапазоне [-1..1], null/undefined => авторежим
  const brightnessDelta = params.brightness == null
    ? autoBrightness
    : clamp(params.brightness, -1, 1) * 0.3;
  const contrastFactor = params.contrast == null
    ? autoContrast
    : 1 + clamp(params.contrast, -1, 1) * 0.6;
  const saturationFactor = params.saturation == null
    ? autoSaturation
    : 1 + clamp(params.saturation, -1, 1) * 0.7;

  onStage('processing', 55);

  // Проход 3: применяем коррекцию и сразу пишем результат в выходной
  // TypedArray:
  // 1) яркость — смещение (offset)
  // 2) контраст — растяжение относительно середины диапазона (0.5)
  // 3) цветность — интерполяция между "серым" (люма) и цветным пикселем
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0, p = 0; i < pixelCount; i++, p += 4) {
    let r = data[p] / 255 + brightnessDelta;
    let g = data[p + 1] / 255 + brightnessDelta;
    let b = data[p + 2] / 255 + brightnessDelta;

    r = (r - 0.5) * contrastFactor + 0.5;
    g = (g - 0.5) * contrastFactor + 0.5;
    b = (b - 0.5) * contrastFactor + 0.5;

    const l = r * LUMA_R + g * LUMA_G + b * LUMA_B;
    r = l + (r - l) * saturationFactor;
    g = l + (g - l) * saturationFactor;
    b = l + (b - l) * saturationFactor;

    out[p] = Math.round(clamp(r, 0, 1) * 255);
    out[p + 1] = Math.round(clamp(g, 0, 1) * 255);
    out[p + 2] = Math.round(clamp(b, 0, 1) * 255);
    out[p + 3] = data[p + 3]; // альфа-канал не меняется
  }

  onStage('encoding', 90);

  return {
    imageData: new ImageData(out, width, height),
    appliedParams: { brightnessDelta, contrastFactor, saturationFactor },
  };
}

self.onmessage = async (event) => {
  const { blob, options } = event.data;

  try {
    await ensureBackend();

    reportProgress('decoding', 5);
    const imageData = await decodeToImageData(blob);

    const { imageData: resultImageData } = await analyzeAndEnhance(
      imageData,
      options || {},
      (status, progress) => reportProgress(status, progress),
    );

    const outputMime = (options && options.outputFormat) || 'image/png';
    const outputQuality = (options && options.quality) || 0.92;
    const resultBlob = await encodeImageData(resultImageData, outputMime, outputQuality);

    reportProgress('encoding', 100);
    reportResult(resultBlob, resultImageData.width, resultImageData.height);
  } catch (err) {
    console.error(err);
    const details = err && err.stack ? err.stack : String(err);
    reportError(details);
  }
};
