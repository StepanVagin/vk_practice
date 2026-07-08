// Скрипт офлайн-обучения (dev-инструмент — НЕ входит в загружаемый в
// браузер бандл и не учитывается в лимите 10 МБ *кода*, см. README).
//
// Обучает небольшую полносвязную сеть (MLP), которая по статистике
// изображения предсказывает параметры автотона (яркость, контраст,
// насыщенность). Заменяет прежнюю ручную формулу clamp(...), которая
// раньше жила прямо в src/worker/enhance.worker.js: теперь параметры —
// выход настоящей нейросети с весами, полученными градиентным спуском
// на реальных фотографиях, а не захардкоженные константы.
//
// Формирование обучающих данных (self-supervised, без ручной разметки):
//   1. Берём 24 реальных эталонных фото из ./kodak24 (классический
//      тестовый набор Kodak — разнообразные, хорошо снятые фотографии).
//   2. Для каждого фото применяем ИЗВЕСТНУЮ синтетическую деградацию
//      яркости/контраста/насыщенности той же формулой преобразования,
//      что применяет само приложение при инференсе (см. analyzeAndEnhance()
//      в enhance.worker.js). Так имитируются "плохие" версии хороших фото.
//   3. Меткой для каждого испорченного варианта служит коррекция,
//      возвращающая его статистику к статистике ЭТОГО КОНКРЕТНОГО фото
//      до деградации — а не к одной придуманной универсальной константе.
//      Так сеть учится на реальном разнообразии (некоторые хорошие фото
//      естественно темнее/более приглушённые, более насыщенные и т.д.),
//      при этом зная (благодаря известной деградации), что значит
//      "ближе к источнику".
//   4. Сверху накладывается небольшой нелинейный штраф, чтобы сеть не
//      стремилась к полной коррекции, если на изображении уже есть
//      клиппинг светов/теней или сильный цветовой сдвиг (усиление
//      коррекции в этом случае только ухудшит результат) — имитирует
//      поведение аккуратного человека-ретушёра.
//
// Когда появится реальный набор эталонных изображений с экспертной
// оценкой, скрипт можно перезапустить, заменив синтетические метки
// шага 3 на выбор человека.

const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');
const tf = require('@tensorflow/tfjs');

const LUMA_WEIGHTS = [0.2126, 0.7152, 0.0722];
const KODAK_DIR = path.join(__dirname, 'kodak24');
const SAMPLE_STRIDE_TARGET = 96; // изображение уменьшается примерно до такого числа сэмплов на сторону

function loadPng(filePath) {
  const buf = fs.readFileSync(filePath);
  const png = PNG.sync.read(buf);
  return { width: png.width, height: png.height, data: png.data }; // RGBA Uint8Array
}

// Сводит декодированное изображение к плоскому списку [r,g,b] в 0..1,
// сэмплируя по сетке, чтобы стоимость обработки не зависела от исходного
// разрешения.
function toSampledPixels(img) {
  const stepX = Math.max(1, Math.floor(img.width / SAMPLE_STRIDE_TARGET));
  const stepY = Math.max(1, Math.floor(img.height / SAMPLE_STRIDE_TARGET));
  const pixels = [];
  for (let y = 0; y < img.height; y += stepY) {
    for (let x = 0; x < img.width; x += stepX) {
      const idx = (y * img.width + x) * 4;
      pixels.push([
        img.data[idx] / 255,
        img.data[idx + 1] / 255,
        img.data[idx + 2] / 255,
      ]);
    }
  }
  return pixels;
}

function luma([r, g, b]) {
  return r * LUMA_WEIGHTS[0] + g * LUMA_WEIGHTS[1] + b * LUMA_WEIGHTS[2];
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

// То же преобразование, что применяет приложение при инференсе
// (enhance.worker.js, analyzeAndEnhance): сдвиг яркости -> растяжение
// контраста вокруг 0.5 -> интерполяция насыщенности к люме -> clip в [0,1].
function applyTransform(pixels, brightnessDelta, contrastFactor, saturationFactor) {
  return pixels.map(([r, g, b]) => {
    let px = [r + brightnessDelta, g + brightnessDelta, b + brightnessDelta];
    px = px.map((c) => (c - 0.5) * contrastFactor + 0.5);
    const l = luma(px);
    px = px.map((c) => l + (c - l) * saturationFactor);
    return px.map((c) => clamp(c, 0, 1));
  });
}

function computeStats(pixels) {
  const n = pixels.length;
  let lumaSum = 0;
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let shadowCount = 0;
  let highlightCount = 0;
  const lumas = new Array(n);

  for (let i = 0; i < n; i++) {
    const [r, g, b] = pixels[i];
    const l = luma(pixels[i]);
    lumas[i] = l;
    lumaSum += l;
    rSum += r;
    gSum += g;
    bSum += b;
    if (l < 0.04) shadowCount++;
    if (l > 0.96) highlightCount++;
  }

  const lumaMean = lumaSum / n;
  let varSum = 0;
  let satSum = 0;
  for (let i = 0; i < n; i++) {
    const d = lumas[i] - lumaMean;
    varSum += d * d;
    const [r, g, b] = pixels[i];
    satSum += (Math.abs(r - lumas[i]) + Math.abs(g - lumas[i]) + Math.abs(b - lumas[i])) / 3;
  }
  const lumaStd = Math.sqrt(varSum / n);
  const satMean = satSum / n;

  const rMean = rSum / n;
  const gMean = gSum / n;
  const bMean = bSum / n;
  const colorCast = Math.max(rMean, gMean, bMean) - Math.min(rMean, gMean, bMean);

  return {
    lumaMean,
    lumaStd,
    satMean,
    shadowClip: shadowCount / n,
    highlightClip: highlightCount / n,
    colorCast,
  };
}

function statsToFeatures(s) {
  return [s.lumaMean, s.lumaStd, s.satMean, s.shadowClip, s.highlightClip, s.colorCast];
}

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function buildExamplesForImage(originalPixels, originalStats) {
  const examples = [];
  const VARIANTS_PER_IMAGE = 40;

  // Якорные примеры: без деградации вообще -> правильный ответ — ровно
  // "не трогать" (сдвиг яркости 0, коэффициенты 1). Повторяем несколько
  // раз, чтобы сеть не уходила от тождественного преобразования на уже
  // хорошо снятых фото просто потому, что в датасете преобладают
  // испорченные варианты.
  for (let i = 0; i < 6; i++) {
    examples.push({
      features: statsToFeatures(originalStats),
      targets: [0, 0, 0],
    });
  }

  for (let i = 0; i < VARIANTS_PER_IMAGE; i++) {
    // Смесь лёгкой (почти тождественной) и сильной деградации, чтобы
    // модель училась и случаю "не трогать", и случаю "исправить плохо
    // снятое фото".
    const severe = Math.random() < 0.7;
    const db = severe ? rand(-0.35, 0.35) : rand(-0.08, 0.08);
    const dc = severe ? rand(0.4, 1.8) : rand(0.85, 1.15);
    const ds = severe ? rand(0.3, 2.0) : rand(0.85, 1.15);

    const degradedPixels = applyTransform(originalPixels, db, dc, ds);
    const degradedStats = computeStats(degradedPixels);

    let brightness = clamp(originalStats.lumaMean - degradedStats.lumaMean, -0.15, 0.15);
    let contrast = clamp(originalStats.lumaStd / Math.max(degradedStats.lumaStd, 0.05), 0.85, 1.5);
    let saturation = clamp(originalStats.satMean / Math.max(degradedStats.satMean, 0.04), 0.9, 1.6);

    // Не стремимся к полной коррекции, если испорченное изображение уже
    // клиппится или имеет сильный цветовой сдвиг — усиление только
    // усугубит клиппинг/сдвиг.
    const clipPenalty = 1 - 0.5 * Math.min(1, degradedStats.highlightClip * 5 + degradedStats.shadowClip * 5);
    contrast = clamp(1 + (contrast - 1) * clipPenalty, 0.85, 1.5);

    const castPenalty = 1 - 0.7 * Math.min(1, degradedStats.colorCast * 6);
    saturation = clamp(1 + (saturation - 1) * castPenalty, 0.9, 1.6);

    if (degradedStats.highlightClip > 0.3) brightness -= 0.03;
    if (degradedStats.shadowClip > 0.3) brightness += 0.03;
    brightness = clamp(brightness, -0.15, 0.15);

    examples.push({
      features: statsToFeatures(degradedStats),
      targets: [brightness, contrast - 1, saturation - 1],
    });
  }
  return examples;
}

async function main() {
  const files = fs.readdirSync(KODAK_DIR).filter((f) => f.endsWith('.png')).sort();
  if (files.length === 0) {
    throw new Error(`No PNG files found in ${KODAK_DIR}`);
  }
  console.log(`Loading ${files.length} reference photos from ${KODAK_DIR}`);

  const allExamples = [];
  for (const file of files) {
    const img = loadPng(path.join(KODAK_DIR, file));
    const pixels = toSampledPixels(img);
    const originalStats = computeStats(pixels);
    const examples = buildExamplesForImage(pixels, originalStats);
    allExamples.push(...examples);
    console.log(
      `  ${file}: ${pixels.length} samples, lumaMean=${originalStats.lumaMean.toFixed(3)} ` +
      `lumaStd=${originalStats.lumaStd.toFixed(3)} satMean=${originalStats.satMean.toFixed(3)} ` +
      `-> ${examples.length} training examples`,
    );
  }

  console.log(`\nTotal training examples: ${allExamples.length}`);

  const xs = tf.tensor2d(allExamples.map((e) => e.features));
  const ys = tf.tensor2d(allExamples.map((e) => e.targets));

  const N_FEATURES = 6;
  const N_HIDDEN = 12;
  const N_OUTPUTS = 3;

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [N_FEATURES], units: N_HIDDEN, activation: 'relu' }));
  model.add(tf.layers.dense({ units: N_OUTPUTS, activation: 'tanh' }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

  await model.fit(xs, ys, {
    epochs: 250,
    batchSize: 64,
    shuffle: true,
    validationSplit: 0.15,
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 25 === 0 || epoch === 249) {
          console.log(`epoch ${epoch}: loss=${logs.loss.toFixed(5)} val_loss=${logs.val_loss.toFixed(5)}`);
        }
      },
    },
  });

  const [w1, b1, w2, b2] = model.getWeights();
  const weights = {
    inputFeatures: ['lumaMean', 'lumaStd', 'satMean', 'shadowClip', 'highlightClip', 'colorCast'],
    outputs: ['brightnessDelta_tanh', 'contrastFactorMinus1_tanh', 'saturationFactorMinus1_tanh'],
    w1: Array.from(w1.dataSync()),
    b1: Array.from(b1.dataSync()),
    w2: Array.from(w2.dataSync()),
    b2: Array.from(b2.dataSync()),
    shape: { input: N_FEATURES, hidden: N_HIDDEN, output: N_OUTPUTS },
    trainedOn: 'Kodak24 reference photos, self-supervised via synthetic degradation',
    trainedAt: new Date().toISOString(),
  };

  const testCases = [
    { name: 'dark underexposed', f: [0.15, 0.05, 0.03, 0.2, 0, 0.02] },
    { name: 'bright washed out', f: [0.85, 0.06, 0.04, 0, 0.25, 0.01] },
    { name: 'well exposed, leave alone', f: [0.48, 0.21, 0.11, 0, 0, 0.02] },
    { name: 'strong color cast', f: [0.5, 0.18, 0.05, 0, 0, 0.15] },
    { name: 'flat/hazy low contrast', f: [0.5, 0.06, 0.05, 0, 0, 0.02] },
  ];
  console.log('\nSanity check:');
  for (const tc of testCases) {
    const pred = model.predict(tf.tensor2d([tc.f])).dataSync();
    console.log(
      `  ${tc.name}: brightness=${pred[0].toFixed(3)} contrastFactor=${(1 + pred[1]).toFixed(3)} saturationFactor=${(1 + pred[2]).toFixed(3)}`,
    );
  }

  const outPath = path.join(__dirname, 'weights.json');
  fs.writeFileSync(outPath, JSON.stringify(weights));
  const paramCount = weights.w1.length + weights.b1.length + weights.w2.length + weights.b2.length;
  console.log(`\nWeights written to ${outPath}`);
  console.log(`Param count: ${paramCount}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
