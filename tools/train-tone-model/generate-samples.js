// Dev-инструмент: собирает пул эталонных изображений samples/ из реальных
// фото Kodak24 по категориям, уже описанным в samples/README.md
// (недоэкспонировано / переэкспонировано / малоконтрастно / малонасыщенно /
// уже хорошее). Не входит в загружаемый в браузер бандл.

const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');

const KODAK_DIR = path.join(__dirname, 'kodak24');
const SAMPLES_DIR = path.join(__dirname, '..', '..', 'samples');

function loadPng(filePath) {
  const buf = fs.readFileSync(filePath);
  return PNG.sync.read(buf);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

const LUMA_WEIGHTS = [0.2126, 0.7152, 0.0722];

// Та же формула преобразования, что в analyzeAndEnhance из enhance.worker.js,
// применённая здесь в обратную сторону (как *деградация*) к пикселям
// полного разрешения.
function degrade(png, brightnessDelta, contrastFactor, saturationFactor) {
  const out = new PNG({ width: png.width, height: png.height });
  const n = png.width * png.height;
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    let r = png.data[idx] / 255;
    let g = png.data[idx + 1] / 255;
    let b = png.data[idx + 2] / 255;

    r += brightnessDelta; g += brightnessDelta; b += brightnessDelta;
    r = (r - 0.5) * contrastFactor + 0.5;
    g = (g - 0.5) * contrastFactor + 0.5;
    b = (b - 0.5) * contrastFactor + 0.5;
    const l = r * LUMA_WEIGHTS[0] + g * LUMA_WEIGHTS[1] + b * LUMA_WEIGHTS[2];
    r = l + (r - l) * saturationFactor;
    g = l + (g - l) * saturationFactor;
    b = l + (b - l) * saturationFactor;

    out.data[idx] = Math.round(clamp(r, 0, 1) * 255);
    out.data[idx + 1] = Math.round(clamp(g, 0, 1) * 255);
    out.data[idx + 2] = Math.round(clamp(b, 0, 1) * 255);
    out.data[idx + 3] = png.data[idx + 3];
  }
  return out;
}

function save(png, name) {
  const outPath = path.join(SAMPLES_DIR, name);
  fs.writeFileSync(outPath, PNG.sync.write(png));
  console.log('wrote', outPath);
}

function main() {
  const plan = [
    { file: 'kodim05.png', name: '01-underexposed.png', degradation: [-0.28, 0.9, 0.95] },
    { file: 'kodim20.png', name: '02-overexposed.png', degradation: [0.3, 0.9, 1.0] },
    { file: 'kodim15.png', name: '03-low-contrast-hazy.png', degradation: [0.05, 0.45, 0.9] },
    { file: 'kodim02.png', name: '04-low-saturation-overcast.png', degradation: [0.0, 1.0, 0.35] },
    { file: 'kodim07.png', name: '05-already-good.png', degradation: [0, 1, 1] },
    { file: 'kodim19.png', name: '06-already-good.png', degradation: [0, 1, 1] },
  ];

  for (const item of plan) {
    const png = loadPng(path.join(KODAK_DIR, item.file));
    const [b, c, s] = item.degradation;
    const result = degrade(png, b, c, s);
    save(result, item.name);
  }
}

main();
