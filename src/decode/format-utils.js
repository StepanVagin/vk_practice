// ==========================================================================
// format-utils.js
// Определение формата исходного файла и приведение HEIC/HEIF к формату,
// который умеет декодировать сам браузер (JPG/PNG/BMP декодируются им
// нативно через createImageBitmap — см. worker/enhance.worker.js).
//
// Почему конвертация HEIC выполняется в главном потоке, а не в Worker:
// библиотека heic2any (обёртка над libheif, скомпилированным в WASM)
// внутри себя обращается к document.createElement(), поэтому не может
// быть безопасно запущена в Worker (там нет DOM). Сама конвертация
// контейнера HEIC — быстрая операция (доли секунды — единицы секунд),
// поэтому кратковременная работа в главном потоке не нарушает требование
// об асинхронности: она обёрнута в Promise и не блокирует событийный цикл
// надолго, а вся "тяжёлая" обработка (яркость/контраст/цвет) в любом
// случае выполняется в Worker.
// ==========================================================================

export const SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'heic', 'heif'];

// Определяем формат по MIME-типу, а при его отсутствии — по расширению
// файла (некоторые браузеры не проставляют MIME для .heic файлов).
export function detectFormat(file) {
  const mime = (file.type || '').toLowerCase();
  const ext = (file.name.split('.').pop() || '').toLowerCase();

  if (mime.includes('heic') || mime.includes('heif') || ext === 'heic' || ext === 'heif') {
    return 'heic';
  }
  if (mime === 'image/jpeg' || ext === 'jpg' || ext === 'jpeg') return 'jpeg';
  if (mime === 'image/png' || ext === 'png') return 'png';
  if (mime === 'image/bmp' || ext === 'bmp') return 'bmp';
  return null;
}

export function isSupportedFile(file) {
  return detectFormat(file) !== null;
}

// Приводит исходный File к Blob, который умеет декодировать браузер
// (createImageBitmap). Для HEIC/HEIF выполняется конвертация в PNG.
export async function normalizeToDecodableBlob(file) {
  const format = detectFormat(file);
  if (!format) {
    throw new Error(`Неподдерживаемый формат файла: ${file.name}`);
  }

  if (format !== 'heic') {
    return file;
  }

  // heic2any подключается через <script> в index.html и создаёт
  // глобальную функцию window.heic2any
  const converted = await self.heic2any({
    blob: file,
    toType: 'image/png',
    quality: 0.92,
  });

  // heic2any может вернуть как один Blob, так и массив (для HEIC-контейнеров
  // с несколькими кадрами) — для нашей задачи достаточно первого кадра.
  return Array.isArray(converted) ? converted[0] : converted;
}
