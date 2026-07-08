// ==========================================================================
// app.js
// Демонстрационная страница: связывает элементы управления с публичным API
// (ImageEnhancerAPI) и показывает результат — прогресс обработки,
// изображение "до/после", кнопки отмены и скачивания.
// ==========================================================================

import { ImageEnhancerAPI, TaskStatus } from './api/image-enhancer-api.js';

const api = new ImageEnhancerAPI();

const els = {
  dropzone: document.getElementById('dropzone'),
  dropzoneText: document.getElementById('dropzoneText'),
  fileInput: document.getElementById('fileInput'),
  autoMode: document.getElementById('autoMode'),
  manualSliders: document.getElementById('manualSliders'),
  brightness: document.getElementById('brightness'),
  contrast: document.getElementById('contrast'),
  saturation: document.getElementById('saturation'),
  processBtn: document.getElementById('processBtn'),
  cancelBtn: document.getElementById('cancelBtn'),
  downloadBtn: document.getElementById('downloadBtn'),
  progressWrap: document.getElementById('progressWrap'),
  progressFill: document.getElementById('progressFill'),
  progressText: document.getElementById('progressText'),
  originalPreview: document.getElementById('originalPreview'),
  resultPreview: document.getElementById('resultPreview'),
  log: document.getElementById('log'),
};

// Человекочитаемые названия статусов задачи для прогресс-бара
const STATUS_LABELS = {
  [TaskStatus.QUEUED]: 'В очереди…',
  [TaskStatus.DECODING]: 'Декодирование изображения…',
  [TaskStatus.ANALYZING]: 'Анализ яркости/контраста/цвета…',
  [TaskStatus.PROCESSING]: 'Применение коррекции…',
  [TaskStatus.ENCODING]: 'Кодирование результата…',
  [TaskStatus.DONE]: 'Готово',
  [TaskStatus.ERROR]: 'Ошибка',
  [TaskStatus.CANCELLED]: 'Отменено',
};

let selectedFile = null;
let currentTaskId = null;
let startedAt = 0;

function log(message) {
  const time = new Date().toLocaleTimeString('ru-RU');
  els.log.textContent += `[${time}] ${message}\n`;
}

function setControlsState({ canProcess, canCancel, canDownload }) {
  els.processBtn.disabled = !canProcess;
  els.cancelBtn.disabled = !canCancel;
  els.downloadBtn.disabled = !canDownload;
}

function updateProgress(status, progress) {
  els.progressWrap.hidden = false;
  els.progressFill.style.width = `${progress}%`;
  els.progressText.textContent = `${STATUS_LABELS[status] || status} (${progress}%)`;
}

// --- Выбор файла (клик или drag&drop) ---

els.fileInput.addEventListener('change', () => {
  if (els.fileInput.files[0]) handleFileSelected(els.fileInput.files[0]);
});

els.dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  els.dropzone.classList.add('dragover');
});
els.dropzone.addEventListener('dragleave', () => els.dropzone.classList.remove('dragover'));
els.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  els.dropzone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) handleFileSelected(e.dataTransfer.files[0]);
});

function handleFileSelected(file) {
  selectedFile = file;
  els.dropzoneText.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} МБ)`;
  els.originalPreview.src = URL.createObjectURL(file);
  els.resultPreview.src = '';
  setControlsState({ canProcess: true, canCancel: false, canDownload: false });
}

// --- Переключение авто/ручного режима ---

els.autoMode.addEventListener('change', () => {
  els.manualSliders.classList.toggle('disabled', els.autoMode.checked);
});
els.manualSliders.classList.add('disabled');

// --- Запуск обработки ---

els.processBtn.addEventListener('click', () => {
  if (!selectedFile) return;

  const options = els.autoMode.checked
    ? {}
    : {
      brightness: Number(els.brightness.value),
      contrast: Number(els.contrast.value),
      saturation: Number(els.saturation.value),
    };

  startedAt = performance.now();
  currentTaskId = api.submitTask(selectedFile, options);
  log(`Задача поставлена: ${currentTaskId}`);
  setControlsState({ canProcess: false, canCancel: true, canDownload: false });
});

// --- Отмена ---

els.cancelBtn.addEventListener('click', () => {
  if (!currentTaskId) return;
  const cancelled = api.cancelTask(currentTaskId);
  if (cancelled) log('Задача отменена пользователем');
});

// --- Скачивание результата ---

els.downloadBtn.addEventListener('click', async () => {
  if (!currentTaskId) return;
  const blob = await api.getResult(currentTaskId);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'enhanced.png';
  a.click();
  URL.revokeObjectURL(url);
});

// --- Событие изменения статуса задачи (единая точка отслеживания прогресса) ---

api.addEventListener('taskstatuschange', async (event) => {
  const { taskId, status, progress, error } = event.detail;
  if (taskId !== currentTaskId) return;

  updateProgress(status, progress);

  if (status === TaskStatus.DONE) {
    const elapsedSec = ((performance.now() - startedAt) / 1000).toFixed(1);
    log(`Обработка завершена за ${elapsedSec} с`);
    const blob = await api.getResult(taskId);
    els.resultPreview.src = URL.createObjectURL(blob);
    setControlsState({ canProcess: true, canCancel: false, canDownload: true });
  } else if (status === TaskStatus.ERROR) {
    log(`Ошибка обработки: ${error}`);
    setControlsState({ canProcess: true, canCancel: false, canDownload: false });
  } else if (status === TaskStatus.CANCELLED) {
    setControlsState({ canProcess: true, canCancel: false, canDownload: false });
  }
});
