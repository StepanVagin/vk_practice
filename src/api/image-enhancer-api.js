// ==========================================================================
// image-enhancer-api.js
// Публичный API модуля улучшения изображений (см. раздел ТЗ
// "Рекомендуемые API модуля"). Реализует очередь задач:
//   - submitTask()    — метод постановки задачи
//   - getTaskStatus()  — метод получения статуса задачи
//   - cancelTask()     — метод прерывания задачи
//   - getResult()       — метод получения готового изображения
//   - событие 'taskstatuschange' — изменение статуса/прогресса задачи
//
// Каждая задача обрабатывается в отдельном Web Worker (src/worker/enhance.worker.js).
// Один Worker на задачу — простой и надёжный способ поддержать мгновенную
// отмену: достаточно вызвать worker.terminate(), не городя систему
// кооперативных флагов отмены внутри вычислений.
// ==========================================================================

import { normalizeToDecodableBlob, isSupportedFile } from '../decode/format-utils.js';

// Возможные статусы задачи на всём её жизненном цикле
export const TaskStatus = Object.freeze({
  QUEUED: 'queued',       // задача создана, ожидает начала обработки
  DECODING: 'decoding',   // определение формата / конвертация HEIC / декодирование пикселей
  ANALYZING: 'analyzing', // анализ статистики изображения (яркость/контраст/цвет)
  PROCESSING: 'processing', // применение коррекции
  ENCODING: 'encoding',   // кодирование результата в выходной формат
  DONE: 'done',           // готово, результат доступен через getResult()
  ERROR: 'error',         // ошибка обработки
  CANCELLED: 'cancelled', // задача прервана пользователем через cancelTask()
});

const WORKER_URL = new URL('../worker/enhance.worker.js', import.meta.url);

class Task {
  constructor(id) {
    this.id = id;
    this.status = TaskStatus.QUEUED;
    this.progress = 0;
    this.error = null;
    this.resultBlob = null;
    this.worker = null;
    this.cancelled = false; // кооперативный флаг для короткой стадии до создания Worker (конвертация HEIC)
  }
}

export class ImageEnhancerAPI extends EventTarget {
  constructor() {
    super();
    /** @type {Map<string, Task>} */
    this._tasks = new Map();
  }

  // --- Метод постановки задачи ---
  // Принимает исходное изображение (File/Blob) и необязательные параметры
  // коррекции { brightness, contrast, saturation } в диапазоне [-1..1]
  // (null/undefined = автоматический расчёт), а также параметры вывода
  // { outputFormat, quality }.
  // Возвращает идентификатор задачи немедленно — вся обработка идёт в фоне,
  // отслеживать её ход нужно через getTaskStatus()/событие 'taskstatuschange'.
  submitTask(file, options = {}) {
    if (!(file instanceof Blob)) {
      throw new TypeError('submitTask ожидает File или Blob с исходным изображением');
    }
    if (!isSupportedFile(file)) {
      throw new Error(`Неподдерживаемый формат файла: ${file.name || file.type}`);
    }

    const taskId = crypto.randomUUID();
    const task = new Task(taskId);
    this._tasks.set(taskId, task);

    this._setStatus(task, TaskStatus.QUEUED, 0);

    // Пайплайн запускается асинхронно и НЕ ожидается здесь — submitTask
    // должен вернуть идентификатор сразу же, как того требует ТЗ.
    this._runPipeline(task, file, options).catch((err) => {
      this._setStatus(task, TaskStatus.ERROR, task.progress, err.message || String(err));
    });

    return taskId;
  }

  // --- Метод получения статуса задачи ---
  getTaskStatus(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) {
      throw new Error(`Задача не найдена: ${taskId}`);
    }
    return { taskId, status: task.status, progress: task.progress, error: task.error };
  }

  // --- Метод прерывания задачи ---
  // Возвращает true, если удалось прервать ещё не завершённую задачу.
  cancelTask(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) return false;

    const finished = [TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED];
    if (finished.includes(task.status)) return false;

    task.cancelled = true;
    if (task.worker) {
      task.worker.terminate(); // мгновенно останавливает вычисления в Worker'е
      task.worker = null;
    }
    this._setStatus(task, TaskStatus.CANCELLED, task.progress);
    return true;
  }

  // --- Метод получения готового изображения ---
  async getResult(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) {
      throw new Error(`Задача не найдена: ${taskId}`);
    }
    if (task.status !== TaskStatus.DONE) {
      throw new Error(`Результат ещё не готов (текущий статус: ${task.status})`);
    }
    return task.resultBlob;
  }

  // Удаляет завершённую задачу из внутреннего реестра, освобождая память
  // (Blob результата). Вызывается после того, как результат больше не нужен.
  disposeTask(taskId) {
    this._tasks.delete(taskId);
  }

  // --- Внутренняя логика ---

  _setStatus(task, status, progress, error = null) {
    task.status = status;
    task.progress = progress;
    task.error = error;
    this.dispatchEvent(new CustomEvent('taskstatuschange', {
      detail: { taskId: task.id, status, progress, error },
    }));
  }

  async _runPipeline(task, file, options) {
    this._setStatus(task, TaskStatus.DECODING, 2);

    // Приведение HEIC к декодируемому браузером формату (см. format-utils.js).
    // Если задачу отменили прямо во время этого шага — прекращаем работу,
    // не дожидаясь создания Worker'а.
    const decodableBlob = await normalizeToDecodableBlob(file);
    if (task.cancelled) return;

    const worker = new Worker(WORKER_URL);
    task.worker = worker;

    worker.onmessage = (event) => {
      const msg = event.data;

      if (msg.type === 'progress') {
        this._setStatus(task, msg.status, msg.progress);
      } else if (msg.type === 'result') {
        task.resultBlob = msg.blob;
        this._setStatus(task, TaskStatus.DONE, 100);
        worker.terminate();
        task.worker = null;
      } else if (msg.type === 'error') {
        this._setStatus(task, TaskStatus.ERROR, task.progress, msg.message);
        worker.terminate();
        task.worker = null;
      }
    };

    worker.onerror = (event) => {
      this._setStatus(task, TaskStatus.ERROR, task.progress, event.message);
      worker.terminate();
      task.worker = null;
    };

    worker.postMessage({
      blob: decodableBlob,
      options: {
        brightness: options.brightness ?? null,
        contrast: options.contrast ?? null,
        saturation: options.saturation ?? null,
        outputFormat: options.outputFormat || 'image/png',
        quality: options.quality ?? 0.92,
      },
    });
  }
}
