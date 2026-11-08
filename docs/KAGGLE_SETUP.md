# ShadowTrack private Kaggle GPU detection

> Этот документ описывает исходный segmentation SAMPLE. Для готового демо используйте [MVP_SETUP.md](MVP_SETUP.md), для нового Detection FULL — [KAGGLE_DETECTION_V2.md](KAGGLE_DETECTION_V2.md).

Для нового bbox detection pipeline и schema v2 используйте
[KAGGLE_DETECTION_V2.md](KAGGLE_DETECTION_V2.md) и
[kaggle_detection_v2.ipynb](../notebooks/kaggle_detection_v2.ipynb).
Ниже сохранена инструкция предыдущего segmentation pipeline для воспроизводимости.

Подготовлен гибридный процесс: YOLO person segmentation в приватном Kaggle GPU
Notebook, затем проверенный импорт и локальный ByteTrack на CPU.
В данном сеансе ничего в Kaggle не загружалось, GPU benchmark не выполнялся.

## Приватность

Dataset и Notebook имеют отдельные настройки доступа. Владелец может сделать
Notebook публичным даже при приватном Dataset, поэтому проверяйте **оба**
переключателя и каждую сохраняемую версию. Private доступен владельцу,
приглашённым участникам и Kaggle согласно его политике; это не обещание локальной
или сквозно зашифрованной обработки. См. [Kaggle datasets](https://www.kaggle.com/docs/datasets).

У notebook по умолчанию `EXECUTE=False`, `MODE="SAMPLE"`. Для запуска настройте
папки inputs и установите `EXECUTE=True`. Дополнительных флагов подтверждения нет.

Кадры не выводятся inline, в stdout/stderr, TensorBoard, W&B или публичные
outputs. Annotated JPEG/GIF и gallery пишутся только в `_private_quality`
в приватном Notebook. Не публикуйте Notebook, его outputs, Dataset,
маски или детекции. Data-only export не является анонимизацией: bbox, время
и силуэты также остаются чувствительными данными.

## Локальная подготовка без загрузки

Из корня проекта:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,detection-cpu,notebooks]"
.venv/Scripts/shadowtrack.exe detection prepare
```

Команда создаёт:

- `artifacts/detection/input_evidence/`: source manifest, исходные frame indices,
  PTS и segment IDs, обычные `.json` без внутреннего gzip и без изображений.
- `artifacts/detection/portable/shadowtrack_detection_code_<hash>.zip`:
  версионированный код, model lock, лицензия, schema, очищенный Notebook
  и соответствующая этому коду папка `input_evidence/` с обычными JSON.
- `artifacts/detection/portable/shadowtrack_detection_code_<hash>/`: та же
  распакованная папка для загрузки в Dataset без архивов.
- `artifacts/detection/portable/code_artifact.json`: SHA256 всего ZIP и fingerprint
  общего detection-кода.

Подготовленные файлы уже созданы в текущем рабочем каталоге. Команда намеренно
отказывается перезаписывать существующие snapshots. Для новой версии используйте
свежие каталоги `--output-dir` и `--evidence-dir`, затем соответствующие пути
в Notebook и локальном импорте. Изменение общего кода после подготовки требует
нового evidence/code snapshot: Notebook проверяет их согласованность.

Веса `yolo11n-seg.pt` уже получены из официального релиза в `.tools/models`.
Это небольшой файл **6 182 636 bytes**, а не обученная в этом проекте модель.
SHA256: `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`.
Не добавляйте `.pt`, исходные MKV, частные outputs и галереи в git.

## Приватная загрузка пользователем

Предпочтительный способ — вручную создать Private Dataset через Kaggle UI.
Загрузите пять MKV, сохранив исходные имена
и байты. Не перекодируйте их и не переименовывайте как замену source hash.
План офиса, старые contact sheets и иные изображения этому этапу не нужны.

Создайте private inputs:

1. Videos: только пять исходных MKV.
2. Tools: распакованная папка кода (`code_manifest.json`, `src/`, `configs/`, `input_evidence/`)
   и закреплённый `yolo11n-seg.pt`. ZIP можно загрузить для автоматической распаковки
   Kaggle; Notebook читает полученную папку, а не файл архива.

Отдельный Evidence Dataset больше не обязателен: `EVIDENCE_DIR = CODE_DIR`
использует `input_evidence/`, включённый в tools. При необходимости можно указать
отдельную папку evidence той же версии.

В качестве альтернативы опытный пользователь может **сам** выполнить Kaggle CLI
после настройки своих credentials вне репозитория:

```powershell
kaggle datasets init -p C:/PRIVATE_STAGING/shadowtrack-videos
# Вручную заполнить dataset-metadata.json: свой owner/id, название и допустимые владельцем условия.
kaggle datasets create -p C:/PRIVATE_STAGING/shadowtrack-videos --dir-mode zip
```

В официальном CLI отсутствие `--public` означает Private; не добавляйте этот флаг.
Метаданные лицензии исходных офисных видео здесь не выдумываются. Проверьте
Private в UI после загрузки. Эти команды приведены для ручного выполнения;
ShadowTrack их не вызывает. См. [официальные dataset-команды Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

Если Kaggle распаковал evidence иначе, найдите `source_manifest.json` в своём
private input и укажите его родительский каталог как `EVIDENCE_DIR`.

## Настройка Notebook и GPU

Откройте `notebooks/kaggle_detection.ipynb` как Private Notebook. В Settings
выберите доступный GPU accelerator. Укажите `CODE_DIR`, `EVIDENCE_DIR`, `VIDEO_DIR`
и `WEIGHTS_INPUT` как пути в `/kaggle/input`. `CODE_DIR` — папка с
`code_manifest.json` или родительский Dataset root. Уникальные вложенные папки
кода, evidence, видео и весов находятся автоматически; неоднозначные пути требуют
более точного каталога. Хеш архива в Notebook вводить не нужно.

### Если Kaggle уже распаковал `.json.gz`

Новый пакет содержит `camera_*_frames.json`, поэтому повторной распаковки evidence
не требуется. Loader также принимает `.json.gz` и распакованный `.json`, когда
manifest содержит SHA256 содержимого. Для старого manifest без этого хеша
используйте новый bundled `input_evidence/`: проверку целостности нельзя заменить
переименованием файла или пересчётом timestamps.

Если traceback ссылается на `shadowtrack_detection_code_4c869407...`, используется
старый пакет. Обновите подключённую версию tools Dataset новым архивом целиком,
откройте новый Notebook и выполните Restart Session / Run All. По умолчанию:

```python
CODE_DIR = Path("/kaggle/input/datasets/mario12508/shadowtrack-tools")
EVIDENCE_DIR = CODE_DIR
```

Bootstrap очищает импортированные модули `shadowtrack` из кеша kernel и выводит
фактический путь `Loaded package`. Пути Dataset могут отличаться; укажите каталог
с актуальным `code_manifest.json`, если подключены несколько пакетов.

Notebook сверяет SHA256 файлов по `code_manifest.json` в распакованной папке
и импортирует `shadowtrack.detection.*` из её `src`. Внутри Notebook нет второй
реализации decoder, YOLO wrapper, chunk writer, schema или checkpoint/resume.

Свежий Kaggle kernel должен иметь предустановленный CUDA-enabled torch. Notebook
проверяет его наличие и устанавливает бинарные пакеты:

- `ultralytics==8.3.161`;
- `av==14.2.0`;
- `opencv-python==4.11.0.86`.

Предустановленный torch не подменяется CPU-only wheel. Записываются версии
torch/CUDA, NumPy, PyAV/других пакетов, `torch.cuda.is_available()`, `nvidia-smi`,
название GPU и доступная/полная VRAM. `auto` выбирает CUDA при её доступности,
иначе CPU. Явный `cuda:0` без CUDA завершается ошибкой. FULL на CPU запрещён.

Веса должны уже лежать в private input; неявная загрузка по имени модели
запрещена. При необходимости пользователь может отдельно включить
`ALLOW_SMALL_PUBLIC_WEIGHTS_DOWNLOAD=True`: скачивается только известный маленький
публичный файл с проверкой размера и SHA256. Офисные данные наружу не отправляются.

## Model pin и лицензия

Используется предобученная COCO **YOLO11n-seg**, только `class_id=0` (`person`).
Файл, URL, SHA256 и параметры закреплены в `configs/detection_model_lock.json`.
Модель не обучается и не дообучается. Сравниваются одинаковые SAMPLE-кадры:

- `nano_640`: imgsz 640, conf 0.10, IoU 0.70, max_det 100, FP32, retina_masks;
- `nano_960`: те же условия, imgsz 960.

Ultralytics публикует код/модели на условиях AGPL-3.0 с альтернативными Enterprise
условиями. Закреплённый текст включён в `licenses/ULTRALYTICS_AGPL-3.0.txt`.
Проверьте применимые условия для своего использования до запуска; этот проект
не принимает за пользователя решение о лицензировании.
Источники: [YOLO11](https://docs.ultralytics.com/models/yolo11/),
[лицензия закреплённого исходника](https://github.com/ultralytics/ultralytics/blob/v8.3.161/LICENSE),
[условия Ultralytics](https://www.ultralytics.com/license).

## SAMPLE, сравнение и visual review

### Прогресс и диагностика долгого запуска

В Notebook включены `SHOW_PROGRESS=True` и `PROGRESS_INTERVAL_SECONDS=10`.
Старт/окончание крупных этапов и сохранение чанков выводятся сразу с flush.
Отдельный поток каждые 10 s печатает heartbeat, в том числе во время долгого
вызова YOLO, прогрева или I/O. Это работает и при подавлении внутренних логов YOLO.
При жёсткой остановке kernel или удержании Python GIL heartbeat также может задержаться.

SAMPLE — длительность выбранных окон, **не бюджет времени выполнения**. На текущих
данных две конфигурации означают 10 полных проходов (5 камер × 2), **288 220 кадров
декодирования**, но только **7 318 кадров YOLO**. Даже после последнего SAMPLE-окна
декодер доходит до EOF для проверки полного числа кадров/PTS. Прогрев, CPU decoding,
SHA256, преобразование RGB, маски RLE и запись outputs занимают отдельное время.

В консоли видны:

- `device_selected` и `cpu_fallback`: выбранное устройство; `auto` может выбрать CPU.
  При необходимости задать обязательный GPU используйте `DEVICE="cuda:0"`.
- `phase`: `source_sha256`, `model_load`, `model_warmup`, `decode_next_frame`,
  `yolo_predict`, `masks_to_cpu`, `mask_rle_encode`, `private_quality`, `chunk_commit`,
  `quality_gallery_save`, export/import/restore.
- Камера/конфигурация, исходный frame index и raw PTS, счётчики decoded/YOLO,
  processed и committed frames, сохранённые/запланированные чанки.
- Прочитанные/полные байты SHA256, копируемые файлы, последнее измерение VRAM.
- `camera_wall_fps` и приблизительный `camera_eta_s`: скорость текущего прохода
  с учётом инференса/I/O. Это не GPU benchmark и не гарантированный ETA всего run.
- `phase_age` и `idle`: длительность этапа и время без изменения счётчиков.
  Неизменные счётчики сами по себе не доказывают зависание.

Числовые файлы main run находятся в
`/kaggle/working/shadowtrack_private/telemetry/<mode>/progress.json`
и `events.jsonl`. Первый содержит текущее состояние, второй — события/heartbeat.
При исключении записываются статус, тип ошибки, `failed_phase` и накопленные
`stage_seconds_inclusive` (вложенные интервалы перекрываются; складывать их нельзя).
Export пишет отдельный лог в `telemetry/<mode>/export/`; telemetry не входит в
data-only checkpoint. Ни кадров, ни масок/массивов изображений в логах нет.

API и CLI `import`, `export`, `local-track`, а также `restore_checkpoint`
показывают прогресс. В Python можно изменить `progress_interval=5` или отключить
вывод параметром `progress=False`. На resume processed/committed включают уже
сохранённые кадры; `inferred_frames_session` считает только новые вызовы текущего run.
COMPLETE checkpoint сообщает `resume_complete` и не запускает модель/decoder.
Текущая уже выполняющаяся старая ячейка не получает новые логи автоматически:
для новой версии нужен обновлённый tools Dataset/Notebook и перезапуск kernel.
Старый checkpoint сохраните отдельно; разные code hashes не смешиваются.

Оставьте `MODE="SAMPLE"` и включите `EXECUTE=True`.
Будет выбран raw-PTS интервал 10 s из каждого из трёх
timestamp segments каждой камеры: суммарно **30 s на камеру**. План одинаков
для обеих конфигураций. На текущих источниках — 3 659 кадров на конфигурацию.

SAMPLE всё равно последовательно декодирует каждый поток до EOF, чтобы проверить
полное число и порядок кадров. Только выбранные окна проходят YOLO. Часть времени
может уйти на CPU decoding, поэтому throughput inference отдельно от end-to-end
времени не означает скорость всего workflow.

`benchmark.json` содержит processed/empty frames, число детекций, synchronized
predict wall time, throughput и peak allocated/reserved VRAM в manifests чанков.
Warm-up на синтетическом пустом изображении не включён. Аппаратное окружение
сохранено по runtime IDs; смешанный CPU/GPU resume не считается полноценным GPU
benchmark. Без измерений значения не подставляются.

Откройте только private output-файлы `_private_quality/<config_hash>/camera_*`:
короткий annotated GIF, JPEG gallery, отдельные примеры, quality index. GIF —
contact preview с явно указанными 10 FPS и raw PTS в подписях, а не синхронное
видео. Проверьте телевизоры/отражения, перекрытия столом, дальних людей, двери,
ноги и различия камер 20/25 FPS. Precision/recall и AP без GT не вычисляются.

`FULL_REVIEW_TEMPLATE.json` остаётся unchecked. После частного просмотра обеих
конфигураций пользователь записывает reviewer, заметки, отмечает проверки,
выбирает `selected_config_hash` и отдельно устанавливает `approved_full=true`.
Template связан SHA256 с COMPLETE SAMPLE manifest и benchmark.

## Точная связь с исходниками

PyAV проходит исходный поток последовательно. Для каждого кадра проверяются
фактические `frame.pts × frame.time_base`, порядковый номер и размер против
сохранённого аудита. После EOF проверяется общее число кадров. Никаких
`timestamp = frame_index/FPS`, перенумерации через seek или переписывания MKV нет.

Если число/порядок/PTS отличаются, сохраняется data-only diagnostic, run получает
`FAILED_DECODE`, итоговый COMPLETE marker отсутствует. Ранние чанки могут быть
на диске как checkpoint, но импорт для трекинга их отклонит. Нельзя «исправить»
это заменой исходного audit PTS на показания другого декодера.

## Checkpoint, export и resume

Чанки разделены по configuration/camera/timestamp segment/frame range. Каждый
содержит `frames.jsonl.gz` и manifest с числом processed/empty frames, SHA256,
source/model/config hashes и метриками. Данные и manifest пишутся во временный
каталог с fsync и атомарным rename. Корневой COMPLETE marker отдельно связывает
manifest SHA256, общее число кадров и число чанков.

При возобновлении проверяются все committed chunks; незавершённые staging-файлы
не считаются результатами. Если процесс упал между rename чанка и обновлением
корневого индекса, целый проверенный чанк восстанавливается из своего manifest.
Смена кода, source evidence, модели, config или selection plan требует нового run.
Целый COMPLETE run при повторном вызове не делает новый inference.

В `finally` notebook создаёт отдельную data-only папку snapshot под
`/kaggle/working/shadowtrack_private/checkpoints/<mode>_<timestamp>/`.
В ней есть `export_manifest.json` с inventory, размерами и SHA256 всех файлов;
`_private_quality` в snapshot не попадает. Перед окончанием сессии скачайте папку
или сохраните **Private** Kaggle Version с outputs. Для сохранения вручную:

```python
from shadowtrack.detection.transport import export_run_directory
export_run_directory(RUN_DIR, RUN_DIR.parent / "manual_checkpoint")
```

Если прошлую сессию прервали, прикрепите распакованную checkpoint-папку как
private input, задайте `RESUME_DIR` и используйте прежние параметры. При пустом
`RUN_DIR` код проверит inventory/SHA256 и атомарно восстановит папку в working.
Для FULL задайте `REVIEWED_SAMPLE_DIR` как папку завершённого SAMPLE и
`FULL_REVIEW_JSON` как файл review. Архивы на входе Notebook не используются.
Если Kaggle распаковал вложенные `frames.jsonl.gz` в `frames.jsonl`, новый loader
проверит SHA256 содержимого и сохранит эту форму без пересжатия и изменения PTS.
Частичный checkpoint пригоден для resume, но не для локального tracking import.
Смена detection-кода требует нового SAMPLE; checkpoints старого code hash
не смешиваются с новой версией.

Cloud GPU quota, время сессии, наличие accelerator и output storage зависят от
аккаунта и текущих ограничений Kaggle; гарантированная длительность не зашита.
Лимит `MAX_WALL_SECONDS` — собственный stop budget, а не квота платформы.
`/kaggle/working` хранит результаты текущей сессии; одно наличие файла на диске
не заменяет Save Version/download. ОС может завершить kernel без `finally`, поэтому
атомарные чанки сохраняются независимо от финального snapshot. См. [Kaggle notebooks](https://www.kaggle.com/docs/notebooks).

FULL включается только вручную, на GPU, с COMPLETE реальным GPU SAMPLE, review
и одной выбранной конфигурацией. Эта подготовка не запускает FULL. Для нового
benchmark после уже завершённого CPU SAMPLE используйте новый RUN_DIR.

## Локальный импорт и ByteTrack

Скачайте data-only export вручную, затем:

```powershell
.venv/Scripts/shadowtrack.exe detection import --source C:/PRIVATE_DOWNLOADS/sample_checkpoint_folder
```

Импорт проверяет export inventory, schema_version, manifests, SHA256, completion
marker, processed frame count, последовательность indices, исходные MKV,
raw PTS, segment IDs, model/config hashes, bbox и наличие корректных RLE-масок.
Повреждённый или незавершённый upload не становится локальным run. Приватные
quality media в локальный detection store не импортируются.

Команда напечатает путь `artifacts/detections/<run_id>`. В `configs/local_track.toml`
укажите этот `detection_run` и выбранный после сравнения `configuration_name`.
`nano_640` в шаблоне — пример, а не измеренный победитель. Затем:

```powershell
.venv/Scripts/shadowtrack.exe local-track --config configs/local_track.toml
```

Это CPU-only `supervision.ByteTrack==0.25.1`, без torch и YOLO. Чанки валидируются
и читаются по одному, masks остаются references на исходные детекции, а output
пишется потоково. Пустые обработанные кадры тоже передаются трекеру. Новый
timestamp segment или nonconsecutive frame index сбрасывает состояние; ID
изолированы камерой, сегментом и началом непрерывного фрагмента.

Номинальные 20/25 FPS используются только для lost-track buffer, не для создания
PTS. Трекер не создаёт наблюдений в отсутствующих кадрах и не объединяет камеры.
Time Alignment Engine и его PARTIAL acceptance остаются без изменений.
