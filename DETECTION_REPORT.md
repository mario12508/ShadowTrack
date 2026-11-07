# ShadowTrack hybrid detection pipeline

**Detection v2: LOCAL_VALIDATED / GPU_PENDING.** Добавлены batched YOLO Detection,
неизменяемые raw frame records schema v2 с пустыми кадрами, отдельный scene_filter,
один decode на камеру, DEMO MP4 как производный результат и FULL coverage всех
исходных кадров. Перенос координат crop камеры 5, checkpoint/export/import и CPU
ByteTrack с local-ID video проверены локально. Реального нового GPU benchmark,
измеренного ускорения и пяти офисных MP4 от detection weights пока нет.
Полная инструкция: [KAGGLE_DETECTION_V2.md](KAGGLE_DETECTION_V2.md).

Локальная приёмка v2: **127 passed** (45 audit/sync, 57 legacy detection, 25 новых).
Интеграционный тест создаёт пять синтетических MKV, выполняет общий engine с
явно synthetic predictor, экспортирует и импортирует schema v2, запускает настоящий
CPU ByteTrack и кодирует пять MP4 с локальными ID. Проверены пустые кадры,
20/25 FPS, ненулевые source PTS и запрет повторного inference после экспорта.
Отдельные тесты проверяют FULL export, jumps/segment boundaries, crop, resume,
повреждение чанков и независимость machine records от ошибки MP4, включая повторный
сбой при закрытии encoder. Resume восстанавливает потерянные/повреждённые MP4 и
PTS sidecars из records без повторного YOLO. Это проверка
контракта, а не качества новой YOLO на людях и не измерение GPU throughput.

Планы по неизменным сохранённым PTS: BENCHMARK — 1 222 кадра на конфигурацию,
DEMO — 3 663 кадра одной конфигурацией, FULL — **144 110 кадров**:
camera 1: 30 083, 2: 30 077, 3: 30 080, 4: 23 906, 5: 29 964. Каждый выбранный
кадр сохраняется независимо от наличия bbox. Все raw кандидаты остаются в records
с решениями scene_filter; у пустого кадра есть отдельная processed запись.
Короткие окна требуют одного полного decode на камеру для проверки порядка/PTS/EOF.

Артефакты локальной проверки: [pytest XML](artifacts/detection/v2/tests.xml),
[планы](artifacts/detection/v2/plan_summary.json),
[проверка пакета и неизменности входов](artifacts/detection/v2/verification.json).
Приёмка реализации: **LOCAL_VALIDATED_GPU_PENDING**. Новые GPU benchmark, визуальная
проверка правил и реальные пять офисных MP4 остаются невыполненными до ручного
запуска Kaggle. Ускорение и превосходство качества не заявляются. FULL не запускался.

Ниже — результаты сохранённого реального segmentation SAMPLE и его review;
они не подменяются показателями новой модели.

**Legacy SAMPLE acceptance: SAMPLE_INTEGRITY_PASS / QUALITY_PARTIAL / FULL_NOT_APPROVED.**
Получен и локально проверен реальный Kaggle GPU SAMPLE пользователя: run
`988a8de217bb1d3c0b951e67`, 60 чанков и 7 318 frame records. У агента не было
облачной загрузки или повторного YOLO inference. FULL не выполнялся.
Данные, Data Audit и Time Alignment Engine сохранены без изменений.
Подробный разбор и следующий patch: [SAMPLE_REVIEW.md](SAMPLE_REVIEW.md).
Дополнение по архиву с фотографиями и исходному ТЗ: явного требования пиксельных
масок не найдено; рекомендован основной bbox detection mode. Подтверждены ложные
маски на лампе и конструкции у лестницы камеры 2. Переключение runtime/schema
в этом review не выполнялось; план миграции и ограничения изложены в отчёте.

## Что реализовано

- [Notebook](notebooks/kaggle_detection.ipynb) импортирует версионированный общий
  `src/shadowtrack/detection` из распакованной папки input. Outputs и execution counts
  очищены; исходное значение `EXECUTE=False`, режим SAMPLE.
- Model/device/version pin, `torch.cuda.is_available`, `nvidia-smi`, available
  VRAM, weights SHA256, запись окружения и отключение внешних logger integrations.
- Sequential frame/PTS guard с diagnostic report и запретом COMPLETE при
  расхождении с аудитом.
- Person segmentation schema, RLE в исходном разрешении, processed-empty records,
  чанки по configuration/camera/segment/frame-range.
- Atomic writes, SHA256, manifest/count checks, checkpoint folder export/restore/resume.
- Notebook использует папки кода, evidence, видео и checkpoint под `/kaggle/input`;
  уникальные вложенные папки находятся автоматически. Программные подтверждения убраны.
- В tools-пакет включён соответствующий коду `input_evidence/` в обычных `.json`.
  Поддержана распаковка `.json.gz`/`.jsonl.gz` в Kaggle с проверкой SHA256 содержимого;
  старые модули очищаются из кеша kernel до импорта нового кода.
- Private-only annotated examples и gallery; data-only export по умолчанию.
- `shadowtrack detection import --source PATH` и CPU
  `shadowtrack local-track --config CONFIG`, с reset на разрывах и детерминированным
  replay сохранённых детекций.
- Проверка с ограниченной памятью: import и tracker обрабатывают один чанк;
  полный набор RLE-масок не накапливается в RAM ноутбука.
- Metadata-only прогресс: отдельный heartbeat, этапы SHA256/model/warmup/decode/
  YOLO/masks/quality/commit/export, decoded/inferred/committed counters, устройство,
  приблизительный ETA камеры и JSON/JSONL telemetry вне detection chunks.
  Повторное RGB преобразование одного кадра для quality исключено.

Формальный envelope: [detection_frame.schema.json](schemas/detection_frame.schema.json).
Дополнительные проверки идентичности, геометрии, исходных PTS и полноты RLE
выполняет общий Python validator.

## Исходные данные и временные шкалы

Portable evidence получен только из реальных `artifacts/inspect/stats.json`,
покамерных `frames.json` и `artifacts/sync/time_alignment.json`. Исходные байты
видео и video frame evidence сверяются SHA256. В каждом detection record сохранены:
camera_id, source_frame_index, raw_media_pts, timestamp_segment_id, bbox_xyxy,
confidence, class_id, mask_rle, model_hash, config_hash.

У пустого кадра есть отдельный envelope: `processed=true`, `frame_status=empty`,
`detections=[]`. Пропущенный кадр не может быть представлен таким envelope.
Число processed frames должно точно покрывать объявленный selection plan.
RLE — uncompressed COCO format, column-major, zero run first; размеры маски
равны исходным H×W. Missing/empty instance mask при наличии bbox — ошибка.

Raw PTS не меняются. Глобальное время не подставляется вместо них; engine
PARTIAL не пересчитывается и межкамерные связи не расширяются. `local-track`
не делает global association. Исходные FPS используются только внутри настройки
ByteTrack lost buffer, а не для синтеза timestamp.

При подготовке PyAV 14.2.0 прошёл smoke test первых восьми кадров всех MKV.
Теперь полученный Kaggle manifest подтверждает десять VALIDATED проходов,
288 220 проверок кадров. Локальный import сверил source SHA256, каждый SAMPLE
frame index/PTS/segment и RLE. Это не повтор полного локального decode.
При mismatch чанки остаются недопустимыми для импорта, run становится
FAILED_DECODE, создаётся metadata-only diagnostic.

## SAMPLE и две конфигурации

Подготовлены один небольшой предобученный `yolo11n-seg` и два режима:
`nano_640` и `nano_960`, одинаковые conf 0.10 / IoU 0.70 / max_det 100,
classes=[0], FP32 и retina_masks. Выбор более крупной модели и обучение не нужны.

На каждую камеру выбирается 30 raw-PTS секунд, по 10 s из каждого из трёх
timestamp segments. На текущих данных план содержит:

- Камера 1: 765 кадров.
- Камера 2: 764 кадра.
- Камера 3: 765 кадров.
- Камера 4: 600 кадров.
- Камера 5: 765 кадров.

Итого **3 659 кадров на конфигурацию**, 15 временных окон и 30 чанков на
конфигурацию при chunk size 128. Полученный COMPLETE SAMPLE точно покрывает этот план.
Оба режима получают одинаковые исходные кадры. SAMPLE не означает ограниченный
EOF-check: поток декодируется целиком, но YOLO применяется только к выборке.

Модель получена из официального versioned asset, размер 6 182 636 bytes;
SHA256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`.
Lock: [detection_model_lock.json](configs/detection_model_lock.json).
Файл весов не включается в исходники/code ZIP; пользователь прикрепляет его
вручную как private input. Лицензия закреплённого компонента — AGPL-3.0,
альтернативные Enterprise terms предоставляет Ultralytics; условия должны быть
рассмотрены перед использованием. См. [лицензию](licenses/ULTRALYTICS_AGPL-3.0.txt)
и [официальные условия](https://www.ultralytics.com/license).

## Измерения и качество

**GPU benchmark: MEASURED в запуске пользователя на Tesla T4.**
Nano_640: 3 422 detections, 17.50 predict FPS, 209.14 s predict time.
Nano_960: 4 615 detections, 13.24 predict FPS, 276.37 s predict time.
Wall time по логу — 2 581.4 s; predict занимает 18.8%, остальное включает decode,
CPU postprocessing и I/O. Точных exclusive decode timings в переданном backup нет.

Data-only архив не содержал галерей. Локально сопоставлены 15 кадров с проверенным
PTS: обнаружены false positives на предметах/ТВ и дубли на частично видимых людях;
есть и полезные дополнительные находки 960. Общее число боксов не доказывает
превосходство 960. AP и итоговый победитель не определены; FULL review unchecked.

Notebook при запуске сохранит `benchmark.json`: synchronized predict
wall time, processed/empty frames, detection count, throughput и VRAM, с hardware
runtime IDs. Decoder time, synthetic warm-up и mask serialization отделены от
predict throughput. CPU-only и mixed-device запуски не принимаются как полноценный
GPU benchmark для FULL.

Private gallery/annotated GIF проверяются вручную по телевизорам и отражениям,
окклюзиям столом, дальним людям, дверям, ногам и 20/25 FPS. Без ground truth нет
AP/precision/recall. Синтетические gallery/preview проверены CPU-тестом только
во временном тестовом каталоге. Кадры реальных офисных видео на этом этапе
в outputs или репозиторий не сохранялись.

FULL требует реального COMPLETE GPU SAMPLE, двух сравниваемых конфигураций,
reviewer, всех checked quality checks, неизменных manifest/benchmark hashes,
одной selected config и отдельного `approved_full=true`. По умолчанию всё это
отсутствует. FULL не запущен.

## Checkpoint, transport и CPU tracking

Chunk не считается завершённым без проверенных manifest, processed count и
SHA256. После fsync он становится видимым атомарным rename. Root COMPLETE marker
появляется только после всех запланированных чанков и полного decode validation
всех камер/конфигураций. Complete orphan chunk может быть восстановлен после
сбоя индекса; неполный staging не принимается.

Export содержит файл inventory, размеры и SHA256. Повреждённый upload,
неподдерживаемый schema_version, wrong source video, duplicate/shuffled indices,
missing mask или незавершённый run отклоняются. Checkpoint пригоден для resume,
но не для tracking import до COMPLETE. Notebook сохраняет data-only папку;
локальный ZIP export также поддерживается. Оба варианта исключают всю private
quality media; маски и координаты всё равно должны оставаться приватными.

ByteTrack закреплён на `supervision==0.25.1` (MIT). Локальный replay использует
только сохранённые bbox/confidence; маски не регенерируются и остаются ссылками
на detection run. Empty frames обновляют состояние трекера. Смена сегмента либо
непоследовательный frame index создаёт новый namespace ID и сбрасывает tracker.
Выход содержит только фактически сопоставленные наблюдения, без вымышленных
предсказанных кадров между ними.

## Проверки

**102 passed, 0 failed, 0 skipped:** все прежние 45 тестов и 57 CPU-проверок detection.
[Последний JUnit](artifacts/detection/sample_review/tests.xml). Новые тесты включают:

- empty frame и полный RLE roundtrip;
- shuffled/duplicate frames, cross-segment chunks, wrong PTS;
- corrupt chunk, interrupted output/upload, отсутствие completion marker;
- missing masks, недопустимые bbox/class/confidence, wrong video hash;
- checkpoint/restore/resume и идемпотентный COMPLETE replay;
- запрет FULL без проверенного GPU SAMPLE и visual review;
- разные FPS и настоящие PTS из FFmpeg/PyAV synthetic integration;
- ByteTrack reset на PTS segment / sparse selection, deterministic CPU replay;
- import → local-track без Torch/YOLO inference;
- notebook validation, очищенные outputs, code artifact hashes и отсутствие media;
- bootstrap из прямой/вложенной распакованной папки Kaggle, неоднозначные input paths;
- folder checkpoint export/restore/import, повреждённая/неполная папка и атомарный сбой export;
- plain/gzip/auto-unpacked evidence, обязательная проверка content SHA256,
  bundled plain evidence и checkpoint после автоматической распаковки gzip;
- private synthetic quality generation и исключение изображений из export.
- heartbeat при блокирующем вызове и перенаправлении stdout, остановка потока
  и failed phase при ошибке, точные decode/inference/commit counters, прогресс
  прерывания, запрет image arrays в telemetry и отключаемый вывод без изменения detections.
- Kaggle download ZIP с directory entries и отказ для traversal directory entry.

Источники/движок сохранены по [preserved_hashes.json](artifacts/detection/preserved_hashes.json).
Проверка исходной подготовки code package 0e05589: [verification.json](artifacts/detection/verification.json).
Проверка полученного SAMPLE: [sample validation](artifacts/detection/sample_review/validation.json).
Фактический runtime: [runtime.json](artifacts/detection/runtime.json).

## Команды и граница этапа

Подробная приватная загрузка, setup, export и resume:
[KAGGLE_SETUP.md](KAGGLE_SETUP.md).

```powershell
.venv/Scripts/shadowtrack.exe detection prepare
# Ручная private загрузка и запуск только SAMPLE в Kaggle.
.venv/Scripts/shadowtrack.exe detection import --source C:/PRIVATE_DOWNLOADS/sample_checkpoint_folder
# Указать imported run_id и выбранную configuration_name в configs/local_track.toml.
.venv/Scripts/shadowtrack.exe local-track --config configs/local_track.toml
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/detection/tests.xml
```

Приёмка реализации, CPU-контрактов и целостности реального GPU SAMPLE выполнена.
Качество требует следующего patch и clip review; FULL не утверждён и не запускался.
Полные видео автоматически не проходили YOLO;
Re-ID, global association и frontend не запускались. Работа остановлена здесь.
