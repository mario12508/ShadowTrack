# ShadowTrack

Рабочий CPU MVP: **FastAPI + React/TypeScript + Docker Compose**. Импортирован
реальный FULL: 144 110 кадров, 1 135 чанков, 136 020 raw bbox. ByteTrack обработал
все пять камер; локальные фрагменты, видео и база хранятся отдельно от детекций.

```powershell
docker compose up --build -d
```

Открыть http://localhost:3000, API http://localhost:8000/docs.
Камеры переключаются по карте из предоставленных макетов. Есть настоящие видео,
выбор bbox, история треков, аналитика, калибровка и экспорт. Завершение эпизода —
120 секунд отсутствия. Глобальные ID выдаются только при поддержанных ассоциациях;
частичная синхронизация не выдаётся за общую точную шкалу.

[Установка и подготовка](MVP_SETUP.md), [сценарий защиты](DEMO_SCRIPT.md).
Исторические этапы подготовки Detection v2 описаны ниже.

Новый основной pipeline: **YOLO Detection schema v2**, Kaggle batch GPU inference
и CPU ByteTrack из сохранённых записей. Главный результат — полные frame records,
включая пустые кадры; MP4 с bbox или локальными ID — производный артефакт.
[Инструкция v2](KAGGLE_DETECTION_V2.md),
[новый notebook](notebooks/kaggle_detection_v2.ipynb).

```powershell
.venv/Scripts/shadowtrack.exe detection prepare-v2
# После ручного приватного BENCHMARK -> выбранного DEMO в Kaggle:
.venv/Scripts/shadowtrack.exe detection import --source C:/PRIVATE_DOWNLOADS/shadowtrack_detection_demo.zip
# В configs/local_track_v2.toml указать imported run_id и configuration_name:
.venv/Scripts/shadowtrack.exe local-track --config configs/local_track_v2.toml
```

Новый GPU benchmark и реальные пять MP4 от detection v2 пока **GPU_PENDING**.
Полная CPU цепочка export -> import -> ByteTrack -> MP4 с локальными ID проверена
на пяти синтетических MKV. FULL отдельно сохраняет детекции всех исходных кадров,
автоматически не запускается. Старый segmentation SAMPLE и evidence не изменены.

Проект для пяти офисных MKV хакатона NTechLab. Этап `inspect` сохраняет параметры
потоков, полный video probe, проверку PTS, 20 preview на камеру, контактные листы,
JSON и HTML. Отдельный Time Alignment Engine описан в разделе ниже.
Старый Kaggle segmentation pipeline сохранён для воспроизводимости полученного
GPU SAMPLE. Новый основной режим Detection v2 пока GPU_PENDING; FULL не запускался.
ReID, обучение и global association отсутствуют. См.
[KAGGLE_SETUP.md](KAGGLE_SETUP.md) и [DETECTION_REPORT.md](DETECTION_REPORT.md).

Результаты реальных данных: [DATA_AUDIT.md](DATA_AUDIT.md),
[HTML report](artifacts/inspect/report.html), [JSON stats](artifacts/inspect/stats.json).

## Установка

Требуются Python 3.11+ и FFmpeg с `ffmpeg` и `ffprobe` в PATH.
Python-зависимость приложения — Pillow; pytest нужен для тестов.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/shadowtrack.exe inspect --config configs/inspect.toml
```

Если FFmpeg не в PATH, задайте абсолютные пути:

```powershell
$env:SHADOWTRACK_FFMPEG = (Resolve-Path .tools/ffmpeg.exe).Path
$env:SHADOWTRACK_FFPROBE = (Resolve-Path .tools/ffprobe.exe).Path
.venv/Scripts/shadowtrack.exe inspect --config configs/inspect.toml
```

В подготовленном рабочем каталоге уже доступны `.venv` и `.tools`;
`scripts/inspect.ps1` настраивает эти пути в текущем процессе:

```powershell
./scripts/inspect.ps1
./scripts/inspect.ps1 -Strict
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/tests.xml
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/shadowtrack inspect --config configs/inspect.toml
.venv/bin/python -m pytest -q
```

FFmpeg распространяется отдельно; загруженный для этого аудита ZIP и EXE находятся
в `.tools`, исключённом из git. Информация о полученных инструментах сохранена
в `artifacts/environment.json`. `.venv` в этом сеансе создана на Python 3.12.14
с доступом к пакетам предоставленного runtime; обычная чистая установка выше
не зависит от этого окружения.

## Входы и выходы

```text
configs/inspect.toml             параметры аудита
src/shadowtrack/                 CLI, probe, previews, report
scripts/                        запуск, чтение документов, исследование событий
tests/                          unit и интеграционные тесты
data/raw/camera_new_1..5.mkv      реальные исходники, только чтение
data/metadata/office_camera.png  исходный план, без сгенерированной калибровки
artifacts/inspect/stats.json     измерения, SHA256, команды, предупреждения
artifacts/inspect/report.html    автономный локальный HTML
artifacts/inspect/camera_*/      контактные листы, preview, все PTS и логи
artifacts/events/               дополнительная проверка общих событий
artifacts/sources/              извлечённый текст DOCX/PDF и provenance
DATA_AUDIT.md                    выводы, наблюдения, гипотезы, ограничения
```

`camera_01` в выходах — порядковый номер файла в отсортированном списке.
Для текущих имён он совпадает с номером `camera_new_1`; в общем случае это
не доказательство соответствия камеры значку на плане. Исходное имя есть в JSON.

CLI поддерживает `--raw-dir`, `--metadata-dir`, `--output-dir`, `--previews 16..24`,
`--expected-cameras`, `--timeout-seconds`, `--strict`. Явные аргументы перекрывают
TOML; пути TOML считаются от его каталога, пути аргументов — от текущего каталога.
Пустой или отсутствующий каталог, нечитаемое видео и ошибки декодирования дают
отчёт со статусом `error` и exit code 1. Отсутствующий план и расхождения FPS —
`warning`; с `--strict` также exit code 1. Ошибки конфигурации дают exit code 2.

`--review-file` подключает качественные наблюдения из JSON только при совпадении
SHA256 всех видео и metadata. Стандартный TOML указывает на
`artifacts/visual_review.json`: это результат аудита, не исходная metadata.
Для другого набора данных удалите строку или подготовьте новый review;
несовпадение хешей даёт предупреждение и исключает старые выводы из отчёта.

## Методика

`ffprobe -show_format -show_streams` даёт codec, разрешение, time base,
декларированный FPS, start/duration и аудиопотоки. Второй проход
`ffprobe -show_frames` декодирует весь первый видеопоток и сохраняет все
`pts_time`, `best_effort_timestamp_time`, DTS, типы кадров и длительности.
Число кадров — длина полученного списка, не произведение FPS и duration.

Из PTS считаются фактическая средняя частота, распределение интервалов,
дубликаты, обратные шаги и разрывы более 1.5 медианного интервала. Отдельно
выводится частота обычных интервалов без этих разрывов. Разрыв PTS сам по себе
не доказывает, сколько физических кадров потеряно или почему он возник.

Preview равномерно распределяются между первым PTS и точкой за 0.25 s до
последнего PTS. FFmpeg выбирает первый доступный кадр после цели; при разрыве
фактический PTS может быть существенно позже. `previews.json` сохраняет оба
времени, а контактный лист подписан фактическим PTS. Изображения сохраняются
в исходном разрешении, на листе уменьшаются с сохранением пропорций.
`-ss` задаётся относительно начала контейнера, `-copyts` сохраняет исходные
PTS в `showinfo`; это проверено тестом с началом потока на 5 s.

Часы, напечатанные внутри изображения, CLI автоматически не распознаёт.
Их чтение и проверка телевизионных событий описаны отдельно в DATA_AUDIT.
Никакое часовое смещение или исправление drift автоматически не применяется.

Полный прогон декодирует около 150 тысяч кадров 5 МП и может занять несколько
минут на камеру. Таймаут 1800 s относится к одному подпроцессу. Для повторного
создания preview после изменения оформления можно использовать
`python scripts/refresh_previews.py`: скрипт проверяет SHA256 всех источников
против завершённого аудита и использует сохранённые результаты полного probe.

## Воспроизведение проверки событий

После настройки путей FFmpeg:

```powershell
.venv/Scripts/python.exe scripts/sample_events.py --videos data/raw/camera_new_1.mkv data/raw/camera_new_3.mkv --times 0.8 1 1.2 1.4 1.6 1.8 4 4.5 5 5.5 6 6.5 --output artifacts/events/tv_cuts
.venv/Scripts/python.exe scripts/evidence_crops.py artifacts/events/tv_cuts
```

Это извлечение кадров по заданным PTS и просмотр прямоугольника телевизора,
без разметки людей, ID или обучения на проверочном видео. Пиксельные crop-области
в `evidence_crops.py` относятся только к осмотренным исходникам 1 и 3 и не являются
геометрической калибровкой. Все команды уточняющей выборки записаны в manifests.

Скрипт `read_sources.py` читает DOCX через ZIP/XML и PDF через опциональный `pypdf`.
Для повторного извлечения текста установите `python -m pip install pypdf`, затем
передайте `--docx` и `--pdf` с реальными путями. Документы рассматриваются как
источники требований; выполнение всего хакатонного задания ими не запускается.

## Ограничения этапа

Нет метрической привязки плана, distortion coefficients, подтверждённых полигонов
зон или общей временной шкалы всех камер. Контактные листы — выборка, не оценка
recall детектора и не доказательство отсутствия человека между кадрами. Экраны,
отражения, двери, мебель, motion blur и низкая детализация дальних ног требуют
учёта на следующих этапах. После этого аудита проект останавливается.

## Time Alignment Engine — отдельный этап от 7 ноября 2026

Существующие `DATA_AUDIT.md`, `artifacts/inspect` и старые manifests остаются
результатом первого этапа. Sync читает их и пишет только собственные результаты:
[TIME_ALIGNMENT.md](TIME_ALIGNMENT.md), [HTML](artifacts/sync/report.html),
[JSON](artifacts/sync/time_alignment.json). Текущая приёмка данных — **PARTIAL**.

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,sync]"
./scripts/sync.ps1 analyze
./scripts/sync.ps1 report
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/sync/tests.xml
```

`scripts/sync.ps1` подключает `.tools` к текущему процессу. После настройки
`SHADOWTRACK_FFMPEG` и `SHADOWTRACK_FFPROBE` можно непосредственно использовать CLI:

```powershell
.venv/Scripts/shadowtrack.exe sync analyze --stats artifacts/inspect/stats.json --evidence configs/sync_evidence.json --output-dir artifacts/sync --reference-camera 3
.venv/Scripts/shadowtrack.exe sync report --alignment artifacts/sync/time_alignment.json
.venv/Scripts/shadowtrack.exe sync analyze --require-complete
```

У `analyze` exit code 0 означает успешное исследование, включая честный PARTIAL.
`--require-complete` возвращает 1 при неполной калибровке; ошибки входных данных,
хешей или зависимостей дают 2. `--no-audio` явно записывает `NOT_REQUESTED`.
`report` читает готовую калибровку и создаёт HTML/PNG без декодирования видео.
Для sync нужны опциональные зависимости NumPy и Matplotlib.

Сохранённые video PTS используются без нового полного probe. MKV и все источники
проверяются по SHA256. Для аудио выполняются отдельные audio-only probe/decode;
кэш связан с хешами исходника, PCM и audio frame PTS. Не применяется async resampling,
склейка через audio PTS breaks или предположение об одинаковых каналах.

Локальная шкала хранится в `artifacts/sync/camera_*_frame_times.npz`:
`raw_media_pts`, `frame_index`, `relative_capture_time_estimate`, интервальные
границы и номера непрерывных сегментов. Это монотонная оценка, физический масштаб
которой не подтверждён. Исправлены только два проверенных OSD-разрыва камеры 1;
их длительность оценивается по соседнему cadence с границами (0; 1] s и явным
допущением, что OSD в этих местах продолжает идти. Все остальные разрывы
сохраняются. Внутри разрывов raw mapping недоступен. Дубликаты/обратные PTS
не получают вымышленного однозначного raw→frame соответствия.

Глобальная шкала — условные относительные секунды первой эпохи камеры 3, не UTC.
Регрессия работает с интервалами, требует минимум два независимых события,
отдельно оценивает сегменты, регистрирует outliers/residuals и не экстраполирует.
Для drift нужны минимум три события с размахом 60 s; `ZERO_NOT_EXCLUDED` означает,
что ненулевой drift не доказан. Bounds модели не являются статистическим CI.

```python
from shadowtrack.sync import TimeAlignment, map_time

alignment = TimeAlignment.load("artifacts/sync/time_alignment.json")
global_time, uncertainty, status = alignment.map_time("1", 3.0)
# Около 1.317 относительных секунд, interval bounds, SUPPORTED_CONDITIONAL.
assert alignment.map_time("2", 100.0) == (None, None, "unavailable")
assert alignment.map_time("1", 904.0) == (None, None, "unavailable")
local = alignment.map_local_time("1", 904.0)  # отдельная локальная оценка
raw_pts, raw_uncertainty, status = alignment.inverse_map_time("1", global_time)
```

Для частых запросов загружайте `TimeAlignment` один раз. Функция
`map_time(camera_id, raw_pts, alignment_path=...)` — удобная обёртка с загрузкой.
`uncertainty` возвращает lower/upper, radius, source model IDs и допущения.
`unavailable` возвращает `(None, None, "unavailable")`; конфликт путей —
`inconsistent`. Статус всего набора не отменяет проверки конкретного запроса.

### Сбор новых опорных событий

```powershell
./scripts/sync.ps1 collect --request configs/sync_event_requests.json --output-dir artifacts/sync/events/new_session
```

Request задаёт для каждого события `id`, описание и **раздельные raw PTS targets**
по камерам; пример в `configs/sync_event_requests.json`. Можно указать прямоугольник
`crops` для визуального осмотра ориентира. Коллектор сохраняет полные кадры,
showinfo-логи, фактические PTS, хеши, листы и manifest с `checked=false`.
Повторно существующий manifest не перезаписывается. Выборка сама по себе
не подтверждает одновременность и не используется в регрессии.

После просмотра в `configs/sync_evidence.json` добавляется отдельная запись
с `checked=true`, reviewer, описанием физического события, интервальными
`observations` и `source.manifest`/SHA256. Событие должно относиться к одному
реальному ориентиру и не пересекать непроверенный разрыв. В этом сеансе
добавлены четыре проверенные смены плана одного телевизора; прочие кандидаты
сохранены unchecked. OCR и сопоставление людей не используются.

`scripts/import_audit_sync_evidence.py` документирует начальный импорт двух
старых TV events и двух OSD-фактов. Он отказывается перезаписывать существующий
review и применим только к исходным хешам первого аудита. Снимок
`artifacts/sync/preserved_audit_hashes.json` позволяет проверить неизменность аудита.

На завершении Time Alignment Engine работа останавливается. Detection, Re-ID
и global tracking требуют следующей отдельной задачи.

## Предыдущий segmentation pipeline без локального GPU

[notebooks/kaggle_detection.ipynb](notebooks/kaggle_detection.ipynb) импортирует общий
`src/shadowtrack/detection` из распакованной versioned code folder. SAMPLE — 30 секунд на камеру
из разных timestamp segments, две конфигурации YOLO11n-seg (640/960). FULL требует
отдельного проверенного GPU SAMPLE и visual review. Загрузка данных всегда ручная,
Dataset, Notebook и outputs должны оставаться Private.

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,detection-cpu,notebooks]"
.venv/Scripts/shadowtrack.exe detection environment
.venv/Scripts/shadowtrack.exe detection prepare
```

Подготовленные snapshots уже находятся в `artifacts/detection/input_evidence`
и `artifacts/detection/portable`. Они не перезаписываются; для новой версии
укажите свежие `--evidence-dir` и `--output-dir`. Создание ZIP не загружает его
в сеть. Для Notebook используйте распакованные папки кода/evidence/checkpoint
под `/kaggle/input`; ZIP на входе ему не нужен. Private upload через UI или Kaggle CLI,
настройка input-папок и ручной SAMPLE run описаны в
[KAGGLE_SETUP.md](KAGGLE_SETUP.md).

Текущий пакет уже включает `input_evidence/` с обычными JSON без `.gz`:
в Notebook задайте `EVIDENCE_DIR = CODE_DIR`. Для обновления замените tools Dataset
новым пакетом целиком и перезапустите kernel. Bootstrap выводит фактический путь
загруженного кода и очищает кеш старых модулей. Распакованные Kaggle JSON/JSONL
принимаются с проверкой SHA256 содержимого.

`run`, export/import/resume и CPU tracker показывают этапы и heartbeat.
В Notebook: `SHOW_PROGRESS=True`, `PROGRESS_INTERVAL_SECONDS=10`. SAMPLE на этих
данных выполняет **288 220 decode frame checks** и **7 318 YOLO frames** для двух
конфигураций: он полностью декодирует видео даже вне выбранных окон.
Камера/config, raw frame/PTS, decoded/inferred/committed, чанки, устройство,
примерный ETA текущей камеры и длительность этапа видны в консоли. Metadata-only
`progress.json` / `events.jsonl` находятся рядом с run в `telemetry/<mode>/`;
инференс, masks, SHA256, запись и export показаны отдельно.

Сохранение checkpoint перед завершением Kaggle session — в `finally` Notebook;
дополнительно пользователь сохраняет Private Version или скачивает checkpoint-папку.
Ручной export/restore использует общие функции:

```python
from shadowtrack.detection.transport import export_run_directory, restore_checkpoint
export_run_directory(RUN_DIR, RUN_DIR.parent / "manual_checkpoint")
# В новой сессии, в пустой каталог, с теми же source/model/config/code hashes:
restore_checkpoint(RESUME_DIR, RUN_DIR)
```

После скачивания COMPLETE data-only папки:

```powershell
.venv/Scripts/shadowtrack.exe detection import --source C:/PRIVATE_DOWNLOADS/sample_checkpoint_folder
# В configs/local_track.toml заменить SET_IMPORTED_RUN_ID и выбрать configuration_name.
.venv/Scripts/shadowtrack.exe local-track --config configs/local_track.toml
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/detection/tests.xml
```

Локально torch/YOLO не нужны. Import проверяет schema, SHA256, полноту, source
hashes, frame indices, PTS, segments и RLE. ByteTrack обрабатывает frozen detections
по одному чанку и потоково пишет результаты. Исходные PTS и Time Alignment Engine
не изменяются; разрывы сохраняют segment boundaries. Настройка 640 в шаблоне
не объявлена победителем до реального GPU benchmark/частного просмотра.

Текущая приёмка: **SAMPLE_INTEGRITY_PASS / QUALITY_PARTIAL / FULL_NOT_APPROVED**.
Получен реальный Kaggle GPU SAMPLE пользователя, импорт проверен локально.
Разбор качества, времени и рекомендуемых изменений: [SAMPLE_REVIEW.md](SAMPLE_REVIEW.md).
FULL не утверждён; дальнейший YOLO run и tracking автоматически не запускались.
