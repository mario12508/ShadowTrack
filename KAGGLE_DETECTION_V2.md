# ShadowTrack Detection v2

Статус нового режима: **LOCAL_VALIDATED / GPU_PENDING**. Старый реальный
segmentation SAMPLE сохранён. Новый YOLO Detection на офисных видео агентом ещё
не запускался: в локальном окружении нет torch/CUDA. Скорость и качество v2 нельзя
приравнивать к прошлым 17.50/13.24 predict FPS или объявлять лучше конкурентов.

## Основной результат и формат

Основной артефакт — `chunks/<config_hash>/camera_<id>/<segment>/<range>/frames.jsonl.gz`
и manifests. **Каждый обработанный кадр получает запись**, включая отсутствие
людей. FULL выбирает все 144 110 исходных кадров одной выбранной конфигурацией.
Ни stride, ни пропуск пустых кадров, ни timestamp через frame_index/FPS не используются.

Schema v2 сохраняет camera_id, source_frame_index, неизменный raw_media_pts,
timestamp_segment_id, bbox_xyxy, confidence, class_id=0, model_hash/config_hash,
task=detect, processed=true, scene_config_hash. У пустого raw кадра:
`frame_status="empty"`, `detections=[]`. Если все raw кандидаты отклонены сценой,
кадр всё равно содержит эти кандидаты и `frame_status="detections"`.
Формальная схема: [detection_frame_v2.schema.json](schemas/detection_frame_v2.schema.json).

В каждом raw detection есть `scene_filter`: accepted/rejected/uncertain,
reasons, rule_version, config_hash. Raw bbox не удаляются; crop-кандидаты имеют
`inference_source="far_crop"` и исходный crop_xyxy. Дубли сохраняются с причиной
и original detection index. В v2 масок нет; schema v1 segmentation по-прежнему
требует корректную маску и импортируется прежним путём.

MP4, JPEG и видео с локальными ID — производные артефакты. Для будущего трекинга
нужно сохранять detection chunks и manifests, а не только скачать MP4.

## Локальная подготовка

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,detection-cpu,notebooks]"
.venv/Scripts/python.exe scripts/build_kaggle_detection_v2.py
.venv/Scripts/shadowtrack.exe detection prepare-v2
```

`prepare-v2` собирает ZIP и такую же распакованную папку в
`artifacts/detection/v2/portable/`. Внутри source code, два notebook, configs,
schema/license и **копия неизменного** `input_evidence/`. Исходные MKV, веса и
офисные изображения в пакет не входят. Старый evidence не пересчитывается и не
перезаписывается. Для v2 identity исходных данных отделён от producer code hash:
смена кода не превращает прежние проверенные PTS в другую запись видео.

Результат `code_artifact.json` содержит filename, SHA256 ZIP и code hash.
Повторную подготовку того же snapshot выполняйте в новый `--output-dir`.
Никакая команда проекта не загружает записи в облако автоматически.

## Модели и версии

Официальные pretrained detection weights закреплены в
[detection_v2_lock.json](configs/detection_v2_lock.json):

- `yolo11n.pt`: 5 613 764 bytes,
  SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- `yolo11s.pt`: 19 313 732 bytes,
  SHA256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.

Источник — release `ultralytics/assets/v8.3.0`. Ultralytics 8.3.161, PyAV 14.2.0,
OpenCV 4.11.0.86. Используется предустановленный CUDA torch Kaggle; notebook не
заменяет его CPU wheel. Лицензия моделей/Ultralytics — AGPL-3.0 либо отдельные
Enterprise terms; текст закреплённой лицензии включён в code bundle.
См. [официальные модели YOLO11](https://docs.ultralytics.com/models/yolo11/).

Новые веса уже скачаны локально только для проверки pin, без загрузки в torch и
без inference. На Kaggle notebook может скачать только эти известные маленькие
публичные файлы с проверкой bytes/SHA256. Альтернатива — прикрепить оба `.pt`
вручную в Private Dataset и отключить `ALLOW_PINNED_WEIGHTS_DOWNLOAD`.

## Запуск Kaggle по шагам

1. Вручную обновить Private tools Dataset целым новым `prepare-v2` bundle.
   Kaggle распакует ZIP; CODE_DIR указывает папку, не архив. Подключить Private
   videos Dataset с пятью MKV. Dataset, Notebook, версии и outputs должны
   оставаться Private: MP4, bbox и история движения не являются анонимными данными.
2. Открыть [kaggle_detection_v2.ipynb](notebooks/kaggle_detection_v2.ipynb),
   выбрать GPU accelerator, при скачивании весов включить internet. Указать
   CODE_DIR, VIDEO_DIR, WEIGHTS_DIR; EVIDENCE_DIR по умолчанию равен CODE_DIR.
   Уникальные вложенные папки находятся автоматически. Legacy `.json.gz` или
   распакованный `.json` читается при наличии content SHA256 в manifest.
3. Оставить MODE=BENCHMARK, EXECUTE=True, BATCH_SIZE=8,
   BENCHMARK_CONFIGURATIONS=["nano_640","small_640"]. По умолчанию — 10 секунд
   непрерывного видео с каждой камеры. При наличии времени можно явно добавить
   nano_960/small_960. FP16 используется на CUDA, FP32 на CPU. Для обязательной
   CUDA задать DEVICE="cuda:0". Программа пишет actual device, versions и VRAM.
4. Запустить Run All. Один полный последовательный decode на камеру обслуживает
   обе конфигурации. EOF/PTS валидация по-прежнему проходит всю запись: короткий
   benchmark не означает короткий decode. Число decode frames теперь 144 110,
   а не прежние 288 220. RAM ограничена batch размером, чанки ограничены 128 frames.
5. Посмотреть benchmark.json и `_private_quality/<config>/camera_*/`: три JPEG
   на камеру/config, где зелёный — accepted, оранжевый — uncertain, красный —
   rejected. В notebook изображения inline не выводятся. Проверить реальные
   проходы, дальнюю зону, лампу, ТВ и мебель; больше боксов не равно лучше.
6. После сравнения явно установить MODE="DEMO" и SELECTED_CONFIGURATION,
   например "small_640". Это пример имени, **не объявленный победитель**.
   Перезапустить ячейки. Run key автоматически учитывает code/config/batch/settings,
   чтобы разные задания не перезаписывали checkpoint друг друга.
7. DEMO обрабатывает один непрерывный 30-second interval каждой камеры и сохраняет
   `demo/camera_1_detected.mp4` ... `demo/camera_5_detected.mp4`. В кадре camera,
   bbox/confidence и raw PTS; global IDs нет. Рядом frame sidecars с исходными
   index/PTS/segment и временем только для воспроизведения. Аудио в derived MP4
   не переносится. Не соединяются несвязанные SAMPLE окна.
8. Скачать через Notebook Outputs
   `/kaggle/working/shadowtrack_detection_v2/shadowtrack_detection_demo.zip`.
   ZIP содержит **все detection records**, включая пустые кадры, manifests,
   пять MP4, sidecars, benchmark и report. Private JPEG examples не включаются;
   при необходимости скачайте их отдельно из private outputs.

Сравнение моделей и DEMO никогда не запускают FULL автоматически. Новый benchmark
и качество ещё GPU_PENDING; опубликованных измерений нового detector pipeline нет.

## Scene filter и crop

[scene_filter_v1.json](configs/scene_filter_v1.json) привязан к source SHA256 и
разрешению каждой камеры. Узкие image-space области взяты из проверенных кадров
SAMPLE, а не выданы за метрическую metadata. Лампа камеры 2, часть ложного
настольного футбола и ТВ имеют объяснимые rejection rules; лестница/края/стекло
частично помечаются uncertain. Не применяется отбрасывание всего кадра, широкой
области стола/двери, стоящих людей или людей с перекрытыми ногами. Узкие правила
тоже требуют проверки на настоящих проходах: они не гарантируют нулевых пропусков.

Камера 5 дополнительно получает небольшой crop `[1080,0,1520,380]` из исходных
2592×1944. Координаты bbox переводятся обратно в источник, исходный frame/PTS
не меняется. Совпадающие raw кандидаты сохраняются с duplicate decision.
Скорость/польза crop пока не измерены; проверена механика переноса координат.

В обычном DEMO рисуются accepted боксы. Uncertain/rejected остаются в JSONL.
Локальный ByteTrack исключает rejected, оставляя uncertain для его confidence
stages; поэтому scene uncertainty не равнозначна уничтожению детекции.

## Export, resume и FULL

Каждый чанк имеет атомарную запись, manifest, processed-frame count и SHA256.
COMPLETE появляется только при полном покрытии плана и проверке EOF всех камер.
Ошибка decode инвалидирует весь run. Перепутанные/missing/duplicate frames,
неверный source hash и повреждённый upload не принимаются.

Ячейка `finally` сохраняет ZIP даже для прерванного run. Partial checkpoint можно
возобновить, но нельзя импортировать для tracking до COMPLETE. В новой Kaggle
сессии подключить его как Private Dataset, задать RESUME_DIR распакованной папкой
и прежние code/config/model/settings. Завершённые чанки проверяются и повторный
YOLO для них не выполняется. При восстановлении потерянного DEMO MP4 используются
сохранённые records и CPU decode, без повторного inference.

Перед завершением сессии вручную сохранить Private Version или скачать ZIP.
Наличие файлов в `/kaggle/working` само по себе не гарантирует их сохранения после
остановки/лимита Kaggle. ОС может оборвать выполнение без `finally`.

Если MP4 encoder недоступен или завершается ошибкой, основные detection records
могут иметь COMPLETE, а `demo_videos.json` — FAILED. Это различимые результаты;
не считать такой DEMO готовым видео. PyAV/libx264 проверены локальным тестом,
но новая GPU среда ещё должна пройти реальный запуск.

FULL включается отдельно после успешного DEMO: MODE="FULL", та же выбранная
configuration/batch и FULL_DEMO_DIR с matching COMPLETE DEMO и его MP4/sidecars.
Проверяются source identity, model/config, scene rules и video hashes. FULL
сохраняет **все кадры всех пяти источников, включая пустые**, и не делает автоматически
полноразмерные MP4. Это сохраняет приоритет машинных данных и не задерживает их
из-за демонстрационного рендера.

## CPU ByteTrack и производное видео с локальными ID

```powershell
.venv/Scripts/shadowtrack.exe detection import --source C:/PRIVATE_DOWNLOADS/shadowtrack_detection_demo.zip
# Указать напечатанный imported run_id и выбранную configuration_name:
.venv/Scripts/shadowtrack.exe local-track --config configs/local_track_v2.toml
```

CLI выбирает pinned model lock по schema v1/v2. При необходимости явно передать
`--model-lock configs/detection_v2_lock.json`. Исходные MKV должны быть локально
неизменными; их SHA256 сверяется с audit. Ни import, ни ByteTrack не загружают YOLO.

Шаблон [local_track_v2.toml](configs/local_track_v2.toml) включает derived video.
Указать detection_run, configuration_name и output paths. Для большого FULL
при необходимости отключить `[video].enabled`, сохранив machine tracks первыми.
Видеовыход — `private_outputs/local_track_v2_videos/camera_*_local_*.mp4`.
Отдельный повторный рендер готовых track rows:

```powershell
.venv/Scripts/shadowtrack.exe detection annotate --track-dir artifacts/local_tracks_v2 --output-dir private_outputs/local_track_v2_videos
```

ID имеют namespace camera/segment/start-frame. После source frame gap или смены
timestamp_segment_id tracker сбрасывается. Renderer делает отдельные MP4 для
разрывов, не склеивает их в непрерывное движение. `mask_ref=null` для detection v2;
legacy v1 ссылки на masks сохранены. Source detection_index указывает на исходный
raw кандидат, даже когда соседние кандидаты отвергнуты сценой. Глобальных ID нет.

## Локальная проверка и граница этапа

```powershell
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/detection/v2/tests.xml
.venv/Scripts/python.exe scripts/verify_detection_v2.py
```

CPU integration создаёт пять настоящих synthetic MKV (20/25 FPS, nonzero PTS),
выполняет общий engine с явно synthetic predictor, export, import, реальный
supervision ByteTrack и реальное кодирование MP4 с local IDs. Проверяются frame
counts, пустые кадры, sidecar PTS, отсутствие masks и отсутствие повторного YOLO.
Другие тесты проверяют FULL coverage, boundaries, resume, crop, scene decisions,
испорченный export и независимость машинных records от сбоя MP4.
Это проверка контракта, **не GPU benchmark и не видео настоящих людей от новой модели**.

Результат полной CPU проверки: **127 passed**, без skipped/failures/errors.
`verification.json` отдельно сверяет исходники audit/sync, массивы времени,
неизменный evidence, сохранённый GPU SAMPLE, ZIP и каждый файл нового code bundle.
Рядом с ZIP создаётся файл `.sha256`; проверка не перезаписывает старый GPU отчёт.

Не запускались FULL, Re-ID, global association, BEV или frontend. Исходные MKV,
Time Alignment Engine, старый GPU SAMPLE и evidence не изменены. Старый
[segmentation notebook](notebooks/kaggle_detection.ipynb) остаётся работоспособным.
