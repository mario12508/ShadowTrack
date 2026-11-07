# ShadowTrack MVP — локальный запуск

Работает на CPU. YOLO повторно не выполняется. Используются FULL schema v2,
144 110 исходных кадров и сохранённые raw PTS. Данные и видео не публикуются.

## Уже подготовленная рабочая копия

```powershell
docker compose up --build -d
```

Интерфейс: http://localhost:3000 — FastAPI: http://localhost:8000/docs.
Подготовленные SQLite, машинные треки и видео лежат в `artifacts/mvp/` и
подключаются томом. Контейнеры не пересчитывают pipeline при старте.

## Подготовка на другом компьютере

Сохранить исходные `data/raw/camera_new_1.mkv` … `camera_new_5.mkv`, прежние
`artifacts/detection/input_evidence/` и `artifacts/sync/` вместе с NPZ. Эти файлы
не меняются. Нужны Python 3.11+ и FFmpeg/ffprobe; Windows-обёртки могут использовать
`.tools/ffmpeg.exe`. Backend и frontend исходники входят в репозиторий.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,web,detection-cpu,sync,notebooks]"
.venv/Scripts/shadowtrack.exe mvp prepare --source C:/path/to/shadowtrack_full.zip --render
docker compose up --build -d
```

Поддерживается как внутренний detection export, так и внешний ZIP загрузки Kaggle
с одним `shadowtrack_detection_full.zip`. Проверяются COMPLETE, все manifests,
SHA256, source identity, точный порядок кадров, PTS и границы сегментов.
Детекции неизменны. ByteTrack и SQLite записываются отдельно.

Для явной подготовки внутри Docker: скопировать `.env.example` в `.env`, задать
`SHADOWTRACK_FULL_ZIP` абсолютным путём и выполнить:

```powershell
docker compose --profile tools run --rm prepare
docker compose up --build -d
```

Рендер использует два CPU worker, не GPU. Возобновление повторно использует
готовые камеры после проверки SHA256. Если прерван незавершённый сегмент, камера
пересчитывается из сохранённых треков, без YOLO. Видео H.264, VFR с исходными PTS,
локальные ID и отдельные frame sidecars. Файлы разделены на timestamp segments:
полные пять камер покрываются без склейки недоказанных разрывов.

## Без Docker, для разработки

```powershell
.venv/Scripts/shadowtrack.exe mvp serve --host 127.0.0.1 --port 8000
cd frontend
npm ci
npm run dev
```

Proxy Vite и nginx направляет `/api` к FastAPI. Большие MP4 отдаются с HTTP Range,
браузер загружает только нужные данные. В interface выбранная камера воспроизводится
полностью; другие ракурсы доступны на карте и в настоящей сетке видеоплееров.
В маленьких thumbnails обновляются кадры с шагом 5 секунд, без декодирования всех
пяти native видео одновременно. Выбор bbox открывает историю локального трека.

## Время, эпизоды и глобальные ID

По умолчанию общий пульт использует время относительно первого кадра каждой
записи. Это не физическая синхронизация. Режим общего времени показывает только
поддержанные участки; камеры без привязки явно недоступны. Исходный Time Alignment
Engine и его доказательства остаются неизменными.

Ручные offsets сохраняются отдельно через Settings с interval, uncertainty,
checked/unchecked и описанием общего события. Непроверенное значение не применяется.
После изменения привязки межкамерные ассоциации инвалидируются и пересчитываются
явно. Никакие ручные identity labels не создаются.

Таймаут эпизода — **120 секунд**. При первой подготовке он задаётся через
`SHADOWTRACK_EPISODE_TIMEOUT` в пределах 60–120; подготовленная база сохраняет
свою версию правил для воспроизводимости. ByteTrack использует короткий lost buffer 45; это другая
задача. Повторное появление после завершённого эпизода не возвращает его старый ID.
Локальные эпизоды считаются в пределах доступного временного сегмента. Завершение
глобального эпизода требует поддержки времени и покрытия всех пяти камер;
неизвестный участок не считается доказанным отсутствием человека.

503 локальных фрагмента ByteTrack не равны 503 людям. Глобальные эпизоды и число
физических людей — отдельные понятия. Неподтверждённый global ID не рисуется.

## Настоящий Re-ID

Используется author-published OSNet x0.25, обученный на MSMT17, не случайные
embeddings и не ImageNet initialization. [Репозиторий автора](https://github.com/KaiyangZhou/deep-person-reid)
и [веса автора](https://huggingface.co/kaiyangzhou/osnet) имеют MIT license.
Commit, SHA256 архитектуры, weights SHA256 и preprocessing закреплены в
`configs/reid_model_lock.json`; лицензия включена в `src/shadowtrack/mvp/vendor/`.

CPU подготовка (web-серверу torch не нужен):

```powershell
.venv/Scripts/python.exe -m pip install "torch==2.6.0"
shadowtrack mvp reid --weights PATH_TO_PINNED_OSNET_PTH --device cpu --samples 4
shadowtrack mvp associate
```

Выбираются несколько качественных наблюдений на трек с разными frame indices,
проверяется raw PTS, source SHA256, bbox, нормализация и согласованность внешности.
Ассоциация учитывает похожесть, конкурирующие кандидаты, время, качественную
смежность ракурсов и конфликты одновременных наблюдений. Метрической геометрии нет.
Недостаток временных доказательств оставляет `GLOBAL_ASSOCIATION_PENDING`.

Опциональный Kaggle путь без YOLO:

```powershell
shadowtrack mvp reid-crops --samples 4
# Вручную загрузить папку crops в PRIVATE Dataset.
# Запустить notebooks/kaggle_reid.ipynb с версионированным MVP code bundle.
shadowtrack mvp embeddings-import --source C:/path/to/shadowtrack_embeddings.zip
shadowtrack mvp associate
```

Notebook читает распакованные папки. Общий encoder импортируется из репозитория;
другой реализации внутри notebook нет. Выход — NPZ + manifest, без изображений.
Сохранить Private Version или скачать ZIP перед завершением Kaggle session.

Для этого опционального пути `scripts/build_mvp_bundle.py` создаёт версионированные
ZIP и папку в `artifacts/mvp/portable/`. В них код и дизайн assets, без исходных
записей, кадров наблюдения, весов и embeddings. Для нового локального развёртывания
приватные raw MKV, сохранённые evidence и sync arrays требуются отдельно.

## Проверки

```powershell
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/mvp/tests.xml
cd frontend
npm run build
npm run test:e2e
```

E2E запускает изолированный headless Chrome на локальном приложении и настоящем
FULL: переключает карту, воспроизводит H.264, выбирает bbox и открывает track detail.
Скриншоты и trace содержат приватные кадры, поэтому сохраняются только в
`private_outputs/mvp_browser_smoke/`, вне исходников. Порт localhost не публикуется.
SQLite WAL выбран для одного компьютера: миграция schema_version=1 в `mvp/db.py`,
индексы по camera/time/track. PostgreSQL для этого однопользовательского MVP не нужен.
