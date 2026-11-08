# Запуск и передача ShadowTrack разработчикам

## Выберите сценарий

**Нужно открыть готовое демо:** код + приватный demo data pack, затем разделы 1–4.
GPU и повторный YOLO не нужны. Это рекомендуемый путь для приёмки и frontend-разработки.

**Нужно пересчитать связи:** тот же комплект. Признаки внешности уже сохранены в
snapshot; кнопка «Пересчитать» использует CPU. Для host-команд см. раздел 5.

**Нужно заново получить ByteTrack/признаки/YOLO:** нужны дополнительные входы,
описанные в последнем разделе. Этот путь значительно дольше простого запуска.

## 1. Что передаёт владелец проекта

Два независимых комплекта:

1. **Код:** актуальный Git-коммит либо новый `shadowtrack_mvp_code_*.zip` с `.sha256`.
   В архиве — исходники, конфиги, frontend, notebooks, тесты и документация.
2. **Приватные данные:** `shadowtrack_demo_data.zip` с `.sha256`. Здесь исходные
   записи и готовые результаты; хранить и передавать отдельно от публичного кода.

Обычный clone не содержит `data/raw/`, `artifacts/mvp/`, `artifacts/detections/`
и ряда других исключённых каталогов. Подготовленная локальная папка владельца
и содержимое Git — разные вещи. `git status --short` показывает незакоммиченные
изменения; source ZIP собирает текущие файлы, включая новые модули.
В истории Git уже есть приватные аудиторские previews: перед публичной публикацией
нужна отдельная проверка состава. Передача команде предполагает приватный канал.

Владелец создаёт пакет локально, без загрузки в облако:

```powershell
python scripts/handoff.py manifest
python scripts/handoff.py pack --output "artifacts/handoff/shadowtrack_demo_data.zip"
```

`manifest` проверяет реальные файлы и пишет их пути, размеры, SHA256, source run
и MTMC revision в `artifacts/handoff/demo_manifest.json`. Для текущего демо —
79 файлов, 1 345 352 709 байт. `pack` создаёт ZIP без сжатия крупных видео и рядом
пишет SHA256. Нужен дополнительный свободный объём примерно 1.35 GB для архива.
Команда не скачивает модели, не делает inference и не публикует записи.

Не копируйте работающую SQLite вместе со случайным `-wal`: в пакет входит
**закрытый** `artifacts/mvp/server/snapshot.sqlite3` и его `snapshot.json`.
Docker volume и `.venv` с компьютера владельца передавать не нужно.

## 2. Куда положить данные на новом компьютере

Установите Python 3.11+ и Docker с Compose. На Windows Docker Desktop должен
быть запущен. Для готового демо локальные FFmpeg, Node.js, CUDA и torch не нужны:
программная среда backend/frontend создаётся в контейнерах.

Откройте терминал **в корне кода**, рядом с `docker-compose.yml`:

```powershell
python scripts/handoff.py restore --source "C:/PRIVATE/shadowtrack_demo_data.zip"
python scripts/handoff.py check
```

Для Linux/macOS при необходимости используйте `python3` вместо `python`.
Утилита использует только стандартную библиотеку — `pip install` для неё не нужен.

Папки после восстановления:

```text
data/raw/
  camera_new_1.mkv ... camera_new_5.mkv     исходные файлы, не переименовывать

data/metadata/
  office_camera.png                       план организаторов

artifacts/mvp/server/
  snapshot.sqlite3                        готовая закрытая база с данными и признаками
  snapshot.json                           её SHA256 и размер

artifacts/mvp/media/
  source/                                 браузерные MP4 пяти камер
  local/                                  MP4 с историческими локальными ID
  highlights/ и прочие индексированные media/sidecars из пакета

artifacts/mvp/local_tracks/
  manifest.json
  camera_1_tracks.jsonl.gz ... camera_5_tracks.jsonl.gz

artifacts/sync/
  time_alignment.json
  camera_1_frame_times.npz ... camera_5_frame_times.npz
```

JSONL.GZ нужны для покадрового экспорта: одной SQLite для этого endpoint недостаточно.
MKV нужны для исходных кадров/кропов; одного MP4 тоже недостаточно.
`time_alignment.json` передаётся вместе со всеми NPZ, на которые он ссылается.

`restore` проверяет размеры, SHA256 и пути, не заменяет несовпадающие существующие
файлы. При прерывании повторите ту же команду — совпадающие файлы используются снова.
Если получен `Existing file differs`, создайте новую рабочую папку для этого набора;
не затирайте старые исходники. Обычная ручная распаковка также допустима в пустую
папку, но внутри проекта должны получиться именно `data/` и `artifacts/`, без
лишнего уровня `shadowtrack_demo_data/data/`.

## 3. Запуск готового приложения

```powershell
docker compose up --build -d
docker compose ps
```

Frontend: http://localhost:3000. API: http://localhost:8000/docs.
Готовность: http://localhost:8000/api/health → `READY`, `cameras: 5`.
Первый build требует интернета для программных зависимостей. Полный CV pipeline
на старте не запускается. Никакой Kaggle token или внешний сервер БД не нужен.

Данные serving SQLite находятся в named volume `shadowtrack_state`.
При первом запуске backend импортирует закрытый snapshot. Браузер читает видео
из `artifacts/mvp/media/`, а кадры и ID — из сохранённых данных.

Чтобы остановить:

```powershell
docker compose down
```

Обычная остановка сохраняет volume. Не добавляйте `-v`, если хотите сохранить
серверные пересчёты и ручные решения. Новая версия кода: `docker compose up --build -d`.
Новый snapshot при уже работающем backend: `docker compose restart backend`.
При изменённом SHA256 snapshot он заменит серверную базу при старте; перед этим
сохраните нужный `/api/mtmc/export`.

## 4. Проверка после запуска

1. В «Мониторе» выбрать камеру на вкладке «Карта».
2. Открыть «Камеры», Play/Pause, перемотать — проверить несколько ракурсов.
3. Открыть «Люди» — точки и список ID, выбрать и снять выбор человека.
4. В «Треках» найти ID 002: в текущем наборе одна карточка с семью фрагментами камер 3/5.
5. В «Аналитике» проверить 281 эпизод, из них 30 межкамерных.
6. Скачать `/api/exports/episodes.csv` и `/api/exports/tracks.json`.
   CSV — по одной строке на эпизод; JSON — все 144 110 кадров, включая пустые.

ID/числа относятся к текущему snapshot. Новый pipeline или набор может дать
другие результаты. Это не число физических людей и не гарантия точности всех ID.

## 5. Разработка без Docker и host-пересчёт

Нужны Python 3.11+, FFmpeg/ffprobe в PATH и Node.js 24 с npm.
Не переносите чужую `.venv` или `node_modules` — создайте свои:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,web,detection-cpu,sync]"
python scripts/handoff.py init-local
.venv/Scripts/python.exe -m shadowtrack mvp serve --host 127.0.0.1 --port 8000
```

`init-local` делает host-копию из закрытого snapshot и отказывается затирать
существующую `artifacts/mvp/shadowtrack.sqlite3`. Он нужен для первого host-запуска;
Docker его не требует. Во втором терминале из корня проекта:

```powershell
cd frontend
npm ci
npm run dev
```

Linux/macOS: замените `.venv/Scripts/python.exe` на `.venv/bin/python`.
Обычный MTMC использует признаки из базы, torch не нужен:

```powershell
.venv/Scripts/python.exe -m shadowtrack mvp mtmc
.venv/Scripts/python.exe -m shadowtrack mvp publish
docker compose restart backend
```

Host- и Docker-базы различны. `publish` передаёт host-результат в сервер через
snapshot; пересчёт на странице «Связи» меняет native Docker-базу. Не запускайте
оба варианта одновременно над одним рабочим комплектом. Для текущего MTMC
выбран timeout 120 s; один ENV-параметр не меняет версию правил сохранённой базы.

Проверки разработчика:

```powershell
.venv/Scripts/python.exe -m pytest -q
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

E2E использует установленный Chrome (`channel: chrome`). Тесты, которым нужны
реальные результаты, могут быть пропущены без расширенного набора проверки.
Полный historical verifier требует также baseline DB и контрольные artifacts;
они не обязательны для запуска демо и не входят в минимальный пакет.

## Типовые проблемы

- **Нет snapshot.json / snapshot.sqlite3:** передан только код. Получите data pack;
  запуск `mvp publish` на машине без подготовленной host-базы проблему не решит.
- **Local time mapping checksum mismatch / нет NPZ:** восстановите весь sync-комплект.
- **Камера не загружается / Media not yet available:** не хватает MP4 из `media/`
  или нарушена вложенность папок. MKV не заменяет браузерный MP4.
- **Покадровый экспорт недоступен:** нужны все пять JSONL.GZ и их manifest.
- **SHA256 mismatch:** файл повреждён или относится к другому набору. Не обходите
  проверку; повторно получите соответствующий оригинал.
- **409 на пересчёте:** другая задача ещё выполняется; прогресс — `/api/mtmc/progress`.
- **Долго считается:** пересчёт CPU под нагрузкой занимал около 2.5 min; в UI есть
  этапы и heartbeat, nginx ждёт до 300 s. Это не запуск YOLO.
- **Изменения host-базы не видны:** нужен `mvp publish`, затем restart backend.
- **Нет нового модуля после clone:** исходники не были включены в переданный commit;
  получите полный актуальный source ZIP либо корректный коммит.
- **Порты 3000/8000 заняты:** остановите прежний экземпляр либо согласованно измените
  порты Compose и proxy. Для диагностики: `docker compose logs --tail 60 backend frontend`.

## Данные для воспроизведения обработки

Эти материалы нужны **сверх готового демо**, если разработчики хотят восстановить
предыдущие стадии или заново получить признаки:

- `artifacts/detection/input_evidence/`: `source_manifest.json` и пять сохранённых
  файлов frame PTS. Из этого набора имена — `camera_1_frames.json` … `camera_5_frames.json`.
- Полный экспорт YOLO либо валидированная папка
  `artifacts/detections/3f764abf202cfb544e6f6a67/` **целиком**: manifests, COMPLETE,
  benchmark и все chunks, не только MP4/галерея.
- Для повторной проверки аудита — `artifacts/inspect/` и полный `artifacts/sync/`.
- Для forensic-проверок текущего алгоритма — baseline/control и JSON/XML отчёты
  в `artifacts/mvp/mtmc/`; для provenance признаков — `artifacts/mvp/reid/` и
  `artifacts/mvp/mtmc/{appearance,normalized_appearance}/`.
- Для нового извлечения Re-ID — pinned OSNet `.pth` в `.tools/models/` с именем и
  SHA256 из `configs/reid_model_lock.json`; нужны CPU torch и исходные видео.
- Для нового YOLO — отдельный Kaggle GPU workflow, pinned веса и конфиги из
  `configs/detection_v2_lock.json`. Это не часть обычного запуска сайта.

На чистой отдельной копии, куда положены MKV, evidence, sync и FULL:

```powershell
.venv/Scripts/python.exe -m shadowtrack mvp prepare --source "PATH_TO_FULL_EXPORT"
```

Команда проверяет детекции, создаёт локальные треки/базу и браузерные MP4 на CPU.
Это долгая подготовка; готовый data pack позволяет её пропустить. `--render`
добавляет локальные аннотированные видео и нужен только по явному решению.
Для восстановления глобального MTMC на новой базе импортируйте сохранённые
Re-ID embeddings либо выполните отдельную подготовку:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[reid]"
$weights = (Get-Content configs/reid_model_lock.json -Raw | ConvertFrom-Json).weights_filename
.venv/Scripts/python.exe -m shadowtrack mvp reid --weights ".tools/models/$weights" --device cpu
.venv/Scripts/python.exe -m shadowtrack mvp appearance
.venv/Scripts/python.exe -m shadowtrack mvp appearance-normalize
.venv/Scripts/python.exe -m shadowtrack mvp mtmc
.venv/Scripts/python.exe -m shadowtrack mvp publish
```

Для готового demo snapshot эти команды не требуются. Не смешивайте оригинальные
видео, detection run, локальные треки и embeddings разных запусков.
Исходные MKV и сохранённые raw PTS остаются неизменными.
