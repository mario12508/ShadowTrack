# Global Episodes — запуск и просмотр

Эта инструкция относится к рабочей папке после восстановления приватных данных.
Чистый Git clone не содержит FULL/базу/видео. Сначала выполните
[инструкцию передачи и запуска](MVP_SETUP.md). Обычный запуск и cached-пересчёт
выполняются на CPU без YOLO/torch.

```powershell
.venv/Scripts/python.exe -m pip install -e ".[web,detection-cpu,sync]"
# Пересчёт host database из готового cache после изменения алгоритма/config:
.venv/Scripts/shadowtrack.exe mvp mtmc
# Передать закрытую согласованную database Docker backend:
.venv/Scripts/shadowtrack.exe mvp publish
docker compose up --build -d
# Если backend уже запущен после нового publish:
docker compose restart backend
```

Backend SQLite находится в native volume `shadowtrack_state`, не в Windows/Linux
shared WAL. `mvp publish` создаёт snapshot с SHA256 в `artifacts/mvp/server`.
Запуск с тем же snapshot сохраняет native пересчёты/reviews. **Новый host publish
заменяет серверную database при рестарте**: предварительно сохраните нужные
серверные решения через `/api/mtmc/export`. Native volume при обычном restart
удалять не нужно.

Повседневный пересчёт: «Пересчитать» на `/associations` или POST
`/api/mtmc/recompute`. Это CPU/cache, без inference, ByteTrack или рендера;
В последнем нагруженном прогоне — около 2.5 min; heartbeat/стадии видны в UI.
Nginx ждёт ответ до 300 s, повторный запрос во время работы получает 409. Ручной review не требуется.

- [Monitor](http://localhost:3000/monitor): сетка пяти камер и Play работают
  по исходной шкале записей, без переключений clock mode.
- [ID 267, камера 4](http://localhost:3000/monitor?camera=4&time=1035&clock=estimated&gtime=1035&global=G0211&track=a6ceb1665bfaf1d2c392).
- [Gantt](http://localhost:3000/timeline): поиск `ID 267`, камеры 4/3/5,
  клик интервала открывает Monitor.
- [Архив](http://localhost:3000/tracks): ID 002 — одна девушка, семь фрагментов камер 3/5.
- [Связи](http://localhost:3000/associations): признаки, причины и пересчёт.

Клик bbox открывает наблюдения ID. Из карточки участников откройте камеры 3 и 5:
ID/цвет сохраняются, открывается исходное время участника. Над человеком —
основной глобальный ID, нумерация с 001 по первому появлению.
Архив содержит одну карточку на глобальный эпизод, source fragments — в истории. Ручной review необязателен. На карте движутся точки, одна на global ID.
Положение на схеме приблизительное; clock API не управляет playback.

`configs/mtmc.json`: cosine 0.82, tolerance 3 s, topology, отдельные transit windows.
`configs/scene_layout.json`: hash-bound static landmarks, приблизительные проекции и image portals.
Raw PTS и исторический Time Alignment Engine не переписываются.

Temporal cache уже подготовлен: 3 427 исходных samples, 29 минут вместе с decode.
Дополнительно подготовлен scene-balanced cache: те же приватные crops и pinned OSNet,
нейтральный участок пола каждой камеры, 61.172 s CPU encoder. Обычный запуск
читает таблицу `appearance_normalized`; исходная `appearance_sample` сохранена.
Для нового набора после `mvp appearance` есть отдельная явная команда:
`shadowtrack mvp appearance-normalize`. Она не нужна для текущего пересчёта,
не скачивает модель и не запускается внутри web container. Manifest/SHA256
проверяются перед атомарным импортом; завершённый encoding можно импортировать
после прерывания, без повторного inference.
**Не запускайте подготовку для обычного пересчёта.** Для нового набора есть явная
команда `shadowtrack mvp appearance`, которой нужны pinned OSNet и CPU torch.
Это отдельная подготовка, не часть web startup.

```powershell
Invoke-WebRequest http://localhost:8000/api/mtmc/export -OutFile artifacts/mvp/mtmc/export.json
Invoke-WebRequest http://localhost:8000/api/exports/tracks.json -OutFile artifacts/mvp/mtmc/all_frame_tracks.json
# Необязательно, только короткий private preview:
.venv/Scripts/shadowtrack.exe mvp mtmc-preview --episode G0211 --seconds 6
# После host-регистрации preview: mvp publish и restart backend.
```

Mapping schema v2 содержит source parent, derived tracklet и frame range.
All-frame export сохраняет 144 110 кадров, включая пустые, исходные index/PTS и
segments; global ID выбирается по конкретному кадру slice. Не смешивайте revisions.
JSONL.GZ ByteTrack не изменены.

```powershell
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/mvp/mtmc/tests.xml
npm --prefix frontend run build
npm --prefix frontend run test:e2e
.venv/Scripts/python.exe scripts/evaluate_mtmc_control.py
.venv/Scripts/python.exe scripts/verify_mtmc.py
.venv/Scripts/python.exe scripts/build_mvp_bundle.py
```

Архив/SHA256: `artifacts/mvp/portable/code_artifact.json`. Source bundle не содержит
базы, frames, crops, embeddings, weights и видео. Контрольные labels — только
для оценки, не вход engine. Baseline: `artifacts/mvp/mtmc/control/baseline.sqlite3`.


Вкладки: «Карта» — навигация по камерам, «Люди» — точки глобальных эпизодов.
Видео сохраняет позицию при переключении этих двух режимов. «Архив» приостанавливает
воспроизведение. При ошибке media доступна кнопка повторной загрузки.

`GET /api/episode-counts?time=129` возвращает число начавшихся/активных/потерянных/
завершённых эпизодов к выбранному времени; без `time` — состояние на конце записи.
`GET /api/exports/episodes.csv` и `.json` — по одной строке на episode ID с камерами
и временными границами. `identity_count=null`: число людей не измерено.
`GET /api/mtmc/progress` — этап, прошедшее время, heartbeat, processed pairs.
Оба маршрута POST `/api/mtmc/recompute` и `/api/associations/recompute` используют
общую блокировку. При параллельном пересчёте возвращается 409.

Проверка всех девяти требований и физических ограничений групп:
`.venv/Scripts/python.exe scripts/check_mvp_acceptance.py`.
