# Прогресс анализа снимка в интерфейсе — design

*2026-07-04. Триггер: жалоба пользователя — «в процессе анализа снимка надо отображать прогресс в
интерфейсе, а то пользователю не понятно что сейчас происходит и сколько это продлится».*

## Проблема

Job store (`backend/app/jobs/store.py`) и API-контракт (`backend/app/schemas/jobs.py::JobRecord`,
`GET /api/jobs/{jid}`) уже несут поля `progress: float` и `message: str | None`, а фронтенд уже
поллит job каждые 800мс (`frontend/lib/api/hooks.ts::useJob`). Но по факту эти поля никогда не
обновляются во время самой обработки:

- `JobRunner.submit` (`backend/app/jobs/runner.py:12`) ставит `progress=0.05` синхронно при приёме
  задачи и больше не трогает её до `progress=1.0` в `_run` после того, как `fn()` уже целиком
  отработала.
- `message` не выставляется вообще нигде в рабочем состоянии — только `error`-текст при исключении.
- На фронтенде (`frontend/app/page.tsx:104-109`) во время `queued`/`running` рисуется статичный
  плейсхолдер «Анализ снимка…» + одна строка про режим — без процента, этапа или времени.

Пайплайны при этом уже содержат естественные контрольные точки: панорама тайлится
(`for tile in iter_tiles(...)`, `backend/app/pipeline/panorama.py:104`) с заранее известным числом
тайлов, а close-up внутри `_uncertainty` гоняет ensemble из 5 фотометрических пертурбаций
(`backend/app/shlif/uncertainty.py::ensemble_phase_labels`), который и есть доминирующая по времени
часть close-up пайплайна (5× пересегментация против 1× в `analyze_image`). Работа — протянуть
колбэк прогресса через уже существующие точки, не изобретая новую инфраструктуру (SSE/WebSocket не
нужны — 800мс-поллинг достаточно гранулярен для прогресс-бара).

## Согласованный скоуп

Прогресс нужен для **обоих режимов** (крупный план и панорама). Визуально — в месте текущего
плейсхолдера «Анализ снимка…»: прогресс-бар + текст текущего этапа + процент, плюс прошедшее время
и ETA (линейная экстраполяция от скорости прогресса). Топбар (`status-badge`) не трогаем — там
остаётся как есть (пульсирующая точка + «анализ»).

**Вне скоупа:** кнопка отмены анализа; WebSocket/SSE вместо поллинга; правка топбар-бейджа; починка
существующей смежной странности — при `max_workers=1` второй параллельный анализ помечается
`running` сразу при `submit`, хотя реально ещё стоит в очереди пула (это не регрессия от текущей
задачи, просто существующая неточность статуса).

## Архитектура

### Backend: колбэк прогресса, без изменения API-контракта

`JobRecord`/эндпоинт `/api/jobs/{jid}` не меняются — `progress`/`message` уже там. Меняется только
то, что их пишут чаще и осмысленнее.

**`backend/app/jobs/runner.py`** — сигнатура `fn` меняется с `Callable[[], dict]` на
`Callable[[Callable[[float, str | None], None]], dict]`:

```python
def submit(self, jid: str, fn) -> None:
    self._store.set_status(jid, "running", progress=0.05)
    self._pool.submit(self._run, jid, fn)

def _run(self, jid: str, fn) -> None:
    def report(progress: float, message: str | None = None) -> None:
        self._store.set_status(jid, "running", progress=progress, message=message)
    try:
        result = fn(report)
        self._store.set_result(jid, result)
        self._store.set_status(jid, "done", progress=1.0)
    except Exception as e:
        self._store.set_status(jid, "error", message=str(e))
```

Затрагивает 2 существующих теста в `backend/tests/test_jobs.py`, которые сейчас вызывают
`runner.submit(jid, lambda: {...})` / `def boom(): raise ...` — обновить на однoargументные
`lambda report: {...}` / `def boom(report): raise ...` (сигнатура — часть намеренного изменения
контракта, не побочная поломка).

**`backend/app/api/analyze.py`** — `work()` становится `work(report)`, пробрасывает `report` в
`panorama.analyze_panorama(..., on_progress=report)` / `closeup.analyze_closeup(..., on_progress=report)`,
и сам репортит для close-up: `report(0.05, "загрузка изображения")` в начале, `report(0.95,
"сохранение результатов")` перед записью `disp`/`_persist_maps` (это I/O, которое сейчас происходит
уже *после* возврата из `analyze_closeup`, но всё ещё внутри `work()`).

**`backend/app/pipeline/closeup.py`** — `analyze_closeup(rgb, cfg, on_progress=None)`. Фиксированные
чекпойнты между уже существующими последовательными шагами (веса грубые, по относительной
дороговизне — сегментация 1×, ensemble uncertainty ~5×, sort/maps дёшево):

| после шага | progress | message |
|---|---|---|
| `loader.load_talc_unet()` | 0.08 | «загрузка модели талька» |
| перед `analyze_image(...)` | 0.15 | «сегментация фаз» |
| перед `_uncertainty(...)` | 0.30 | «оценка неопределённости» (далее — подпрогресс, см. ниже) |
| после `_uncertainty(...)` | 0.80 | «классификация сорта» |
| после `_sort_card(...)` | 0.88 | «построение карт» |

`_uncertainty(rgb, cfg, on_progress=None)` пробрасывает подпрогресс из ensemble в диапазон
0.30→0.75 (внутри общего интервала до 0.80):

```python
def _uncertainty(rgb, cfg, on_progress=None) -> dict:
    ...
    def on_step(i, total):
        if on_progress:
            on_progress(0.30 + 0.45 * (i / total), f"оценка неопределённости ({i}/{total})")
    u = ensemble_uncertainty(small, cfg, on_step=on_step)
    ...
```

**`backend/app/shlif/uncertainty.py`** — `ensemble_phase_labels`/`ensemble_uncertainty` получают
опциональный `on_step: Callable[[int, int], None] | None = None`, вызываемый после каждой из 5
пертурбаций с `(i, total)` — чисто механический хук, без знания о джобах/долях (масштабирование —
забота вызывающей стороны, как выше в `closeup.py`). `panorama.py` вызывает `ensemble_uncertainty`
потайлово и не передаёт `on_step` — там гранулярность уже даёт цикл по тайлам.

**`backend/app/shlif/tiling.py`** — новая функция `count_tiles(path, cfg) -> int`, повторяющая
границы циклов `iter_tiles` (`image_size` + `decode_factor` + `step`-арифметика) **без декодирования
тайлов** — только для оценки общего числа тайлов заранее:

```python
def count_tiles(path: str | Path, cfg) -> int:
    w, h = image_size(path)
    factor = decode_factor(w, h, int(cfg.max_pixels))
    W, H = w // factor, h // factor
    tile = int(cfg.tile)
    step = max(1, tile - int(cfg.overlap))
    n_y = len(range(0, max(1, H - 1), step))
    n_x = len(range(0, max(1, W - 1), step))
    return n_x * n_y
```

Приближённо (не учитывает отбрасывание тайлов `<8px` на краю) — для прогресс-бара точность до
одного тайла не важна.

**`backend/app/pipeline/panorama.py`** — `_run_panorama(..., on_progress=None)` и
`analyze_panorama(path, cfg, jid, on_progress=None)` пробрасывают колбэк:

- `report(0.05, "загрузка модели")` после `loader.load_talc_unet()`/`loader.load_ore_unet()`.
- Внутри `for tile in iter_tiles(...)`, после `n_tiles += 1`:
  `report(0.10 + 0.80 * min(1.0, n_tiles / total), f"сегментация тайлов ({n_tiles}/{total})")`
  где `total = max(1, count_tiles(path, cfg.tiling))`.
- `report(0.92, "построение карты сорта")` после цикла, перед `aggregate_section`/стежкой overlay.
- В `analyze_panorama`, перед сохранением JPEG: `report(0.97, "сохранение результатов")`.

### Frontend: `AnalysisProgress` вместо статичного плейсхолдера

**`frontend/lib/progress.ts`** (новый, чистые функции — под `node:test`, без React):

- `formatDuration(seconds: number): string` — `"14 с"` / `"1 мин 32 с"`.
- `computeEta(elapsedSec: number, progress: number): number | null` — `null`, если
  `progress < 0.08` (слишком шумно на старте) или `progress <= 0`; иначе
  `max(0, elapsedSec / progress - elapsedSec)`.
- `clampPct(progress: number): number` — `round(clamp(progress, 0, 1) * 100)`.

**`frontend/components/AnalysisProgress.tsx`** (новый): принимает `job: Job | undefined`,
`startedAt: number`, `fallback: string`. Тикает раз в ~500мс (`setInterval` + `Date.now()`) для
прошедшего времени. Рендерит: `.hint` («Анализ снимка…», без изменений), `.sub` с
`job?.message || fallback`, трек-бар (переиспользует визуальный язык `.stackbar` — пилюля,
`var(--surface-2)`, `border-radius: 999px`) с заливкой на `clampPct(job?.progress ?? 0)`%, и строку
метаданных: `{pct}%` + прошедшее время + (если есть) `· осталось ≈ {eta}`.

**`frontend/app/globals.css`** — добавить `.progress-track`/`.progress-fill`/`.progress-meta`
рядом с существующим `.stackbar` (тот же паттерн: track `height: 6-8px; background: var(--surface-2);
border-radius: 999px; overflow: hidden`, fill `background: var(--brand); transition: width .3s ease`),
`.progress-meta` — `font-family: var(--font-mono); font-size: 11.5px; color: var(--muted);` как у
`.stage-empty .sub`.

**`frontend/app/page.tsx`** — в ветке `queued`/`running` (строки 104-109, внутри `.zoom-vp`) заменить
статичный `<div className="stage-empty">...</div>` на `<AnalysisProgress job={job.data} startedAt={...}
fallback={...} />`, оставив ветку `status === "error"` без изменений. `startedAt` — новый
`useState<number | null>(null)`, выставляется в `runAnalyze` (`setStartedAt(Date.now())`) вместе с
`setJobId(null)`; передаётся в компонент как `startedAt ?? Date.now()`. `fallback` — существующий
текст по режиму (`"панорама · сегментация тайлов"` / `"крупный план · сегментация фаз"`),
используется пока `job.message` ещё `null` (между `submit` и первым `report(...)` от пайплайна), и
на этапе загрузки файла (`analyze.isPending`, `job` ещё `undefined`) — в этом случае `fallback`
меняется на `"загрузка файла на сервер"`.

## Тестирование

- **Backend** (`backend/tests/`):
  - `test_jobs.py` — обновить 2 существующих теста под новую сигнатуру `fn(report)`; добавить
    проверку, что `report()`, вызванный из `fn`, отражается в `store.get(jid).progress`/`.message`
    до того, как джоба перейдёт в `done`.
  - `test_pipeline.py` / новый `test_closeup_progress.py` — колбэк, переданный в
    `analyze_closeup(rgb, cfg, on_progress=cb)`, вызывается ≥1 раз, значения `progress`
    монотонно не убывают и лежат в `[0, 1]`, среди сообщений встречается «оценка
    неопределённости».
  - `test_uncertainty.py` — `ensemble_uncertainty(rgb, cfg, on_step=cb)` вызывает `cb` ровно 5 раз
    (по числу `_PERTURBATIONS`) с `i` от 1 до `total=5`.
  - `test_tiling_feather.py` / новый `test_tiling_count.py` — `count_tiles(path, cfg)` совпадает (±
    отброшенные краевые тайлы) с фактическим числом итераций `iter_tiles` на тестовом изображении.
  - `test_panorama.py` — колбэк в `analyze_panorama(..., on_progress=cb)` вызывается многократно,
    финальное значение перед возвратом ≤ 1.0, среди сообщений — «сегментация тайлов».
  - Полный `cd backend && .venv/bin/pytest -q` должен остаться зелёным.
- **Frontend** (`frontend/tests/`):
  - Новый `progress.test.mjs` — `formatDuration`, `computeEta` (включая `null` при низком
    `progress`), `clampPct` (включая clamp вне `[0,1]`).
  - `npm run build` (type-check) должен остаться зелёным.
- **Ручная проверка:** `docker compose up` (или локальный dev-режим) → загрузить снимок в обоих
  режимах → визуально убедиться, что бар/этап/% реально двигаются во время обработки, а не прыгают
  сразу 5%→100%.

## Риски / открытые вопросы для плана

- Веса чекпойнтов close-up (0.08/0.15/0.30/0.80/0.88) подобраны по относительной дороговизне шагов,
  не измерены профайлером — в плане можно уточнить на реальном изображении, если пропорции
  окажутся сильно кривыми (не критично: это ориентир для пользователя, не точный SLA).
- ETA — линейная экстраполяция, будет заметно врать в первые секунды и на неравномерных этапах
  (ensemble после сегментации ускоряет прогресс скачком) — это ожидаемо и приемлемо, порог 8%
  снижает худшие случаи, но не устраняет их полностью.
