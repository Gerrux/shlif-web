# Панорама: экспозиционная неустойчивость магнетит/сульфид — design

*2026-07-04. Триггер: жалоба пользователя — «определение магнетитов и сульфидов на панорамах
страдает сильно», уточнение — «когда есть пересвеченный участок, магнетит определяется как
сульфид».*

## Проблема (подтверждена экспериментально)

`backend/app/shlif/segment.py::segment_phases` делит sulfide/magnetite/matrix трёхклассовым
Otsu-порогом по L-каналу (Lab), **пересчитываемым заново на гистограмме яркости каждого
изображения/тайла** — без абсолютной привязки. WB (`gray_world_white_balance`) убирает только
цветовой оттенок, CLAHE — только локальный контраст; ни один не якорит абсолютную яркость.

Смоделировав пересвет (gain×gamma) на примере шлифа из датасета, получили рост доли пикселей
magnetite→sulfide: **11.4% (gain 1.3) → 37.7% (gain 1.6) → 45.7% (gain 2.0)** от истинной площади
магнетита; одновременно тёмная матрица уезжает в «магнетит» тем же механизмом. Это конкретизация
задокументированной находки №2 (`hakaton_nornikel/CLAUDE.md`) — не то, что решают конкуренты
(`../hakaton_nornikel/competitor_analysis.md` разбирает цветовую неразличимость
магнетит/пирротин, не экспозиционный дрейф).

Механизм уже отловится существующим `uncertainty.py::ensemble_uncertainty` (gamma/gain
пертурбации, включая «засветить») — но **он подключён только к `closeup.py`**, панорама
(`panorama.py::_run_panorama`) его не вызывает вовсе и продолжает молча использовать чисто
классическую посегментную метку.

Отдельно: тот же классический ore/matrix-гейт (`~matrix`) на панорамных тайлах решает, войдёт ли
тайл в RF-классификацию сорта (`ore_frac >= min_ore`, `panorama.py:94-102`). Уже обученная
`unet_ore.pt` (бинарная руда/матрица, IoU 0.975 против классических 0.81, устойчива к экспозиции
по построению — обучена на WB+CLAHE-нормализованном входе) лежит в `backend/models/unet_ore.pt`,
но не подключена — README прямо помечает это как «planned follow-up work». Готовый референсный
инференс уже есть в origin-репо: `hakaton_nornikel/scripts/sam2_prelabel.py::build_unet` +
`unet_ore_decision` (тайловый луп, per-tile `wb_clahe`, argmax≠0), а полная copy-paste-спека
(вход/выход, препроцессинг, зависимости) — в `hakaton_nornikel/docs/models_for_shlif_web.md §2`.

## Согласованный скоуп

**Задача 1 — ensemble-uncertainty в панораму (основная).** Прогонять
`ensemble_uncertainty`/`find_low_conf_zones` потайлово внутри `_run_panorama`, аналогично тому,
как это уже делает `closeup.py::_uncertainty`. Спорные зоны «магнетит vs сульфид» на
пересвеченных участках попадают в ответ как «на проверку», а не как уверенная (но неверная)
метка. Низкий риск — переиспользует протестированный код, паттерн уже есть в кодовой базе.

**Задача 2 — unet_ore.pt в ore/matrix-гейт панорамы.** Когда `unet_ore.pt` присутствует на диске
(и torch/GPU доступны), заменить классический `~segment_phases(...).labels==0` тайловым
инференсом U-Net для решения ore/matrix (используется и в `ore_frac`-гейте, и в отображении).
Иначе (модель/torch недоступны) — **graceful fallback на текущую классику**, без падений (тот же
паттерн, что уже применён в `talc_unet.py::build_talc_unet` — вернуть `None`, если чекпойнт или
torch/smp недоступны).

**Вне скоупа (стретч, не делаем сейчас):** пункт 3 из обсуждения — якорение самого
`segment_phases` к стабильному ориентиру вместо плавающего Otsu. Трогает функцию, от которой
также зависит `features.py` (признаки RF-классификатора сорта, F1 0.84) → требует отдельной
ре-валидации, выше риск, вне текущего запроса.

## Архитектура

### Задача 1: `backend/app/pipeline/panorama.py`

- В цикле по тайлам (`for tile in iter_tiles(...)`), после `matrix = segment_phases(...)`, вызвать
  `ensemble_uncertainty(rgb, cfg)` (импорт из `app.shlif.uncertainty`) на **сыром `rgb` тайла**
  (та же сигнатура, что `closeup.py::_uncertainty` использует — принимает необработанный RGB и
  сама вызывает `preprocess`/`segment_phases` внутри перебора пертурбаций).
- Копить `low_conf_zones` per-tile со сдвигом в координаты дисплея (`dx0,dy0`) — тот же паттерн
  смещения, что уже применяется для `talc_disp` (строки `dx0,dy0,dx1,dy1` уже посчитаны в цикле).
- Копить агрегированную `undetermined_fraction` (взвешенно по площади тайла или простым средним —
  решить в плане) для итоговых метрик вердикта, по аналогии с `talc_frac`.
- Добавить в возврат `_run_panorama`/`analyze_panorama`: `low_conf_zones: list[dict]`,
  `undetermined_fraction: float` в `verdict.metrics`, консистентно с тем, что `closeup.py` уже
  отдаёт клиенту (см. `backend/app/pipeline/closeup.py` и соответствующий API-контракт в
  `backend/app/api/analyze.py` — свериться при реализации, не ломать существующий панорамный
  ответ).
- **Производительность:** ensemble — это 5 прогонов `segment_phases` на тайл; на 2048px тайлах
  панорамы это может быть заметно медленнее текущего однопрохода. Даунскейлить тайл перед
  ensemble (как `closeup.py::_uncertainty` даунскейлит до `_UNC_MAX_SIDE=1024`) и решить в плане,
  на каком размере считать (не на полном 2048, чтобы не удвоить-утроить время панорамы).

### Задача 2: новый модуль-обёртка + правка `panorama.py`

- Новый файл `backend/app/shlif/ore_unet.py` (по образцу существующего `talc_unet.py`):
  - `build_ore_unet(ckpt="unet_ore.pt", device=None) -> (model, device) | None` — guard на
    существование чекпойнта + успешный импорт `torch`/`segmentation_models_pytorch`; при неудаче
    `None` (никогда не бросает).
  - `ore_unet_mask(rgb, model, device, tile=512) -> np.ndarray[bool]` — тайловый инференс с
    `wb_clahe` **обязательно** (совпадает с обучением, см. `models_for_shlif_web.md §2`);
    `argmax(1) != 0` → ore. Портировать логику из
    `hakaton_nornikel/scripts/sam2_prelabel.py::build_unet`/`unet_ore_decision`, адаптировав под
    CPU-safe guarded-import паттерн `talc_unet.py` (никогда не импортировать torch на верхнем
    уровне модуля).
- `backend/app/pipeline/loader.py`: добавить `load_ore_unet()` — `@lru_cache`, аналогично
  `load_classifier()`, использует `settings.models_dir / "unet_ore.pt"`.
- `backend/app/pipeline/panorama.py`: в `_run_panorama`, если `load_ore_unet()` вернула модель —
  заменить `matrix = segment_phases(pre, cfg.segment).labels == 0` на
  `matrix = ~ore_unet_mask(rgb, model, device)`; иначе — текущее поведение без изменений.
  `ore_frac` считается от того же `matrix`, что уже даёт исправление и для гейта, и для отображения
  без дополнительных изменений схемы.
- **Совместимость.** `unet_ore.pt` — бинарный (руда/матрица), НЕ делит sulfide/magnetite внутри
  «руды» — это по-прежнему через классический `segment_phases`/уже задача 1 (uncertainty)
  прикрывает нестабильность именно этого внутреннего деления. Задача 2 фиксирует только внешнюю
  границу руда/матрица.

## Тестирование

- Новые/расширенные тесты в `backend/tests/`:
  - `test_panorama.py` (или новый `test_panorama_uncertainty.py`) — по образцу
    `test_closeup_uncertainty.py`: синтетическая панорама с резким пересветом одного тайла →
    `low_conf_zones`/`undetermined_fraction` присутствуют и в разумных пределах.
  - Новый `test_ore_unet.py` (по образцу `test_loader.py`/`test_resolve_threshold.py`) —
    `build_ore_unet` возвращает `None`, когда чекпойнт отсутствует (`monkeypatch models_dir`);
    `@pytest.mark.skipif` для реального инференса, если веса/torch недоступны в CI-окружении (тот
    же паттерн, что `test_panorama.py` уже использует для `classifier.pkl`).
  - `test_panorama.py::test_panorama_does_not_mutate_shared_config` — убедиться, что новый путь
    (ore U-Net) тоже не мутирует общий `@lru_cache`-конфиг.
- Полный прогон `cd backend && .venv/bin/pytest -q` должен остаться зелёным; не ломать
  существующие 39+ тестов.
- Ручная проверка: прогнать `_run_panorama` на реальной панораме (или синтетической с резким
  пересветом) через скрипт-пробник (как в этой сессии) и визуально сверить overlay/зоны.

## Риски / открытые вопросы для плана

- Точный размер даунскейла для ensemble на тайле панорамы (компромисс скорость/точность).
- Как агрегировать `undetermined_fraction` по всей панораме (среднее по руда-тайлам? взвешенно по
  `ore_density`, как `aggregate_section`?) — решить в implementation plan.
- Нужно ли добавлять `unet_ore` в ответ API как отдельный флаг «модель использована: classic/unet»
  для прозрачности перед жюри (WORKLOG-стиль честности) — небольшое дополнение, решить в плане.
