import os
import math
import random
import hashlib
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rules import (
    LUX_RULES,
    ROOM_TO_FIXTURES,
    ROOM_TO_BRAND,
    BRAND_GROUPS,
    TYPICAL_LM,
    BRAND_PRICE_COEF,
    FIXTURE_COEFF,
)

# ===== Загрузка модели и каталога =====
MODEL_PATH = "models/fixtures_model.pkl"
DATASET_PATH = "data/lighting_dataset.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)

# ===== FastAPI =====
app = FastAPI(title="Lighting AI Calculator", version="4.0")

# Раздача статики
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

# ===== Входные данные API =====
class InputData(BaseModel):
    room_type: str
    area_m2: Optional[float] = None       # используется для помещений
    length_m: Optional[float] = None      # используется для улицы/эвакуационного выхода
    ceiling_h: float
    budget: float


# ===== Вспомогательные функции =====
def _seed_from_payload(room_type: str, area_m2: float, ceiling_h: float) -> int:
    """
    Делаем выбор типа/люменов/бренда детерминированным для одинакового ввода.
    Бюджет намеренно не учитываем в seed.
    """
    s = f"{room_type}|{round(area_m2, 3)}|{round(ceiling_h, 3)}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)


def _calc_area_for_outdoor(room_type: str, area_m2: Optional[float], length_m: Optional[float]) -> float:
    """
    Для улицы и аварийного выхода считаем условную площадь как длина × стандартная ширина.
    Для остальных типов используем area_m2.
    """
    if room_type == "Улица":
        # типовая ширина пешеходной/локальной зоны
        width = 3.0
        return max(0.0, (length_m or 0.0) * width)
    if room_type == "Аварийный выход":
        # минимальная ширина эвакуационного прохода
        width = 1.2
        return max(0.0, (length_m or 0.0) * width)
    # помещения внутри зданий
    return float(area_m2 or 0.0)


def _pick_model_name_from_dataset(fixture_type: str, brand_hint: Optional[str], rng: random.Random) -> tuple[str, str]:
    """
    Возвращает (model_name, brand), стараясь выбрать модель из датасета под тип светильника
    и, если задано, под нужный бренд. Если подходящих строк нет — формирует fallback имя.
    """
    subset = df[df["fixture_type"] == fixture_type]

    if brand_hint:
        # ищем модели, чьё имя начинается с brand_hint + пробел
        branded = subset[subset["model_name"].str.startswith(brand_hint + " ", na=False)]
        if not branded.empty:
            row = branded.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
            model_name = str(row["model_name"])
            brand = model_name.split()[0]
            return model_name, brand

    if not subset.empty:
        row = subset.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        model_name = str(row["model_name"])
        brand = model_name.split()[0]
        return model_name, brand

    # fallback, если датасет пуст по этому типу
    brand = brand_hint or "DemoBrand"
    suffix = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2)) + str(rng.randint(1000, 9999))
    model_name = f"{brand} {fixture_type} {suffix}"
    return model_name, brand


# ===== Основной эндпоинт =====
@app.post("/predict")
def predict(data: InputData):
    # 1) Площадь с учётом линейных outdoor-сценариев
    area_m2 = _calc_area_for_outdoor(data.room_type, data.area_m2, data.length_m)

    # 2) Нормативы
    if data.room_type not in LUX_RULES:
        return {"error": f"Неизвестный тип помещения: {data.room_type}"}
    norm = LUX_RULES[data.room_type]
    required_lux = norm["lux"]
    norm_ref = norm["norm"]

    # 3) Поправка на высоту потолка (выше 3 м — +5% люменов за каждый метр)
    height_factor = 1.0 + 0.05 * max(0.0, (data.ceiling_h or 0.0) - 3.0)

    # 4) Суммарные люмены
    total_lumens = int(round(required_lux * area_m2 * height_factor))

    # 5) Детерминированный выбор типа/люменов/бренда
    seed = _seed_from_payload(data.room_type, area_m2, data.ceiling_h or 0.0)
    rng = random.Random(seed)

    # тип светильника
    fixture_type = rng.choice(ROOM_TO_FIXTURES[data.room_type])

    # световой поток одного светильника
    lm_low, lm_high = TYPICAL_LM.get(fixture_type, (2500, 3500))
    fixture_lm = rng.randint(lm_low, lm_high)

    # группа брендов по типу помещения (administrative/domestic/industrial/outdoor)
    group = ROOM_TO_BRAND.get(data.room_type, "domestic")
    brand = rng.choice(BRAND_GROUPS[group])

    # 6) Физический baseline по количеству
    baseline_count = max(1, math.ceil((total_lumens or 0) / max(1, fixture_lm)))

    # 7) Прогноз ML
    pred_ml = int(round(model.predict([[area_m2, data.ceiling_h or 0.0, required_lux, fixture_lm]])[0]))

    # финальное количество: не ниже физики и не выше 1.5× базового
    fixtures_count = min(max(baseline_count, pred_ml), max(1, int(baseline_count * 1.5)))

    # 8) Имя модели из датасета
    model_name, brand_from_name = _pick_model_name_from_dataset(fixture_type, brand, rng)
    brand = brand_from_name

    # 9) Цена за штуку
    base_rate = 0.5
    price_rub = int(fixture_lm * base_rate *
                    BRAND_PRICE_COEF.get(brand, 1.0) *
                    FIXTURE_COEFF.get(fixture_type, 1.0))

    # нормализация цен
    if group in ("domestic", "administrative"):
        price_rub = max(800, min(price_rub, 6000))

    total_cost = int(fixtures_count * price_rub)

    # 10) Бюджет
    warning = None
    if total_cost > data.budget:
        # самый дешевый бренд в группе
        cheap_brand = min(BRAND_GROUPS[group], key=lambda b: BRAND_PRICE_COEF.get(b, 1.0))

        if cheap_brand != brand:
            # подбор модели этого бренда из датасета
            subset = df[(df["fixture_type"] == fixture_type) &
                        (df["model_name"].str.startswith(cheap_brand + " ", na=False))]
            if not subset.empty:
                row = subset.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
                model_name = str(row["model_name"])
            else:
                # fallback, если в датасете нет такого сочетания
                suffix = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2)) + str(rng.randint(1000, 9999))
                model_name = f"{cheap_brand} {fixture_type} {suffix}"

            brand = cheap_brand
            price_rub = int(fixture_lm * base_rate *
                            BRAND_PRICE_COEF.get(brand, 1.0) *
                            FIXTURE_COEFF.get(fixture_type, 1.0))
            if group in ("domestic", "administrative"):
                price_rub = max(800, min(price_rub, 6000))
            total_cost = int(fixtures_count * price_rub)

        if total_cost > data.budget:
            warning = f"⚠ Недостаточно средств: требуется {total_cost:,} ₽, бюджет {data.budget:,} ₽.".replace(",", " ")
        else:
            warning = f"✅ Подобран более доступный бренд ({brand}), чтобы уложиться в бюджет."

    # 11) Ответ
    return {
        "room_type": data.room_type,
        "area_m2": round(area_m2, 2),
        "length_m": data.length_m,
        "ceiling_h": data.ceiling_h,
        "budget": int(data.budget),
        "required_lux": int(required_lux),
        "total_lumens": int(total_lumens),
        "fixture_type": fixture_type,
        "fixture_lm": int(fixture_lm),
        "fixtures_count": int(fixtures_count),
        "price_rub": int(price_rub),
        "total_cost": int(total_cost),
        "model_name": model_name,
        "norm_ref": norm_ref,
        "warning": warning,
    }
