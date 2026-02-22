import os
import math
import hashlib
from pathlib import Path
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

def baseline_fixture_count(total_lumens: int, fixture_lm: int) -> int:
    """
    Deterministic physics baseline:
    minimum fixtures needed to meet required luminous flux.
    """
    total_lumens = int(total_lumens or 0)
    fixture_lm = int(fixture_lm or 1)
    return max(1, math.ceil(max(0, total_lumens) / max(1, fixture_lm)))

# ===== Вспомогательные функции =====
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
def pick_model_name_deterministic(df: pd.DataFrame, fixture_type: str, brand: str) -> str:
    """
    Pick the cheapest model within a fixture_type (optionally filtered by brand if present).
    Falls back to a generic name if dataset doesn't contain needed columns.
    """
    subset = df[df.get("fixture_type") == fixture_type] if "fixture_type" in df.columns else df.copy()

    # If dataset contains brand column, try to filter by it.
    if "brand" in subset.columns:
        subset_brand = subset[subset["brand"] == brand]
        if len(subset_brand) > 0:
            subset = subset_brand

    if len(subset) == 0:
        return f"{fixture_type} Model"

    # Prefer cheapest by price if column exists
    if "price_rub" in subset.columns:
        row = subset.sort_values("price_rub", ascending=True).iloc[0]
    else:
        row = subset.iloc[0]

    # Use model_name if exists
    if "model_name" in row.index:
        return str(row["model_name"])

    return f"{fixture_type} Model"

def deterministic_fixture_type(room_type: str) -> str:
    variants = ROOM_TO_FIXTURES.get(room_type, [])
    return sorted(variants)[0] if variants else "Панельный"


def deterministic_brand(room_type: str) -> str:
    group = ROOM_TO_BRAND.get(room_type, "domestic")
    candidates = BRAND_GROUPS.get(group, [])
    if not candidates:
        return "Generic"
    return min(candidates, key=lambda b: BRAND_PRICE_COEF.get(b, 1.0))

def deterministic_fixture_type(room_type: str) -> str:
    variants = ROOM_TO_FIXTURES.get(room_type, [])
    return sorted(variants)[0] if variants else "Панельный"


def deterministic_fixture_lm(fixture_type: str) -> int:
    lm_low, lm_high = TYPICAL_LM.get(fixture_type, (2500, 3500))
    return int((lm_low + lm_high) / 2)


def deterministic_brand(room_type: str) -> str:
    group = ROOM_TO_BRAND.get(room_type, "domestic")
    candidates = BRAND_GROUPS.get(group, [])
    if not candidates:
        return "Generic"
    return min(candidates, key=lambda b: BRAND_PRICE_COEF.get(b, 1.0))

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

    # 5) Детерминированный выбор

    fixture_type = deterministic_fixture_type(data.room_type)
    fixture_lm = deterministic_fixture_lm(fixture_type)
    brand = deterministic_brand(data.room_type)
    model_name = pick_model_name_deterministic(df, fixture_type, brand)

    # 6) Физический baseline по количеству
    baseline_count = baseline_fixture_count(total_lumens, fixture_lm)

    # 7) Прогноз ML
    pred_ml = int(round(model.predict([[area_m2, data.ceiling_h or 0.0, required_lux, fixture_lm]])[0]))

    # финальное количество: не ниже физики и не выше 1.5× базового
    fixtures_count = min(max(baseline_count, pred_ml), max(1, int(baseline_count * 1.5)))


    # 9) Цена за штуку
    base_rate = 0.5
    price_rub = int(fixture_lm * base_rate *
                    BRAND_PRICE_COEF.get(brand, 1.0) *
                    FIXTURE_COEFF.get(fixture_type, 1.0))

    # нормализация цен
    group = ROOM_TO_BRAND.get(data.room_type, "domestic")

    if group in ("domestic", "administrative"):
        price_rub = max(800, min(price_rub, 6000))

    total_cost = int(fixtures_count * price_rub)

    # 10) Бюджет
    warning = None
    if data.budget is not None and total_cost > data.budget:
        # самый дешевый бренд в группе
        group_brands = BRAND_GROUPS.get(group, [])
        if group_brands:
            cheap_brand = min(group_brands, key=lambda b: BRAND_PRICE_COEF.get(b, 1.0))
        else:
            cheap_brand = brand  # fallback

        if cheap_brand != brand:
            # подбор модели этого бренда из датасета (детерминированно)
            subset = df[df["fixture_type"] == fixture_type].copy()

            # если в model_name зашит бренд префиксом "BRAND ..."
            subset_brand = subset[subset["model_name"].str.startswith(cheap_brand + " ", na=False)]
            if not subset_brand.empty:
                # берём самую дешёвую, если есть price_rub в датасете
                if "price_rub" in subset_brand.columns:
                    row = subset_brand.sort_values("price_rub", ascending=True).iloc[0]
                else:
                    row = subset_brand.iloc[0]
                model_name = str(row["model_name"])
            else:
                # fallback: без RNG (детерминированно)
                model_name = f"{cheap_brand} {fixture_type} DEMO"

            brand = cheap_brand

            # пересчёт цены детерминированно
            price_rub = int(
                fixture_lm * base_rate
                * BRAND_PRICE_COEF.get(brand, 1.0)
                * FIXTURE_COEFF.get(fixture_type, 1.0)
            )
            if group in ("domestic", "administrative"):
                price_rub = max(800, min(price_rub, 6000))

            total_cost = int(fixtures_count * price_rub)

        if total_cost > data.budget:
            warning = (
                f"⚠ Недостаточно средств: требуется {total_cost:,} ₽, бюджет {data.budget:,} ₽."
                .replace(",", " ")
            )
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
