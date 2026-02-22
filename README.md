# EN: Hybrid Lighting Calculation Service

## 📌 Project Overview

A hybrid indoor lighting calculation service combining a deterministic regulatory formula (physics baseline) with a LightGBM-based regression model to estimate the required number and type of luminaires based on room type, floor area, ceiling height, and available budget.

The demo version operates on a synthetic dataset that reproduces regulatory calculation logic with added variability. The system architecture, however, is designed as production-ready: the model is integrated into a REST API, includes validation and constraint logic, and can be extended to real luminaire catalogs and photometric data.

---

## 🎯 Problem Formulation

The lighting calculation task is formalized as follows:

1. Required luminous flux computation:

Φ = E_norm × S × K

where:

* E_norm — required illuminance (lux)
* S — room area
* K — maintenance/safety factor

2. Baseline fixture count:

N = ceil(Φ / Φ_fixture)

3. ML regression layer for context-aware adjustment of fixture quantity.

4. Constraint layer:

* lower bound: physics baseline
* upper bound: 1.5 × baseline
* budget validation and brand adjustment

Overall system pipeline:

Norms → Physics Baseline → ML Regression → Constraints → Final Result

---

## 🧠 Architecture

### 1️⃣ Regulatory Layer

* Room type reference dictionary
* Mapping to required illuminance levels
* Maintenance factor handling

### 2️⃣ Deterministic Physics Baseline

* Required luminous flux calculation
* Minimum compliant fixture count

### 3️⃣ ML Regression (LightGBM)

* Features: area, ceiling height, required illuminance, luminaire luminous flux
* Target: fixture count
* Output rounded to integer values

### 4️⃣ Constraint Layer

* Ensures fixture count is not below baseline
* Caps output at 1.5 × baseline
* Applies budget validation and fallback logic

---

## 📊 Dataset

The project uses a synthetic dataset (`lighting_dataset.csv`) generated from regulatory formulas with controlled variability.

This enables:

* Reproducible model training
* Baseline vs ML comparison
* Clear demonstration of the ML pipeline

The architecture is designed to support real-world datasets (photometric curves, manufacturer catalogs, reflectance parameters).

---

## 📈 Model Evaluation

Evaluation on the full dataset:

| Model                   | MAE    | RMSE   | R²     | acc@1  |
| ----------------------- | ------ | ------ | ------ | ------ |
| Baseline (physics)      | 0.8646 | 0.9298 | 0.9995 | 1.0000 |
| ML raw                  | 0.5709 | 1.3349 | 0.9990 | 0.9109 |
| Final (API constraints) | 0.9617 | 1.2113 | 0.9992 | 0.9546 |

### Interpretation

* The deterministic baseline already performs strongly on synthetic rule-based data.
* The ML model reduces systematic bias (lower MAE).
* Due to increased variance, ML may introduce occasional larger deviations, increasing RMSE.
* The constraint layer stabilizes predictions for production use.

This behavior is expected when training on data closely aligned with deterministic formulas. The ML component is positioned as a corrective layer that becomes more valuable when applied to real-world, non-ideal datasets (reflectance variations, installation losses, catalog diversity, etc.).

---

## ⚙️ Tech Stack

* Python
* LightGBM
* Pandas / NumPy
* FastAPI
* REST API
* Cloud deployment

---

## 🚀 Running the Project

```bash
python data_generator.py
python train.py
python evaluate.py
uvicorn calculator.main:app --reload
```

## ⚠️ Limitations

* Synthetic dataset
* Simplified luminous model (no IES photometric curves)
* No spatial light distribution simulation
* No glare (UGR) modeling

---

## 🔮 Future Improvements

* Integration of real photometric data
* Cost optimization layer
* Extended luminaire catalog
* Validation on real engineering projects

---

## 💡 Summary

The project demonstrates a hybrid engineering approach to lighting calculation automation, combining regulatory physics, ML regression, and constraint logic within a production-ready REST API.

---

# RU: Гибридный сервис расчета освещения помещений

## 📌 Описание проекта

Гибридный сервис расчета освещения помещений, сочетающий детерминированную нормативную формулу (physics baseline) и ML-регрессию на базе LightGBM для расчета количества и марки светильников исходя из типа помещения, площади и высоты потолка помещения, доступного бюджета.

В демонстрационной версии используется синтетический датасет, воспроизводящий нормативную логику расчёта с добавлением вариативности. Архитектура сервиса при этом спроектирована как production-ready: модель интегрирована в REST API, предусмотрена валидация, ограничивающая логика и возможность масштабирования под реальные каталоги светильников и фактические фотометрические данные.

Сервис поддерживает 30+ типов помещений, учитывает нормативную освещённость (люкс), коэффициент запаса и бюджетные ограничения, и развёрнут в виде REST API на FastAPI.

---

## 🎯 Постановка задачи

Задача расчёта освещения формализуется как:

1. Определение требуемого светового потока:

Φ = E_norm × S × K

где:

* E_norm — нормативная освещённость (люкс)
* S — площадь помещения
* K — коэффициент запаса

2. Определение количества светильников:

N = ceil(Φ / Φ_fixture)

3. Дополнительная ML-регрессия для контекстной корректировки количества светильников.

4. Ограничение результата (constraint layer):

* количество не ниже физического baseline
* верхний предел — 1.5 × baseline
* проверка бюджетных ограничений

Таким образом система реализует гибридную архитектуру:

Norms → Physics Baseline → ML Regression → Constraints → Result

---

## 🧠 Архитектура

### 1️⃣ Нормативный слой

* Справочник типов помещений
* Привязка к нормативной освещённости
* Учёт коэффициента запаса

### 2️⃣ Физический baseline

* Детерминированный расчёт требуемых люменов
* Минимально допустимое количество светильников

### 3️⃣ ML-регрессия (LightGBM)

* Признаки: площадь, высота потолка, нормативная освещённость, световой поток светильника
* Целевая переменная: количество светильников
* Предсказание округляется до целого числа

### 4️⃣ Constraint layer

* Нижняя граница: baseline
* Верхняя граница: 1.5 × baseline
* Проверка бюджета

---

## 📊 Датасет

Используется синтетический датасет (lighting_dataset.csv), сгенерированный на основе нормативных правил с добавлением вариативности.

Это позволяет:

* воспроизводимо обучать модель
* сравнивать ML с физическим baseline
* демонстрировать корректную ML-пайплайн архитектуру

---

## 📈 Оценка модели

Метрики на полном датасете:

| Model                   | MAE    | RMSE   | R²     | acc@1  |
| ----------------------- | ------ | ------ | ------ | ------ |
| Baseline (physics)      | 0.8646 | 0.9298 | 0.9995 | 1.0000 |
| ML raw                  | 0.5709 | 1.3349 | 0.9990 | 0.9109 |
| Final (API constraints) | 0.9617 | 1.2113 | 0.9992 | 0.9546 |

### Интерпретация

* Детерминированный baseline уже демонстрирует высокую точность на синтетических данных.
* ML уменьшает систематическое смещение (bias), снижая MAE.
* Constraint layer стабилизирует редкие выбросы ML.

На синтетических данных, основанных на нормативной формуле, детерминированный baseline уже демонстрирует высокую точность. ML-сегмент в данной конфигурации снижает систематическое смещение (bias), однако за счёт увеличения дисперсии может давать более редкие, но более крупные отклонения. Это подчёркивает роль ML как корректирующего слоя, который потенциально раскрывает преимущества при наличии реальных, неидеальных данных (учёт отражений, потерь, вариативности монтажа и других факторов).

---

## ⚙️ Технологии

* Python
* LightGBM
* Pandas / NumPy
* FastAPI
* REST API
* Cloud deployment

---

## 🚀 Развёртывание

```bash
python data_generator.py
python train.py
python evaluate.py
uvicorn calculator.main:app --reload
```

---

## ⚠️ Ограничения

* Синтетический датасет
* Упрощённая светотехническая модель (без IES-кривых)
* Нет пространственного моделирования распределения света
* Не учитывается ослепление (UGR)

---

## 🔮 Возможные улучшения

* Использование реальных фотометрических данных
* Оптимизационный слой для минимизации стоимости
* Расширение каталога светильников
* Валидация на реальных проектах

---

## 💡 Итог

Проект демонстрирует гибридный инженерный подход к автоматизации светотехнических расчётов: сочетание нормативной физики, ML-регрессии и ограничивающей логики в production-ready API.