# EN: 💡 AI Lighting Recommendation Calculator
A web-based AI service that calculates the optimal number and type of luminaires for various environments — from residential apartments to industrial facilities.
The system uses a LightGBM-based machine learning model trained on regulatory lighting standards (GOST, SNIP, SP equivalents) to provide fast and reliable recommendations.

---

## 🌐 Live Demo

```https://ai-lighting-suggestion-calculator.onrender.com```

(Note: initial load may take a few seconds due to Render cold start.)

---

## ⚙️ Features

Lighting calculation based on room type, area, ceiling height, and budget  
Support for 30+ room categories (offices, warehouses, schools, stadiums, cafés, etc.)  
AI inference time < 1 second  
Cost-aware luminaire recommendations  
Clean, production-ready UI  

---

## 🏗 Project Structure
```
calculator/
├── main.py                 # FastAPI application entry point
├── models/                 # Trained LightGBM models
├── static/
│   ├── images/             # Backgrounds and visual assets
│   └── styles.css          # Styling (Tailwind)
├── templates/
│   └── index.html          # Main UI template
├── requirements.txt
└── README.md
```

---

## 💻 Running Locally
1. Install dependencies
```pip install -r requirements.txt```

2. Start the server
```uvicorn main:app --reload```

3. Open
```http://127.0.0.1:8000``` in your browser

---

## ☁️ Running on Render

If you want to deploy the calculator locally:  
1) Go to render.com  
2) Create a New Web Service  
3) Specify the GitHub repository  

In the settings:

Build Command: ```pip install -r requirements.txt```  
Start Command: ```uvicorn main:app --host 0.0.0.0 --port 10000```  
After deployment, Render will provide a link to access the site  

---

## 📸 Interface
Main screen – project description.  
Calculator – enter room parameters and get instant results.  
About the project – explanation of calculation principles and standard examples.  

---

## 🛠 Tech Stack

- FastAPI (REST backend)
- LightGBM (ML model)
- Pandas / NumPy (data preprocessing)
- Tailwind CSS (frontend styling)
- Render (deployment)

---

## 🧠 System Design Notes

The system is designed as a lightweight, production-ready machine learning service, not as a laptop-based prototype.

Design Principles  

1. Regulatory-aware modeling. Training incorporates structured lighting standards to ensure domain-specific compliance.  
2. Feature-driven architecture. Input parameters (room type, dimensions, budget constraints) are transformed into structured features for model output.  
3. Low-latency inference. The model is preloaded into memory upon service startup to ensure sub-second response times.  
4. API-first deployment. The calculator runs as a REST service, allowing for integration into ERP systems, sales tools, or engineering workflows.  
5. Scalability. The architecture supports the addition of new room categories, product catalogs, or updated regulatory standards without redesigning the pipeline.  

## 🎯 Purpose

This project demonstrates the practical application of machine learning in engineering decision support systems.  

It serves as:  
A blueprint for a manufacturing-focused machine learning portfolio;  
A reference architecture for specialized AI services;  
A foundation for scalable recommendation platforms for commercial lighting.  

---

## ❗ Note

The program interface is only in Russian; English localization will be implemented in the future.

---

# RU: 💡 ИИ-калькулятор подбора освещения
Интерактивный веб-сервис, который помогает рассчитать оптимальное количество и тип светильников для любого помещения от квартиры до производственного цеха.  
Калькулятор использует AI-модель на базе LightGBM, обученную по нормативам ГОСТ, СНиП и СП, чтобы выдавать рекомендации быстро и точно.  

---

## 🌐 Онлайн-версия
Доступна по ссылке:
```ai-lighting-suggestion-calculator.onrender.com```
(Если страница загружается дольше обычного – это нормально, Render запускает веб-сервис)

---

## ⚙️ Возможности
1. Расчёт освещённости по типу помещения, площади, высоте и бюджету
2. Поддержка 30+ типов помещений (офисы, склады, кафе, школы, стадионы и др.)
3. Мгновенный результат (AI-расчёт занимает < 1 секунды)
4. Рекомендации по типу и стоимости светильников
5. Современный дизайн с мягким синим интерфейсом и фоном в едином стиле

---

## 📁 Структура проекта
```
calculator/
├── main.py                 # Основной сервер FastAPI
├── models/                 # Обученные модели ML (LightGBM)
├── static/
│   ├── images/             # Изображения (фон, светильники)
│   └── styles.css          # Стили Tailwind (если локально)
├── templates/
│   └── index.html          # Основной интерфейс калькулятора
├── requirements.txt        # Список зависимостей
└── README.md
```
---

## 💻 Запуск локально
1. Установите зависимости
```pip install -r requirements.txt```

2. Запустите сервер
```uvicorn main:app --reload```

3. Откройте в браузере
```http://127.0.0.1:8000```

---

## ☁️ Запуск на Render

Если вы хотите развернуть калькулятор у себя:  
1) зайдите на render.com  
2) создайте New Web Service  
3) укажите репозиторий GitHub  

В настройках:

Build Command: ```pip install -r requirements.txt```  
Start Command: ```uvicorn main:app --host 0.0.0.0 --port 10000```  
После деплоя Render предоставит ссылку, по которой будет доступен сайт  

---

## 📸 Интерфейс
Главный экран – описание проекта.  
Калькулятор – ввод параметров помещения и моментальный результат.  
О проекте – объяснение принципов расчёта и примеры нормативов.  

---

## 🛠 Используемые технологии
- FastAPI – Web-сервер и API для расчётов
- LightGBM – Основная ML-модель подбора освещения
- Pandas / NumPy – Обработка входных данных
- Tailwind CSS – Современная адаптивная верстка
- Render – Хостинг и авторазвёртывание приложения

---

## 🧠 Заметки о проектировании системы

Система разработана как легковесный, готовый к производству сервис машинного обучения, а не как прототип на основе ноутбука.

Принципы проектирования

1. Моделирование с учетом нормативных требований. Обучение включает в себя стандарты структурированного освещения для обеспечения соответствия предметной области.
2. Архитектура, ориентированная на признаки. Входные параметры (тип помещения, размеры, бюджетные ограничения) преобразуются в структурированные признаки для вывода модели.
3. Вывод с низкой задержкой. Модель предварительно загружается в память при запуске сервиса для обеспечения времени отклика менее секунды.
4. Развертывание с приоритетом API. Калькулятор работает как REST-сервис, что позволяет интегрировать его в ERP-системы, инструменты продаж или инженерные рабочие процессы.
5. Масштабируемость. Архитектура поддерживает добавление новых категорий помещений, каталогов продукции или обновленных нормативных стандартов без перепроектирования конвейера.

---

## 🎯 Назначение

Этот проект демонстрирует практическое применение машинного обучения в системах поддержки принятия инженерных решений.

Он служит:
Проектом портфеля решений в области машинного обучения, ориентированным на производство;  
Эталонной архитектурой для специализированных сервисов ИИ;  
Основой для масштабируемых платформ рекомендаций коммерческого освещения.  

---

## ❗ Примечание

Интерфейс программы только на русском языке, английская локализация будет внедрена в будущем
