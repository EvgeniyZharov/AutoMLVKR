# 🤖 AutoML Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-ff4b4b)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

---

## 🚀 Описание проекта

**AutoML Platform** — это модульное интерактивное приложение на базе Streamlit для автоматизированного машинного обучения. Оно объединяет в себе обучение, валидацию и предсказание для задач классификации, регрессии, анализа временных рядов и прогноза температуры электронных компонентов.

Приложение ориентировано на интуитивное взаимодействие с пользователем без необходимости программирования. Всё, что требуется — загрузить CSV и выбрать нужный тип модели.

---

## 🔥 Основные возможности

* 📊 **Регрессия**: Linear Regression, Gradient Boosting, Decision Trees
* 🔍 **Классификация**: Logistic Regression, Random Forest, Decision Tree
* 📈 **Прогноз временных рядов**: LSTM, GRU, Bidirectional RNNs, CNN-LSTM, ConvLSTM2D
* 🌡 **Прогноз температуры компонентов**: временные ряды + инженерия признаков
* 📁 Интерактивная загрузка пользовательских данных
* 🧠 Обучение и переобучение моделей в один клик
* 📉 Метрики: Accuracy, F1, MSE, MAPE
* 💾 Сохранение обученных моделей в `trained_models/`
* 🌐 Возможность работы в режиме API (`mode=app&api_key=`)

---

## 🧩 Структура проекта

```
AutoML-System/
├── Pipelines/
│   ├── BasePipeline.py              # Абстрактный класс пайплайна
│   ├── ClassificationPipeline.py    # Классификация
│   ├── RegressionPipeline.py        # Регрессия
│   ├── TimeSeriesPipeline.py        # Временные ряды
│   └── ElectronicPipeline.py        # Температура компонентов
├── interface.py                     # Основной запуск Streamlit
├── trained_models/                  # Папка для сохранения моделей
└── README.md                        # Документация проекта
```

---

## ⚙️ Установка и запуск

1. Клонируйте репозиторий

```bash
git clone https://github.com/yourname/automl-system.git
cd automl-system
```

2. Установите зависимости

```bash
pip install -r requirements.txt
```

3. Запустите приложение

```bash
streamlit run interface.py
```

4. Откройте в браузере:

```
http://localhost:8501/?mode=train
```

---

## 🧪 Пример данных

### 🔹 Для регрессии/классификации:

```csv
feature1,feature2,target
1.2,3.4,0
2.3,1.2,1
```

### 🔹 Для временного ряда:

```csv
value
123.4
124.1
...
```

### 🔹 Для температурного прогноза (электроника):

* Компоненты (`components.csv`)
* Измерения (`temperature.csv`)

---

## ✅ TODO

* [ ] Поддержка AutoML-библиотек (AutoSklearn, TPOT)
* [ ] Отображение графиков ошибок/предсказаний
* [ ] Docker-образ
* [ ] Экспорт отчёта в PDF
* [ ] Интеграция с базой данных

---

## 👨‍💻 Автор

* Разработчик: [Евгений Жаров](https://github.com/EvgeniyZharov)
* По вопросам: `eezharov@edu.hse.ru`
* Файл для десктопного запуска: [Ссылка на Гугл Диск с .exe-файлом](https://drive.google.com/file/d/1BKm3uOMsK9q3DOpPHZRVnLtdwLCr4rIV/view?usp=drive_link)
* Датасет компонентов: [Ссылка на Гугл Диск с .csv-файлом](https://drive.google.com/file/d/1GB2VsMM1M0EIBwpXcaUhy9TlL2Kimjjj/view?usp=drive_link)
* Датасет температурой: [Ссылка на Гугл Диск с .csv-файлом](https://drive.google.com/file/d/1XG9ZTZzoalubkV45_R30BYcnD1ZdNplM/view?usp=drive_link)
