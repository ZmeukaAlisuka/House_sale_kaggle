# House price - Machine Learning & Deep Learning

Решение соревнования Kaggle House price по регрессии (предсказание цены дома).

---

##  Описание проекта

В проекте реализован полный цикл разработки Data Science решения:
1. **EDA (EDA.ipynb)** — разведочный анализ данных, исследование распределений признаков, корреляций и пропусков, визуализация с помощью matplotlib и seaborn.
2. **Предобработка данных** — обработка пропущенных значений, удаление неинформативных текстовых полей, One-Hot кодирование категориальных признаков.
3. **Классическое моделирование (model_Titanic.ipynb)** — сравнение одиночных моделей (Logistic Regression, Decision Tree, Random Forest, CatBoost, LightGBM, XGBoost, Sklearn GB) и **ансамблей (Voting, Stacking)** на 5-фолдовой кросс-валидации (StratifiedKFold).
4. **Deep Learning (models_dl.py, 	rainer_dl.py)** — реализация полносвязной нейронной сети (MLP) на PyTorch с BatchNorm, Dropout, оптимизатором AdamW и косинусным шедулером CosineAnnealingLR.
5. **Тюнинг гиперпараметров (model_tuning.ipynb)** — подбор оптимальных параметров через RandomizedSearchCV и сборка оптимизированного ансамбля.
6. **Автоматизированный пайплайн (main.py)** — запуск полного цикла обучения (ML + DL), автоматический выбор лучшей модели и сохранение сабмита submission.csv.

---
##  СПИСОК ИСХОДНЫХ ПРИЗНАКОВ И ОТМЕТКА ОБ ИХ ОБРАБОТКЕ:
SalePrice - the property's sale price in dollars. This is the target.✅
MSSubClass: The building class✅
MSZoning: The general zoning classification✅
LotFrontage: Linear feet of street connected to property✅
LotArea: Lot size in square feet✅
Street: Type of road access✅
Alley: Type of alley access✅
LotShape: General shape of property✅
LandContour: Flatness of the property✅
Utilities: Type of utilities available✅
LotConfig: Lot configuration✅
LandSlope: Slope of property✅
Neighborhood: Physical locations within Ames city limits✅
Condition1: Proximity to main road or railroad✅
Condition2: Proximity to main road or railroad (if a second is present)✅
BldgType: Type of dwelling✅
HouseStyle: Style of dwelling✅
OverallQual: Overall material and finish quality✅
OverallCond: Overall condition rating✅
YearBuilt: Original construction date✅
YearRemodAdd: Remodel date✅
RoofStyle: Type of roof✅
RoofMatl: Roof material✅
Exterior1st: Exterior covering on house✅
Exterior2nd: Exterior covering on house (if more than one material)✅
MasVnrType: Masonry veneer type✅
MasVnrArea: Masonry veneer area in square feet✅
ExterQual: Exterior material quality✅
ExterCond: Present condition of the material on the exterior✅
Foundation: Type of foundation✅
BsmtQual: Height of the basement✅
BsmtCond: General condition of the basement✅
BsmtExposure: Walkout or garden level basement walls✅
BsmtFinType1: Quality of basement finished area✅
BsmtFinSF1: Type 1 finished square feet✅
BsmtFinType2: Quality of second finished area (if present)✅
BsmtFinSF2: Type 2 finished square feet✅
BsmtUnfSF: Unfinished square feet of basement area✅
TotalBsmtSF: Total square feet of basement area✅
Heating: Type of heating✅
HeatingQC: Heating quality and condition✅
CentralAir: Central air conditioning✅
Electrical: Electrical system✅
1stFlrSF: First Floor square feet✅
2ndFlrSF: Second floor square feet✅
LowQualFinSF: Low quality finished square feet (all floors)✅
GrLivArea: Above grade (ground) living area square feet✅
BsmtFullBath: Basement full bathrooms✅
BsmtHalfBath: Basement half bathrooms✅
FullBath: Full bathrooms above grade✅
HalfBath: Half baths above grade✅
BedroomAbvGr: Number of bedrooms above basement level✅
KitchenAbvGr: Number of kitchens✅
KitchenQual: Kitchen quality✅
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)✅
Functional: Home functionality rating✅
Fireplaces: Number of fireplaces✅
FireplaceQu: Fireplace quality✅
GarageType: Garage location✅
GarageYrBlt: Year garage was built✅
GarageFinish: Interior finish of the garage✅
GarageCars: Size of garage in car capacity✅
GarageArea: Size of garage in square feet✅
GarageQual: Garage quality✅
GarageCond: Garage condition✅
PavedDrive: Paved driveway✅
WoodDeckSF: Wood deck area in square feet✅
OpenPorchSF: Open porch area in square feet✅
EnclosedPorch: Enclosed porch area in square feet✅
3SsnPorch: Three season porch area in square feet✅
ScreenPorch: Screen porch area in square feet✅
PoolArea: Pool area in square feet✅
PoolQC: Pool quality✅
Fence: Fence quality✅
MiscFeature: Miscellaneous feature not covered in other categories✅
MiscVal: $Value of miscellaneous feature✅
MoSold: Month Sold✅
YrSold: Year Sold✅
SaleType: Type of sale✅
SaleCondition: Condition of sale✅
## Итоговая сводная таблица результатов (5-Fold Cross-Validation)

Метрики валидации на 5 фолдах кросс-валидации ($y = \ln(1 + \text{SalePrice})$):
* **RMSLE** — Root Mean Squared Logarithmic Error (основная метрика соревнования Kaggle)
* **MAE ($)** — Средняя абсолютная ошибка в реальных долларах
* **$R^2$** — Коэффициент детерминации (доля объясненной дисперсии цен)

| Модель / Архитектура                            | Тип модели | Этап пайплайна | RMSLE (mean ± std) | MAE ($) | $R^2$ |
|:------------------------------------------------| :---: | :---: | :---: | :---: | :---: |
| **Voting Ensemble (Cat + LGBM + RF)**           | **Ансамбль** | **Tuning & Ensembling** | **0.1238 ± 0.0165** | **$14,670.28** | **0.9021** |
| **Hybrid Blend (Trees 93.8% + ResNet 6.2%)**    | **ML + DL Бленд** | **Tuning & Ensembling** | **0.1247 ± 0.0159** | **$14,710.15** | **0.9015** |
|  **CatBoost (Optuna tuned)**                    | Градиентный бустинг | Tuning | 0.1243 ± 0.0163 | $14,920.40 | 0.8988 |
| **CatBoost (Baseline)**                         | Градиентный бустинг | Baseline | 0.1271 ± 0.0171 | $15,191.06 | 0.8938 |
| **LightGBM (Optuna tuned)**                     | Градиентный бустинг | Tuning | 0.1278 ± 0.0175 | $15,310.22 | 0.8924 |
| **LightGBM (Baseline)**                         | Градиентный бустинг | Baseline | 0.1354 ± 0.0182 | $16,211.23 | 0.8797 |
| **Random Forest (Tuned)**                       | Бэггинг | Tuning | 0.1386 ± 0.0157 | $16,840.10 | 0.8735 |
| **XGBoost (Baseline)**                          | Градиентный бустинг | Baseline | 0.1411 ± 0.0190 | $17,006.12 | 0.8683 |
| **Random Forest (Baseline)**                    | Бэггинг | Baseline | 0.1436 ± 0.0184 | $17,358.55 | 0.8653 |
| **Tabular ResNet (LayerNorm, GELU, Cosine LR)** | **Deep Learning (PyTorch)** | **DL Tuning** | **0.1533 ± 0.0071** | **$18,190.78** | **0.8148** |
| **HouseMLP (BatchNorm, ReLU, AdamW)**           | Deep Learning (PyTorch) | DL Baseline | 0.1541 ± 0.0235 | $18,137.28 | 0.8477 |
| **Ridge Regression (alpha=10.0)**               | Линейная регуляризованная | Baseline | 0.1553 ± 0.0201 | $18,485.49 | 0.8447 |
| **Decision Tree (depth=6)**                     | Одиночное дерево | Baseline | 0.2039 ± 0.0248 | $24,849.56 | 0.7231 |

---

## Структура репозитория

```text
Дома (ML + DL)/
│
├── train.csv                     # Обучающий датасет Ames Housing (1460 строк, 81 признак)
├── test.csv                      # Тестовый датасет Kaggle (1459 строк, 80 признаков)
├── submission.csv                # Финальный сабмит для Kaggle (1459 строк, Id + SalePrice)
│
├── EDA.ipynb                     # Разведочный анализ данных (распределения, корреляции, пропуски)
├── preprocessing.py              # Модульный пайплайн очистки, заполнения пропусков и OHE
│
├── ml_baseline.ipynb             # Сравнение 15 моделей классического ML на 5-Fold CV
├── ml_tuning.ipynb               # Оптимизация (Optuna, RandomizedSearch) и VotingRegressor
│
├── dl_baseline.ipynb             # Базовый PyTorch MLP (BatchNorm, Dropout, масштабирование y)
├── dl_tuning.ipynb               # Tabular ResNet (LayerNorm, GELU, CosineAnnealingLR) и Blending
│
├── config.py                     # Централизованный конфигурационный файл (пути, сиды, сетки параметров)
├── main.py                       # Продакшн-оркестратор: обучение ансамбля, валидация и генерация сабмита
├── requirements.txt              # Зависимости проекта
└── README.md                     # Документация проекта, архитектура и итоговые метрики
```

---

##  Инструкция по запуску

### 1. Установка окружения
```bash
pip install -r requirements.txt
```

### 2. Запуск основного пайплайна
Запуск полного цикла (загрузка данных, модульная предобработка, 5-Fold Bagging обучение лучшего ансамбля моделей, валидация и сохранение финального файла `submission.csv`):
```bash
python main.py
```

