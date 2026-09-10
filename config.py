"""
Конфигурация проекта House price Learning & Deep Learning.
Здесь задаются пути к данным, сиды, настройки кросс-валидации и параметры моделей.
"""

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Пути к файлам
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TRAIN_PROCESSED_PATH = "train_processed.csv"
SUBMISSION_PATH = "submission.csv"

# Общие настройки
TARGET_COL = "SalePrice"
ID_COL = "Id"
RANDOM_STATE = 42

# Модели, требующие масштабирования признаков (Линейные + KNN)
MODELS_SCALED = {
    'Ridge (alpha=10)': Ridge(alpha=10.0, random_state=RANDOM_STATE),
    'Lasso (alpha=0.0005)': Lasso(alpha=0.0005, random_state=RANDOM_STATE),
    'ElasticNet (alpha=0.001)': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE),
    'KNN (k=5, uniform)': KNeighborsRegressor(n_neighbors=5, weights='uniform'),
    'KNN (k=5, distance)': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'KNN (k=10, distance)': KNeighborsRegressor(n_neighbors=10, weights='distance'),
    'KNN (k=10, manhattan)': KNeighborsRegressor(n_neighbors=10, weights='distance', metric='manhattan')
}

# Древесные модели и бустинги (масштабирование НЕ нужно)
MODELS_UNSCALED = {
    'Decision Tree (default)': DecisionTreeRegressor(random_state=RANDOM_STATE),
    'Decision Tree (depth=6)': DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    'Random Forest (100 trees)': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Random Forest (200 trees, sqrt)': RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1),
    'Random Forest (200 trees, depth=15)': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    'CatBoost (500 trees)': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=RANDOM_STATE, verbose=0),
    'LightGBM (500 trees)': LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, verbosity=-1),
    'XGBoost (500 trees)': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
}

# Кросс-валидация
N_SPLITS = 5
SHUFFLE = True

# Список колонок для удаления, декодирования и преобразование (из EDA)

DROP_COLUMNS = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold']

ORDINAL_COLUMNS = [
    'GarageQual',
    'GarageCond',
    'LotShape',
    'LandSlope',
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'HeatingQC',
    'KitchenQual',
    'Functional',
    'FireplaceQu',
    'GarageFinish',
    'PavedDrive'
]

ORDINAL_MAPPINGS = {
    'GarageQual': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },

    'GarageCond': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },

    'LotShape': {
        'Reg': 0,
        'IR1': 1,
        'IR2': 2,
        'IR3': 3,
    },

    'LandSlope': {
        'Gtl': 0,
        'Mod': 1,
        'Sev': 2,
    },
    'ExterQual': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },
    'ExterCond': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },
    'BsmtQual': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },
    'BsmtCond': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },
    'BsmtExposure': {
        'No': 1,
        'Mn': 2,
        'Av': 3,
        'Gd': 4,
    },
    'BsmtFinType1': {
        'Unf': 1,
        'LwQ': 2,
        'Rec': 3,
        'BLQ': 4,
        'ALQ': 5,
        'GLQ': 6,
    },
    'BsmtFinType2': {
        'Unf': 1,
        'LwQ': 2,
        'Rec': 3,
        'BLQ': 4,
        'ALQ': 5,
        'GLQ': 6,
    },
    'HeatingQC': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },
    'KitchenQual': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },
    'Functional': {
        'Sev': 1,
        'Maj2': 2,
        'Maj1': 3,
        'Mod': 4,
        'Min2': 5,
        'Min1': 6,
        'Typ': 7,
    },
    'FireplaceQu': {
        'Po': 1,
        'Fa': 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5,
    },
    'GarageFinish': {
        'Unf': 1,
        'RFn': 2,
        'Fin': 3,
    },
    'PavedDrive': {
        'N': 0,
        'P': 1,
        'Y': 2,
    }
}

CATEGORICAL_COLUMNS = [
    'MSZoning',
    'Street',
    'Alley',
    'LandContour',
    'Utilities',
    'LotConfig',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'Foundation',
    'Heating',
    'CentralAir',
    'Electrical',
    'GarageType',
    'Fence',
    'MiscFeature',
    'SaleType',
    'SaleCondition',
    'MasVnrType'

]


# =============================================================
# НАСТРОЙКИ ТЮНИНГА ГИПЕРПАРАМЕТРОВ RandomForest
# =============================================================

# Сетка значений гиперпараметров для перебора
RF_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.3, 0.5, 0.7]
}
# Параметры самого процесса поиска RandomizedSearchCV
RF_SEARCH_CONFIG = {
    'n_iter': 25,                                  # количество случайных попыток
    'scoring': 'neg_root_mean_squared_error',       # метрика
    'random_state': RANDOM_STATE,                   # сид случайности
    'n_jobs': -1,                                   # ядра процессора
    'verbose': 1                                    # подробность вывода логов
}

# =============================================================
# ТНАСТРОЙКИ ТЮНИНГА ДЛЯ LIGHTGBM (OPTUNA)
# =============================================================

OPTUNA_N_TRIALS = 30  # Количество попыток умного поиска

LGBM_PARAM_DIST = {
    'num_leaves': (15, 63),  # диапазон количества листьев
    'learning_rate': (0.01, 0.1),  # диапазон скорости обучения
    'n_estimators': (300, 800, 100),  # от 300 до 800 с шагом 100
    'colsample_bytree': (0.4, 0.8),  # доля колонок на дерево
    'subsample': (0.5, 0.9),  # доля строк на дерево
    'reg_alpha': (1e-3, 10.0),  # диапазон L1-регуляризации
    'reg_lambda': (1e-3, 10.0)  # диапазон L2-регуляризации
}

# =============================================================
# НАСТРОЙКИ ТЮНИНГА ДЛЯ CATBOOST (OPTUNA)
# =============================================================
CATBOOST_PARAM_DIST = {
    'depth': (4, 8),                            # глубина симметричных деревьев
    'learning_rate': (0.01, 0.1),               # скорость обучения
    'iterations': (500, 1000, 100),             # от 500 до 1000 с шагом 100
    'l2_leaf_reg': (1.0, 10.0),                 # L2-регуляризация листьев
    'random_strength': (0.1, 10.0),             # шум при выборе сплитов
    'bagging_temperature': (0.0, 1.0)           # интенсивность бэггинга
}

# =============================================================
# НАСТРОЙКИ АНСАМБЛЯ (VotingRegressor)
# =============================================================
# Веса моделей в ансамбле: [CatBoost, LightGBM, Random Forest]
VOTING_WEIGHTS = [0.50, 0.35, 0.15]

# ===================================== БЛОК НЕЙРОННЫХ СЕТЕЙ (DL)==================================================

DL_CONFIG = {
    "epochs": 60,
    "batch_size": 32,
    "lr": 0.001,
    "hidden_dim": 64, # мы сделаем гибко от hidden_dim
    "dropout": 0.2
}

# Продвинутые параметры для тюнинга нейросети (Tabular ResNet + Cosine Scheduler)
DL_TUNING_CONFIG = {
    "epochs": 100,            # Увеличиваем число эпох (планировщик LR поможет сойтись глубже)
    "batch_size": 32,         # Оптимальный размер батча для табличных данных
    "lr": 0.001,              # Начальная скорость обучения (Peak LR)
    "weight_decay": 0.01,     # L2-регуляризация в оптимизаторе AdamW
    "hidden_dim": 128,        # Размер скрытого пространства (было 64, увеличиваем емкость)
    "num_blocks": 2,          # Количество остаточных блоков (ResNet blocks)
    "dropout": 0.15,          # Вероятность зануления нейронов для борьбы с переобучением
    "eta_min": 1e-5           # Минимальный learning rate в конце обучения (для CosineAnnealing)
}

# =============================================================
# ФИНАЛЬНЫЕ ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ (РЕЗУЛЬТАТ ТЮНИНГА)
# =============================================================
BEST_CAT_PARAMS = {
    'depth': 6,
    'learning_rate': 0.0489,
    'iterations': 900,
    'l2_leaf_reg': 1.3478,
    'random_strength': 3.2359,
    'bagging_temperature': 0.3140,
    'random_state': RANDOM_STATE,
    'verbose': 0
}
BEST_LGBM_PARAMS = {
    'num_leaves': 16,
    'learning_rate': 0.0171,
    'n_estimators': 800,
    'colsample_bytree': 0.4700,
    'subsample': 0.6811,
    'reg_alpha': 0.0029,
    'reg_lambda': 0.0112,
    'random_state': RANDOM_STATE,
    'verbosity': -1,
    'n_jobs': -1
}
BEST_RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 25,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_features': 0.3,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}