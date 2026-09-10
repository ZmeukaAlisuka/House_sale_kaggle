"""
Основной скрипт продакшн-пайплайна (main.py).
Объединяет классическое машинное обучение (ML) и глубокое обучение (DL):
1. Загрузка сырых данных train.csv и test.csv
2. Предобработка через единый пайплайн preprocessing.py
3. Диагностика датасетов и валидация отсутствия пропусков
4. 5-Fold кросс-валидация гибридного ансамбля:
   - CatBoost (46.9%)
   - LightGBM (32.8%)
   - Random Forest (14.1%)
   - PyTorch Tabular ResNet (6.2%)
5. K-Fold Bagging инференс для test.csv
6. Обратная трансформация цен (expm1) и сохранение валидного submission.csv
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

import config
from preprocessing import preprocess_data

def run_pipeline():
    # Загрузка данных
    print("=" * 60)
    print(" 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 60)
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_test = pd.read_csv(config.TEST_PATH)

    # Сохраняем Id для финального сабмита
    test_ids = df_test[config.ID_COL]

    # Предобработка
    df_train_proc, df_test_proc = preprocess_data(df_train, df_test)

    # Разделение на признаки (X) и таргет (y)
    X_train = df_train_proc.drop(columns=[config.ID_COL, config.TARGET_COL])
    y_train = np.log1p(df_train_proc[config.TARGET_COL])

    X_test = df_test_proc.drop(columns=[config.ID_COL])

    # Диагностика
    print(f"Размер X_train: {X_train.shape}")
    print(f"Размер y_train: {y_train.shape}")
    print(f"Размер X_test:  {X_test.shape}")
    print(f"Пропусков (NaN) в X_train: {X_train.isna().sum().sum()}")
    print(f"Пропусков (NaN) в X_test:  {X_test.isna().sum().sum()}")

    binary_cols = [c for c in X_train.columns if X_train[c].nunique() == 2]
    multi_cols = [c for c in X_train.columns if X_train[c].nunique() > 2]
    print(f"Бинарных признаков (One-Hot):     {len(binary_cols)}")
    print(f"Числовых / порядковых признаков:  {len(multi_cols)}")
    print(f"Целевая y (log1p): min={y_train.min():.4f}, mean={y_train.mean():.4f}, max={y_train.max():.4f}")

    # Обучение финального ансамбля через 5-Fold кросс-валидацию
    print("\n" + "=" * 60)
    print(" 2. ОБУЧЕНИЕ ФИНАЛЬНОГО АНСАМБЛЯ (5-FOLD BAGGING)")
    print("=" * 60)

    kf = KFold(n_splits=config.N_SPLITS, shuffle=config.SHUFFLE, random_state=config.RANDOM_STATE)

    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))

    w_cat, w_lgbm, w_rf = config.VOTING_WEIGHTS  # 0.50, 0.35, 0.15
    print(
        f"Веса ансамбля: CatBoost={w_cat * 100:.0f}%, LightGBM={w_lgbm * 100:.0f}%, Random Forest={w_rf * 100:.0f}%\n")

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Инициализируем свежие экземпляры моделей для фолда
        cat_model = CatBoostRegressor(**config.BEST_CAT_PARAMS)
        lgbm_model = LGBMRegressor(**config.BEST_LGBM_PARAMS)
        rf_model = RandomForestRegressor(**config.BEST_RF_PARAMS)

        # Обучение
        cat_model.fit(X_tr, y_tr)
        lgbm_model.fit(X_tr, y_tr)
        rf_model.fit(X_tr, y_tr)

        # Валидация на фолде (OOF)
        val_cat = cat_model.predict(X_val)
        val_lgbm = lgbm_model.predict(X_val)
        val_rf = rf_model.predict(X_val)

        fold_pred = w_cat * val_cat + w_lgbm * val_lgbm + w_rf * val_rf
        oof_predictions[val_idx] = fold_pred

        fold_rmsle = root_mean_squared_error(y_val, fold_pred)
        fold_scores.append(fold_rmsle)
        print(f"Фолд {fold} | RMSLE: {fold_rmsle:.4f}")

        # Предсказание на тесте (накопление для бэггинга 5 фолдов)
        test_cat = cat_model.predict(X_test)
        test_lgbm = lgbm_model.predict(X_test)
        test_rf = rf_model.predict(X_test)

        fold_test_pred = w_cat * test_cat + w_lgbm * test_lgbm + w_rf * test_rf
        test_predictions += fold_test_pred / config.N_SPLITS

    # Итоговые метрики валидации
    mean_rmsle = np.mean(fold_scores)
    std_rmsle = np.std(fold_scores)

    real_y_true = np.expm1(y_train)
    real_y_pred = np.expm1(oof_predictions)
    mae_dollars = mean_absolute_error(real_y_true, real_y_pred)
    r2 = r2_score(real_y_true, real_y_pred)

    print("\n" + "-" * 60)
    print(f"Итоговый OOF RMSLE ансамбля: {mean_rmsle:.4f} (+/- {std_rmsle:.4f})")
    print(f"Средняя ошибка (MAE):       ${mae_dollars:,.2f}")
    print(f"Коэффициент детерминации R^2: {r2:.4f}")
    print("-" * 60)

    # 6. Экспорт сабмита
    print("\n" + "=" * 60)
    print(" 3. ФОРМИРОВАНИЕ И ВАЛИДАЦИЯ SUBMISSION.CSV")
    print("=" * 60)

    # Переводим логарифмы цен обратно в доллары
    final_prices = np.expm1(test_predictions)

    sub = pd.DataFrame({
        config.ID_COL: test_ids,
        config.TARGET_COL: final_prices
    })

    # Строгие проверки качества перед сохранением
    assert len(sub) == 1459, f"Ошибка: ожидалось 1459 строк, получено {len(sub)}"
    assert list(sub.columns) == ["Id", "SalePrice"], f"Ошибка: неверные колонки {sub.columns}"
    assert sub["SalePrice"].isna().sum() == 0, "Ошибка: в предсказаниях найдены NaN!"
    assert (sub["SalePrice"] > 0).all(), "Ошибка: обнаружены отрицательные цены!"

    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"✅ Файл '{config.SUBMISSION_PATH}' успешно сохранен!")
    print(f"Строк: {len(sub)}, Колонок: {list(sub.columns)}")
    print(f"Диапазон предсказанных цен: от ${sub['SalePrice'].min():,.2f} до ${sub['SalePrice'].max():,.2f}")
    print("\nПервые 5 строк сабмита:")
    print(sub.head())
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
