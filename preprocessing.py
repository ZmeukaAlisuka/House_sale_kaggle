"""Обработка данных перед загрузкой в модель"""

import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import OneHotEncoder


# Обработка потерянных числовых признаков:
def fill_missing_numerical_features( df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Заполняем LotFrontage медианой по району (Neighborhood)
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda group: group.fillna(group.median())
    )
    # Страховка на случай, если где-то остался NaN:
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    # 2. Список числовых признаков, где NaN означает отсутствие объекта (ставим 0)
    zero_fill_cols = [
        'MasVnrArea',
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        'BsmtFullBath',
        'BsmtHalfBath',
        'GarageCars',
        'GarageArea'
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

# Ordinal Encoding

def apply_ordinal_mappings(
    df: pd.DataFrame,
    mappings: dict
) -> pd.DataFrame:

    df = df.copy()

    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = (
                df[column]
                .map(mapping)
                .fillna(0)
            )

    return df


# One-hot Encoding
def apply_one_hot_encoding(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_columns: list
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None

    # Признаки, где пропуск в тесте — это случайность (заполняем самым частым значением из train):
    mode_fill_cols = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'SaleType']
    for col in mode_fill_cols:
        if col in train_df.columns:
            mode_val = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(mode_val)
            if test_df is not None and col in test_df.columns:
                test_df[col] = test_df[col].fillna(mode_val)

    train_df[categorical_columns] = train_df[categorical_columns].fillna('None')
    if test_df is not None:
        test_df[categorical_columns] = test_df[categorical_columns].fillna('None')

    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        drop='first'  # аналог drop_first=True в get_dummies
    )

    # Запоминаем категории train и трансформируем train в массив нулей и единиц:
    train_encoded = encoder.fit_transform(train_df[categorical_columns])

    # Получаем названия колонок:
    feature_names = encoder.get_feature_names_out(categorical_columns)
    # Превращаем массив обратно в DataFrame:
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=feature_names,
        index=train_df.index,
        dtype=int
    )
    # Убираем старый текст и приклеиваем новые 0 и 1:
    train_df = train_df.drop(columns=categorical_columns)
    train_df = pd.concat([train_df, train_encoded_df], axis=1)

    # 5. Если передан тест — переводим его по тем же правилам (только transform):
    if test_df is not None:
        test_encoded = encoder.transform(test_df[categorical_columns])
        test_encoded_df = pd.DataFrame(
            test_encoded,
            columns=feature_names,
            index=test_df.index,
            dtype=int
        )
        test_df = test_df.drop(columns=categorical_columns)
        test_df = pd.concat([test_df, test_encoded_df], axis=1)

    return train_df, test_df


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Возраст дома и время с момента последней реконструкции
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = (df['YrSold'] - df['YearRemodAdd'])

    # Флаг наличия гаража (1 - есть, 0 - нет)
    df['HasGarage'] = df['GarageYrBlt'].notna().astype(int)

    # Исправление опечатки 2207 года в тесте (все продажи были до 2010 года)
    df.loc[df['GarageYrBlt'] > 2010, 'GarageYrBlt'] = 2007

    # Возраст гаража (если гаража нет-> 0, .clip(lower=0)-защитит от случайных отрицательных значений)
    df['GarageAge'] = (df['YrSold'] - df['GarageYrBlt']).fillna(0).clip(lower=0)

    # Циклические признаки месяца продажи (зима-весна-лето-осень)
    df['MoSold_sin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
    df['MoSold_cos'] = np.cos(2 * np.pi * df['MoSold'] / 12)

    # Удаляем исходные временные столбцы (заданы в config.DROP_COLUMNS)
    df = df.drop(columns=config.DROP_COLUMNS)
    return df


def preprocess_data(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None

    # Заполнение числовых пропусков
    train_df = fill_missing_numerical_features(train_df)
    if test_df is not None:
        test_df = fill_missing_numerical_features(test_df)

    # Создание возрастных признаков
    train_df = create_age_features(train_df)
    if test_df is not None:
        test_df = create_age_features(test_df)

    # PoolQC: NaN -> 0, любое значение -> 1
    train_df['PoolQC'] = train_df['PoolQC'].notna().astype(int)
    if test_df is not None:
        test_df['PoolQC'] = test_df['PoolQC'].notna().astype(int)

    # Ordinal Encoding
    train_df = apply_ordinal_mappings(train_df, config.ORDINAL_MAPPINGS)
    if test_df is not None:
        test_df = apply_ordinal_mappings(test_df, config.ORDINAL_MAPPINGS)

    # One-hot Encoding
    train_df, test_df = apply_one_hot_encoding(
        train_df,
        test_df,
        config.CATEGORICAL_COLUMNS
    )
    return train_df, test_df



