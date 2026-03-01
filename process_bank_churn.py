import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def split_train_val(df: pd.DataFrame, target_col: str, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розбиває вхідний датафрейм на тренувальний та валідаційний набори з використанням стратифікації.
    """
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df[target_col]
    )
    return train_df, val_df

def get_feature_col_names(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Визначає назви числових, категоріальних та бінарних колонок.
    """
    # Згідно з вашою логікою: Surname та ID видаляємо
    # NumOfProducts переносимо в категоріальні
    # HasCrCard та IsActiveMember - бінарні
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    categorical_cols = ['Geography', 'Gender', 'NumOfProducts']
    binary_cols = ['HasCrCard', 'IsActiveMember']
    
    return numeric_cols, categorical_cols, binary_cols

def scale_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Масштабує числові ознаки за допомогою StandardScaler.
    """
    scaler = StandardScaler().fit(train_df[numeric_cols])
    
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    
    return train_df, val_df, scaler

def encode_categorical_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Кодує категоріальні ознаки за допомогою OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])
    
    return train_df, val_df, encoder, encoded_cols

def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Dict[str, Any]:
    """
    Основна функція препроцесингу, що готує дані для навчання моделі.
    
    Args:
        raw_df: Початковий DataFrame (train.csv)
        scale_numeric: Чи потрібно масштабувати числові ознаки (StandardScaler)
        
    Returns:
        Словник з обробленими даними та об'єктами препроцесингу.
    """
    target_col = 'Exited'
    
    # 1. Розбиття на набори
    train_df, val_df = split_train_val(raw_df, target_col)
    
    # 2. Визначення типів колонок
    numeric_cols, categorical_cols, binary_cols = get_feature_col_names(train_df)
    
    # 3. Масштабування (опціонально)
    scaler = None
    if scale_numeric:
        train_df, val_df, scaler = scale_features(train_df, val_df, numeric_cols)
    
    # 4. Кодування категоріальних змінних
    train_df, val_df, encoder, encoded_cols = encode_categorical_features(train_df, val_df, categorical_cols)
    
    # 5. Формування фінальних наборів X та y
    input_cols = numeric_cols + encoded_cols + binary_cols
    
    X_train = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    
    X_val = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()
    
    return {
        'X_train': X_train,
        'train_targets': train_targets,
        'X_val': X_val,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(
    new_df: pd.DataFrame, 
    scaler: Optional[StandardScaler], 
    encoder: OneHotEncoder,
    numeric_cols: List[str],
    categorical_cols: List[str],
    binary_cols: List[str]
) -> pd.DataFrame:
    """
    Функція для обробки нових даних (наприклад, test.csv) перед передбаченням.
    Використовує вже навчені скейлер та енкодер.
    """
    input_df = new_df.copy()
    
    # Масштабування
    if scaler is not None:
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Кодування
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    
    # Відбір фінальних колонок
    input_cols = numeric_cols + encoded_cols + binary_cols
    return input_df[input_cols]