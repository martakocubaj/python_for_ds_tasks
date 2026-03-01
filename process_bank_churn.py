import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def split_train_val(df, target_col):
    """Розбиваємо дані на train та validation"""
    train_df, val_df = train_test_split(
        df, 
        test_size=0.25, 
        random_state=42, 
        stratify=df[target_col]
    )
    return train_df, val_df

def get_feature_col_names():
    """Просто повертаємо списки назв колонок"""
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    categorical_cols = ['Geography', 'Gender', 'NumOfProducts']
    binary_cols = ['HasCrCard', 'IsActiveMember']
    return numeric_cols, categorical_cols, binary_cols

def scale_features(train_df, val_df, numeric_cols):
    """Навчаємо скейлер на трені і змінюємо обидва набори"""
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    
    return train_df, val_df, scaler

def encode_categorical_features(train_df, val_df, categorical_cols):
    """Навчаємо енкодер і створюємо нові колонки (0 та 1)"""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])
    
    return train_df, val_df, encoder, encoded_cols

def preprocess_data(raw_df, scale_numeric=True):
    """Головна функція, яка збирає все разом"""
    target_col = 'Exited'
    
    # 1. Спліт
    train_df, val_df = split_train_val(raw_df, target_col)
    
    # 2. Назви колонок
    num_cols, cat_cols, bin_cols = get_feature_col_names()
    
    # 3. Масштабування (якщо потрібно)
    scaler = None
    if scale_numeric:
        train_df, val_df, scaler = scale_features(train_df, val_df, num_cols)
    
    # 4. Кодування
    train_df, val_df, encoder, enc_cols = encode_categorical_features(train_df, val_df, cat_cols)
    
    # 5. Створюємо фінальні списки колонок для X
    input_cols = num_cols + enc_cols + bin_cols
    
    # 6. Вибираємо тільки потрібні колонки
    X_train = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    
    X_val = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()
    
    # Повертаємо все як один словник
    return {
        'X_train': X_train,
        'train_targets': train_targets,
        'X_val': X_val,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(new_df, scaler, encoder):
    """Обробка нових даних (наприклад, test.csv)"""
    # Отримуємо імена колонок знову
    num_cols, cat_cols, bin_cols = get_feature_col_names()
    
    input_df = new_df.copy()
    
    if scaler is not None:
        input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    enc_cols = list(encoder.get_feature_names_out(cat_cols))
    input_df[enc_cols] = encoder.transform(input_df[cat_cols])
    
    final_cols = num_cols + enc_cols + bin_cols
    return input_df[final_cols]