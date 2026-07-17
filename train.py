import numpy as np
import pandas as pd
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import kagglehub

def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("jsphyg/weather-dataset-rattle-package")
    csv_file_path = os.path.join(path, 'weatherAUS.csv')
    
    print("Loading dataset...")
    df = pd.read_csv(csv_file_path)
    print(f"Original shape: {df.shape}")
    
    # 1. Drop columns with high missing values (>40%)
    threshold = 0.4
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
    print(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    # 2. Drop Date and Location
    df = df.drop(columns=['Date', 'Location'], errors='ignore')
    
    # 3. Drop rows where Target is missing
    df = df.dropna(subset=['RainTomorrow'])
    
    # 4. Handle Categorical columns and Encoders
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {cat_cols}")
    
    encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # 5. Handle Numerical missing values (Median imputation)
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)
    
    # Splitting Features and Target
    X = df.drop(columns=['RainTomorrow'])
    y = df['RainTomorrow']
    feature_columns = list(X.columns)
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Build ANN Model
    n_features = X_train_scaled.shape[1]
    print(f"Number of input features: {n_features}")
    print("Building ANN model...")
    
    ann_model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    ann_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Train
    print("Training ANN model...")
    ann_model.fit(
        X_train_scaled, y_train,
        epochs=30,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save Artifacts
    print("Saving model and preprocessors...")
    ann_model.save('rain_prediction_ann.h5')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    # Save a small subset of training data for SHAP background
    # SHAP explainer requires a background dataset
    background_data = pd.DataFrame(X_train_scaled[:100], columns=feature_columns)
    joblib.dump(background_data, 'shap_background.pkl')
    
    print("Training and saving completed successfully!")

if __name__ == '__main__':
    main()
