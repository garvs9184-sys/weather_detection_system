"""
Weather Detection & Classification Machine Learning Pipeline
Trains an ensemble Random Forest model on comprehensive meteorological feature vectors.
Saves model artifacts, scaler, metadata, and evaluation metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Features:
# 1. temperature: in Celsius (-25 to 45)
# 2. humidity: percentage (10 to 100)
# 3. wind_speed: km/h (0 to 120)
# 4. precipitation: mm (0 to 80)
# 5. cloud_cover: percentage (0 to 100)
# 6. atmospheric_pressure: hPa (960 to 1045)
# 7. visibility: km (0.1 to 20.0)
# 8. uv_index: 0 to 12

FEATURE_NAMES = [
    'temperature',
    'humidity',
    'wind_speed',
    'precipitation',
    'cloud_cover',
    'atmospheric_pressure',
    'visibility',
    'uv_index'
]

CLASSES = [
    'Sunny / Clear',
    'Partly Cloudy / Overcast',
    'Rainy / Drizzle',
    'Thunderstorm',
    'Snowy / Blizzard',
    'Foggy / Mist'
]

def generate_synthetic_weather_data(n_samples_per_class=2000):
    """
    Generates realistic, physically-grounded meteorological records
    for 6 distinct weather classifications with natural variance and noise.
    """
    records = []
    
    # 1. Sunny / Clear
    for _ in range(n_samples_per_class):
        temp = np.random.normal(26.0, 7.5) # typically pleasant to hot
        temp = np.clip(temp, 5.0, 48.0)
        humidity = np.random.uniform(15.0, 55.0)
        wind = np.random.exponential(9.0)
        wind = np.clip(wind, 0.0, 35.0)
        precip = 0.0 # No rain
        cloud = np.random.uniform(0.0, 25.0)
        pressure = np.random.normal(1018.0, 4.0)
        vis = np.random.uniform(12.0, 20.0)
        uv = np.random.uniform(5.0, 11.5)
        records.append([temp, humidity, wind, precip, cloud, pressure, vis, uv, 'Sunny / Clear'])
        
    # 2. Partly Cloudy / Overcast
    for _ in range(n_samples_per_class):
        temp = np.random.normal(19.0, 6.5)
        temp = np.clip(temp, 2.0, 36.0)
        humidity = np.random.uniform(40.0, 75.0)
        wind = np.random.normal(14.0, 6.0)
        wind = np.clip(wind, 2.0, 45.0)
        precip = np.random.choice([0.0, 0.05, 0.1], p=[0.85, 0.1, 0.05])
        cloud = np.random.uniform(40.0, 95.0)
        pressure = np.random.normal(1014.0, 5.0)
        vis = np.random.uniform(9.0, 18.0)
        uv = np.random.uniform(1.5, 6.0)
        records.append([temp, humidity, wind, precip, cloud, pressure, vis, uv, 'Partly Cloudy / Overcast'])
        
    # 3. Rainy / Drizzle
    for _ in range(n_samples_per_class):
        temp = np.random.normal(16.0, 6.0)
        temp = np.clip(temp, 3.0, 32.0) # liquid precipitation
        humidity = np.random.uniform(78.0, 99.0)
        wind = np.random.normal(22.0, 9.0)
        wind = np.clip(wind, 4.0, 60.0)
        precip = np.random.exponential(7.0) + 1.2
        precip = np.clip(precip, 1.2, 45.0)
        cloud = np.random.uniform(75.0, 100.0)
        pressure = np.random.normal(1005.0, 6.0)
        vis = np.random.uniform(2.5, 9.0)
        uv = np.random.uniform(0.0, 2.5)
        records.append([temp, humidity, wind, precip, cloud, pressure, vis, uv, 'Rainy / Drizzle'])
        
    # 4. Thunderstorm
    for _ in range(n_samples_per_class):
        temp = np.random.normal(23.0, 5.0)
        temp = np.clip(temp, 14.0, 38.0)
        humidity = np.random.uniform(82.0, 100.0)
        wind = np.random.normal(55.0, 16.0) # severe gusty winds
        wind = np.clip(wind, 32.0, 115.0)
        precip = np.random.exponential(25.0) + 15.0 # intense downpour
        precip = np.clip(precip, 15.0, 95.0)
        cloud = np.random.uniform(88.0, 100.0)
        pressure = np.random.normal(988.0, 8.0) # sharp low-pressure system
        vis = np.random.uniform(0.8, 5.0) # reduced visibility
        uv = np.random.uniform(0.0, 1.5)
        records.append([temp, humidity, wind, precip, cloud, pressure, vis, uv, 'Thunderstorm'])
        
    # 5. Snowy / Blizzard
    for _ in range(n_samples_per_class):
        temp = np.random.normal(-5.0, 5.5)
        temp = np.clip(temp, -26.0, 1.2) # freezing temperatures
        humidity = np.random.uniform(65.0, 96.0)
        wind = np.random.normal(25.0, 12.0)
        wind = np.clip(wind, 2.0, 85.0)
        precip = np.random.exponential(5.0) + 0.8 # snow liquid equivalent
        precip = np.clip(precip, 0.8, 35.0)
        cloud = np.random.uniform(70.0, 100.0)
        pressure = np.random.normal(1010.0, 8.0)
        vis = np.random.uniform(0.5, 7.0)
        uv = np.random.uniform(0.0, 2.5)
        records.append([temp, humidity, wind, precip, cloud, pressure, vis, uv, 'Snowy / Blizzard'])
        
    # 6. Foggy / Mist
    for _ in range(n_samples_per_class):
        temp = np.random.normal(9.0, 5.0)
        temp = np.clip(temp, -2.0, 22.0)
        humidity = np.random.uniform(92.0, 100.0) # saturated air near dew point
        wind = np.random.uniform(0.5, 11.0) # calm stagnant air essential for radiation/advection fog
        precip = np.random.choice([0.0, 0.1, 0.2], p=[0.7, 0.2, 0.1]) # trace condensation
        cloud = np.random.uniform(35.0, 85.0)
        pressure = np.random.normal(1017.0, 5.0)
        vis = np.random.uniform(0.1, 1.8) # severely restricted visibility (<2 km by definition)
        uv = np.random.uniform(0.0, 2.0)
        records.append([temp, humidity, wind, precip, cloud, pressure, vis, uv, 'Foggy / Mist'])
        
    df = pd.DataFrame(records, columns=FEATURE_NAMES + ['weather_condition'])
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def train_and_export():
    print("Generating comprehensive meteorological training dataset...")
    df = generate_synthetic_weather_data(n_samples_per_class=2500)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'weather_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path} with {len(df)} records across {len(CLASSES)} classes.")
    
    X = df[FEATURE_NAMES]
    y = df['weather_condition']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training split: {X_train.shape[0]} samples | Testing split: {X_test.shape[0]} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Fitting Random Forest Classifier with 120 trees...")
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES).tolist()
    
    feature_importances = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in rf_model.feature_importances_]))
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    
    print(f"\n================ MODEL EVALUATION ================")
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print(f"Feature Importances:")
    for feat, imp in sorted_importances.items():
        print(f"  - {feat:22s}: {imp * 100:.2f}%")
    print("===================================================\n")
    
    # Pack model bundle
    bundle = {
        'model': rf_model,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'classes': CLASSES,
        'accuracy': float(acc),
        'feature_importances': sorted_importances
    }
    
    model_path = os.path.join(models_dir, 'weather_model.joblib')
    joblib.dump(bundle, model_path)
    print(f"Saved complete ML pipeline bundle to {model_path}")
    
    metrics = {
        'accuracy': round(float(acc) * 100, 2),
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_names': FEATURE_NAMES,
        'classes': CLASSES,
        'feature_importances': sorted_importances,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model metrics to {metrics_path}")

if __name__ == '__main__':
    train_and_export()
