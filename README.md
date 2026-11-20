# 🌤️ AtmosphereAI — Machine Learning Weather Detection System

An end-to-end, production-grade Machine Learning weather detection & classification web application. Powered by a calibrated **Random Forest Ensemble model** trained on rich multi-variable atmospheric telemetry data, featuring a responsive **glassmorphic interactive dashboard**, **real-time live worldwide city weather telemetry integration via Open-Meteo**, and **interactive ML model diagnostics**.

---

## 🌟 Key Features

1. **Precision Machine Learning Engine**:
   - **Ensemble Random Forest Classifier** (120 de-correlated decision trees, max depth 16) with **StandardScaler** normalization.
   - **99.97% test accuracy** evaluated on 15,000 meteorological observation samples.
   - 8 atmospheric telemetry features:
     - `temperature` (°C)
     - `humidity` (%)
     - `wind_speed` (km/h)
     - `precipitation` (mm)
     - `cloud_cover` (%)
     - `atmospheric_pressure` (hPa)
     - `visibility` (km)
     - `uv_index` (0-12)
   - 6 distinct weather classifications:
     - ☀️ **Sunny / Clear**
     - ⛅ **Partly Cloudy / Overcast**
     - 🌧️ **Rainy / Drizzle**
     - ⛈️ **Thunderstorm**
     - ❄️ **Snowy / Blizzard**
     - 🌫️ **Foggy / Mist**

2. **Interactive Weather Simulator**:
   - Real-time what-if sliders for all 8 atmospheric telemetry inputs with immediate ML inference feedback.
   - Dynamic canvas animation engine that shifts visually between sunny sunbeams, rain downpours, storm lightning flashes, snow crystals, and rolling fog.
   - 6 instant climate presets:
     - *Summer Clear Sky* (Dubai / Phoenix)
     - *Monsoon Downpour* (Mumbai / Singapore)
     - *Severe Thunderstorm* (Texas Plains)
     - *Arctic Blizzard* (Reykjavik / Toronto)
     - *Radiation Fog* (London / San Francisco)
     - *Mild Overcast* (Paris / Seattle)
   - Actionable lifestyle advice: Attire recommendations, outdoor activity suitability, and severe weather alert badges.

3. **Live Global City Detection (Satellite & Station Telemetry)**:
   - Query **any city worldwide** (e.g., Tokyo, London, New York, Mumbai, Dubai, Reykjavik, Paris, Sydney).
   - Instant "My Location" GPS geolocation support.
   - Fetches live real-time atmospheric measurements from Open-Meteo and feeds them into our ML classifier to verify and detect ground conditions.

4. **Model Diagnostics & Explainability**:
   - Gini impurity feature importance rankings.
   - Interactive Confusion Matrix heatmap across all 6 classes.
   - Machine learning architecture details and physical boundary logic.

5. **REST API Endpoint**:
   - `POST /api/predict`: Returns weather classification, probabilistic confidence scores, and advisory metadata.
   - `GET /api/live_weather?city=<city>`: Resolves coordinates, pulls live telemetry, and returns ML prediction.
   - `GET /api/metrics`: Returns model accuracy, confusion matrix, and feature importances.
   - In-app interactive JSON editor and cURL documentation.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Installed packages: `Flask`, `scikit-learn`, `numpy`, `pandas`, `joblib`, `requests`

### 2. Run the Web Application
Open PowerShell or Command Prompt:

```powershell
cd "C:\Users\garv8\.gemini\antigravity\scratch\weather-detection-system"
python app.py
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 3. (Optional) Re-train or Fine-tune the ML Model
```powershell
python train_model.py
```
This regenerates the dataset in `data/weather_dataset.csv`, fits the model, and outputs `models/weather_model.joblib` and `models/model_metrics.json`.

---

## 📡 REST API Examples

### Predict Weather from Sensor Data
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 28.5,
    "humidity": 88.0,
    "wind_speed": 45.0,
    "precipitation": 25.0,
    "cloud_cover": 95.0,
    "atmospheric_pressure": 992.0,
    "visibility": 3.0,
    "uv_index": 1.0
  }'
```

**Response:**
```json
{
  "confidence": 99.8,
  "prediction": "Thunderstorm",
  "probabilities": {
    "Thunderstorm": 99.8,
    "Rainy / Drizzle": 0.2,
    "Sunny / Clear": 0.0,
    "Partly Cloudy / Overcast": 0.0,
    "Snowy / Blizzard": 0.0,
    "Foggy / Mist": 0.0
  },
  "metadata": {
    "icon": "⛈️",
    "headline": "Severe Convective Thunderstorm",
    "severity": "Severe Weather Warning",
    "clothing": "Stay indoors!",
    "outdoor_activity": "Halt all outdoor activities immediately."
  },
  "status": "success"
}
```
