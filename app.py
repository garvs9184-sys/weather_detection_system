import os
import json
import joblib
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from voice_agent import SUPPORTED_LANGUAGES, generate_voice_response

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'weather_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'model_metrics.json')

# Cache model in memory
_model_bundle = None

def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if os.path.exists(MODEL_PATH):
            _model_bundle = joblib.load(MODEL_PATH)
        else:
            # If not yet trained, run the training pipeline automatically
            from train_model import train_and_export
            train_and_export()
            _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle

WEATHER_METADATA = {
    'Sunny / Clear': {
        'icon': '☀️',
        'theme': 'sunny',
        'headline': 'Clear & Sunny Weather',
        'description': 'Atmospheric skies are clear with abundant solar radiation, dry-to-moderate air, and excellent visibility.',
        'severity': 'Normal Conditions',
        'severity_level': 'low',
        'clothing': 'Light breathable attire, sunglasses, UV sunscreen protection (SPF 30+).',
        'outdoor_activity': 'Great for outdoor athletics, hiking, cycling, and travel.',
        'precautions': 'Stay hydrated if temperature exceeds 30°C.'
    },
    'Partly Cloudy / Overcast': {
        'icon': '⛅',
        'theme': 'cloudy',
        'headline': 'Overcast & Cloud Formations',
        'description': 'Elevated cumulus or stratus cloud layers blocking direct sunlight. Moderate humidity and stable atmospheric pressure.',
        'severity': 'Normal Conditions',
        'severity_level': 'low',
        'clothing': 'Comfortable casual clothes, light jacket or windbreaker in breeze.',
        'outdoor_activity': 'Pleasant conditions for outdoor walking, photography, and sports.',
        'precautions': 'Keep a pocket umbrella handy if clouds darken.'
    },
    'Rainy / Drizzle': {
        'icon': '🌧️',
        'theme': 'rainy',
        'headline': 'Rain & Continuous Precipitation',
        'description': 'Atmospheric saturation leading to continuous precipitation, damp air, and reduced surface visibility.',
        'severity': 'Moderate Alert',
        'severity_level': 'medium',
        'clothing': 'Waterproof raincoat, umbrella, water-resistant footwear.',
        'outdoor_activity': 'Indoor work or leisure recommended; drive carefully on wet roads.',
        'precautions': 'Expect slick roads, reduced braking distance, and ponding water.'
    },
    'Thunderstorm': {
        'icon': '⛈️',
        'theme': 'storm',
        'headline': 'Severe Convective Thunderstorm',
        'description': 'Deep convective storm cells characterized by sharp barometric drops, violent wind gusts, torrential precipitation, and lightning hazard.',
        'severity': 'Severe Weather Warning',
        'severity_level': 'high',
        'clothing': 'Heavy protective gear if required outdoors; stay indoors!',
        'outdoor_activity': 'Halt all outdoor activities immediately. Seek sturdy shelter.',
        'precautions': 'Avoid tall isolated trees, open fields, and metal structures. Unplug sensitive electronics.'
    },
    'Snowy / Blizzard': {
        'icon': '❄️',
        'theme': 'snowy',
        'headline': 'Snowfall & Sub-Zero Precipitation',
        'description': 'Freezing surface temperatures accompanied by frozen precipitation, diminished visibility, and icy conditions.',
        'severity': 'Winter Hazard Warning',
        'severity_level': 'medium',
        'clothing': 'Thermal undergarments, insulated down coat, gloves, woolen beanie, slip-resistant boots.',
        'outdoor_activity': 'Winter sports appropriate with gear; exercise extreme caution driving.',
        'precautions': 'Beware of black ice on asphalt and hypothermia exposure.'
    },
    'Foggy / Mist': {
        'icon': '🌫️',
        'theme': 'foggy',
        'headline': 'Dense Fog & Obscured Visibility',
        'description': 'Saturated boundary layer air near dew point with calm or light winds creating suspended water droplets that severely diminish visibility below 2 km.',
        'severity': 'Visibility Advisory',
        'severity_level': 'medium',
        'clothing': 'Warm layered clothing; moisture-resistant outer layer.',
        'outdoor_activity': 'Proceed with caution when navigating or commuting.',
        'precautions': 'Turn on fog lights or low-beam headlights when driving; reduce transit speed.'
    }
}

PRESET_SCENARIOS = {
    'summer_heatwave': {
        'name': '☀️ Summer Clear Sky (Dubai / Phoenix)',
        'description': 'High temperature, low humidity, intense UV index, zero rain.',
        'values': {
            'temperature': 38.5,
            'humidity': 22.0,
            'wind_speed': 12.0,
            'precipitation': 0.0,
            'cloud_cover': 5.0,
            'atmospheric_pressure': 1014.0,
            'visibility': 18.0,
            'uv_index': 9.5
        }
    },
    'monsoon_downpour': {
        'name': '🌧️ Heavy Monsoon Rain (Mumbai / Singapore)',
        'description': 'Drenching rainfall, very high humidity, dense cloud cover.',
        'values': {
            'temperature': 24.0,
            'humidity': 94.0,
            'wind_speed': 28.0,
            'precipitation': 22.0,
            'cloud_cover': 95.0,
            'atmospheric_pressure': 1003.0,
            'visibility': 4.5,
            'uv_index': 0.8
        }
    },
    'supercell_storm': {
        'name': '⛈️ Severe Thunderstorm (Texas Plains)',
        'description': 'Plunging pressure, gale force gusts, intense downpour.',
        'values': {
            'temperature': 22.5,
            'humidity': 92.0,
            'wind_speed': 68.0,
            'precipitation': 42.0,
            'cloud_cover': 98.0,
            'atmospheric_pressure': 985.0,
            'visibility': 1.8,
            'uv_index': 0.2
        }
    },
    'arctic_blizzard': {
        'name': '❄️ Freezing Blizzard (Toronto / Reykjavik)',
        'description': 'Deep sub-zero temperatures, snowfall, gusty winds, low visibility.',
        'values': {
            'temperature': -8.0,
            'humidity': 86.0,
            'wind_speed': 42.0,
            'precipitation': 6.5,
            'cloud_cover': 90.0,
            'atmospheric_pressure': 1008.0,
            'visibility': 2.1,
            'uv_index': 0.5
        }
    },
    'dense_fog': {
        'name': '🌫️ Radiation Fog (London / San Francisco)',
        'description': 'Extremely high humidity, calm breeze, critically low visibility (< 1 km).',
        'values': {
            'temperature': 8.5,
            'humidity': 98.0,
            'wind_speed': 3.5,
            'precipitation': 0.0,
            'cloud_cover': 65.0,
            'atmospheric_pressure': 1019.0,
            'visibility': 0.4,
            'uv_index': 0.3
        }
    },
    'temperate_overcast': {
        'name': '⛅ Mild Overcast Afternoon (Paris / Seattle)',
        'description': 'High cloud cover, moderate pleasant breeze, dry ground.',
        'values': {
            'temperature': 18.0,
            'humidity': 62.0,
            'wind_speed': 15.0,
            'precipitation': 0.0,
            'cloud_cover': 78.0,
            'atmospheric_pressure': 1016.0,
            'visibility': 14.0,
            'uv_index': 3.2
        }
    }
}

def evaluate_ml_model(feature_dict):
    """Evaluates the Random Forest model on a dictionary of atmospheric parameters."""
    bundle = get_model_bundle()
    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']

    feature_values = [float(feature_dict[feat]) for feat in feature_names]
    x_raw = np.array([feature_values])
    x_scaled = scaler.transform(x_raw)

    pred_class = model.predict(x_scaled)[0]
    probs = model.predict_proba(x_scaled)[0]

    prob_dict = {cls_name: round(float(prob) * 100, 2) for cls_name, prob in zip(model.classes_, probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    confidence = sorted_probs.get(pred_class, 0.0)

    meta = WEATHER_METADATA.get(pred_class, {})
    return pred_class, confidence, sorted_probs, meta

def fetch_live_telemetry_internal(city=None, lat=None, lon=None):
    """Fetches real-time telemetry from Open-Meteo and classifies it with ML."""
    resolved_city = city or "Location"
    
    if city:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=5)
        if geo_resp.status_code == 200:
            geo_data = geo_resp.json()
            if 'results' in geo_data and len(geo_data['results']) > 0:
                first = geo_data['results'][0]
                lat = first['latitude']
                lon = first['longitude']
                country = first.get('country', '')
                resolved_city = f"{first.get('name', city)}, {country}" if country else first.get('name', city)
            else:
                return {'status': 'error', 'message': f'City "{city}" could not be found.'}
        else:
            return {'status': 'error', 'message': 'Failed to reach geocoding service.'}

    if lat is None or lon is None:
        lat, lon = 51.5074, -0.1278
        resolved_city = "London, United Kingdom"

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current="
        f"temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,cloud_cover,visibility"
        f"&daily=uv_index_max&timezone=auto"
    )
    w_resp = requests.get(weather_url, timeout=5)
    if w_resp.status_code != 200:
        return {'status': 'error', 'message': 'Failed to retrieve live weather data.'}

    w_data = w_resp.json()
    current = w_data.get('current', {})
    daily = w_data.get('daily', {})

    temp = float(current.get('temperature_2m', 20.0))
    humidity = float(current.get('relative_humidity_2m', 60.0))
    wind_speed = float(current.get('wind_speed_10m', 10.0))
    precipitation = float(current.get('precipitation', 0.0))
    cloud_cover = float(current.get('cloud_cover', 20.0))
    pressure = float(current.get('surface_pressure', 1013.25))
    visibility_m = float(current.get('visibility', 10000.0))
    visibility_km = max(0.1, round(visibility_m / 1000.0, 1))
    
    uv_index = 3.0
    if 'uv_index_max' in daily and len(daily['uv_index_max']) > 0 and daily['uv_index_max'][0] is not None:
        uv_index = float(daily['uv_index_max'][0])

    telemetry = {
        'temperature': temp,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'precipitation': precipitation,
        'cloud_cover': cloud_cover,
        'atmospheric_pressure': pressure,
        'visibility': visibility_km,
        'uv_index': uv_index
    }

    pred_class, confidence, sorted_probs, meta = evaluate_ml_model(telemetry)

    return {
        'status': 'success',
        'location': {
            'city': resolved_city,
            'latitude': lat,
            'longitude': lon,
            'elevation': w_data.get('elevation', 0),
            'timezone': w_data.get('timezone', 'UTC'),
            'observed_time': current.get('time', '')
        },
        'telemetry': telemetry,
        'ml_prediction': {
            'detected_weather': pred_class,
            'confidence': confidence,
            'probabilities': sorted_probs,
            'metadata': meta
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify({'status': 'success', 'presets': PRESET_SCENARIOS})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({'status': 'success', 'data': data})
    else:
        get_model_bundle()
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({'status': 'success', 'data': data})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        bundle = get_model_bundle()
        feature_names = bundle['feature_names']

        feature_dict = {}
        for feat in feature_names:
            if feat not in data:
                return jsonify({'status': 'error', 'message': f'Missing feature: {feat}'}), 400
            try:
                feature_dict[feat] = float(data[feat])
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': f'Invalid numeric value for {feat}'}), 400

        pred_class, confidence, sorted_probs, meta = evaluate_ml_model(feature_dict)

        return jsonify({
            'status': 'success',
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': sorted_probs,
            'metadata': meta,
            'features_received': feature_dict
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/live_weather', methods=['GET'])
def live_weather():
    city = request.args.get('city', '').strip()
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    try:
        result = fetch_live_telemetry_internal(city=city if city else None, lat=lat, lon=lon)
        if result.get('status') == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/voice_languages', methods=['GET'])
def voice_languages():
    return jsonify({
        'status': 'success',
        'languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/voice_assistant', methods=['POST'])
def voice_assistant():
    try:
        data = request.get_json(force=True) or {}
        query = data.get('query', '').strip()
        lang = data.get('lang', 'en')
        current_telemetry = data.get('current_telemetry', {})
        current_prediction = data.get('current_prediction', 'Sunny / Clear')

        if not query:
            return jsonify({'status': 'error', 'message': 'Empty speech query received.'}), 400

        # Process multilingual voice query
        response_payload = generate_voice_response(
            query=query,
            lang_code=lang,
            current_telemetry=current_telemetry,
            current_prediction=current_prediction,
            live_weather_fn=fetch_live_telemetry_internal,
            ml_predict_fn=evaluate_ml_model
        )

        return jsonify({
            'status': 'success',
            **response_payload
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Weather Detection ML Web Application on http://0.0.0.0:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
