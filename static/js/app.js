/**
 * AtmosphereAI - Client Application Logic
 * Dynamic Canvas Weather Simulation, Real-time ML Inference, Live Telemetry, & Analytics
 */

// Global State
let currentTheme = 'sunny';
let currentWeatherType = 'Sunny / Clear';
let debounceTimer = null;
let modelMetricsCache = null;

// DOM Elements
const bodyEl = document.body;
const weatherCanvas = document.getElementById('weatherCanvas');
const weatherForm = document.getElementById('weatherForm');
const realtimeToggle = document.getElementById('realtimeToggle');
const presetChips = document.querySelectorAll('.preset-chip');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

// Feature sliders configuration
const FEATURE_KEYS = [
  'temperature',
  'humidity',
  'wind_speed',
  'precipitation',
  'cloud_cover',
  'atmospheric_pressure',
  'visibility',
  'uv_index'
];

/* =================================================================
   1. Dynamic Background Weather Canvas Animation Engine
   ================================================================= */
class WeatherParticleSystem {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.particles = [];
    this.currentMode = 'Sunny / Clear';
    this.lightningAlpha = 0;
    this.lastLightning = Date.now();

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.initParticles();
    this.animate();
  }

  resize() {
    this.width = this.canvas.width = window.innerWidth;
    this.height = this.canvas.height = window.innerHeight;
    this.initParticles();
  }

  setMode(mode) {
    if (this.currentMode !== mode) {
      this.currentMode = mode;
      this.initParticles();
    }
  }

  initParticles() {
    this.particles = [];
    const count = this.getParticleCount();

    for (let i = 0; i < count; i++) {
      this.particles.push(this.createParticle(true));
    }
  }

  getParticleCount() {
    switch (this.currentMode) {
      case 'Rainy / Drizzle': return 160;
      case 'Thunderstorm': return 260;
      case 'Snowy / Blizzard': return 120;
      case 'Foggy / Mist': return 40;
      case 'Partly Cloudy / Overcast': return 25;
      case 'Sunny / Clear':
      default:
        return 35;
    }
  }

  createParticle(randomY = false) {
    const y = randomY ? Math.random() * this.height : -20;
    const x = Math.random() * this.width;

    switch (this.currentMode) {
      case 'Rainy / Drizzle':
        return {
          x: x,
          y: y,
          speedY: 10 + Math.random() * 8,
          speedX: -1 - Math.random() * 2,
          length: 12 + Math.random() * 12,
          opacity: 0.25 + Math.random() * 0.45
        };
      case 'Thunderstorm':
        return {
          x: x,
          y: y,
          speedY: 16 + Math.random() * 12,
          speedX: -3 - Math.random() * 4,
          length: 18 + Math.random() * 16,
          opacity: 0.4 + Math.random() * 0.5
        };
      case 'Snowy / Blizzard':
        return {
          x: x,
          y: y,
          speedY: 1.2 + Math.random() * 2.2,
          speedX: Math.sin(Math.random() * Math.PI) * 1.5,
          radius: 1.8 + Math.random() * 3.5,
          opacity: 0.4 + Math.random() * 0.5,
          swing: Math.random() * 3
        };
      case 'Foggy / Mist':
        return {
          x: Math.random() * this.width,
          y: Math.random() * this.height,
          speedX: 0.2 + Math.random() * 0.5,
          speedY: (Math.random() - 0.5) * 0.15,
          radius: 60 + Math.random() * 90,
          opacity: 0.04 + Math.random() * 0.08
        };
      case 'Partly Cloudy / Overcast':
        return {
          x: Math.random() * this.width,
          y: Math.random() * (this.height * 0.6),
          speedX: 0.3 + Math.random() * 0.6,
          speedY: (Math.random() - 0.5) * 0.1,
          radius: 50 + Math.random() * 80,
          opacity: 0.05 + Math.random() * 0.08
        };
      case 'Sunny / Clear':
      default:
        return {
          x: Math.random() * this.width,
          y: Math.random() * this.height,
          speedY: -0.3 - Math.random() * 0.5,
          speedX: (Math.random() - 0.5) * 0.4,
          radius: 1.5 + Math.random() * 2.5,
          opacity: 0.15 + Math.random() * 0.35,
          pulse: Math.random() * Math.PI
        };
    }
  }

  animate() {
    this.ctx.clearRect(0, 0, this.width, this.height);

    // Lightning Flash Effect for Thunderstorms
    if (this.currentMode === 'Thunderstorm') {
      const now = Date.now();
      if (now - this.lastLightning > 3500 + Math.random() * 6000) {
        this.lightningAlpha = 0.65;
        this.lastLightning = now;
      }
      if (this.lightningAlpha > 0.01) {
        this.ctx.fillStyle = `rgba(230, 240, 255, ${this.lightningAlpha})`;
        this.ctx.fillRect(0, 0, this.width, this.height);
        this.lightningAlpha *= 0.88;
      }
    }

    // Render particles based on mode
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];

      if (this.currentMode === 'Rainy / Drizzle' || this.currentMode === 'Thunderstorm') {
        this.ctx.strokeStyle = this.currentMode === 'Thunderstorm'
          ? `rgba(180, 210, 255, ${p.opacity})`
          : `rgba(90, 180, 255, ${p.opacity})`;
        this.ctx.lineWidth = this.currentMode === 'Thunderstorm' ? 2 : 1.4;
        this.ctx.beginPath();
        this.ctx.moveTo(p.x, p.y);
        this.ctx.lineTo(p.x + p.speedX * 2, p.y + p.length);
        this.ctx.stroke();

        p.x += p.speedX;
        p.y += p.speedY;

        if (p.y > this.height) {
          this.particles[i] = this.createParticle(false);
        }
      } else if (this.currentMode === 'Snowy / Blizzard') {
        p.swing += 0.03;
        p.x += Math.sin(p.swing) * 0.8;
        p.y += p.speedY;

        this.ctx.fillStyle = `rgba(255, 255, 255, ${p.opacity})`;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        this.ctx.fill();

        if (p.y > this.height) {
          this.particles[i] = this.createParticle(false);
        }
      } else if (this.currentMode === 'Foggy / Mist' || this.currentMode === 'Partly Cloudy / Overcast') {
        p.x += p.speedX;
        p.y += p.speedY;

        const grad = this.ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius);
        grad.addColorStop(0, `rgba(220, 230, 245, ${p.opacity})`);
        grad.addColorStop(1, 'rgba(220, 230, 245, 0)');

        this.ctx.fillStyle = grad;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        this.ctx.fill();

        if (p.x - p.radius > this.width) {
          p.x = -p.radius;
        }
      } else {
        // Sunny / Clear Golden Dust Particles
        p.pulse += 0.04;
        p.y += p.speedY;
        p.x += p.speedX;
        const currentRadius = p.radius + Math.sin(p.pulse) * 0.5;

        this.ctx.fillStyle = `rgba(251, 191, 36, ${p.opacity})`;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, Math.max(0.5, currentRadius), 0, Math.PI * 2);
        this.ctx.fill();

        if (p.y < -10) {
          this.particles[i] = this.createParticle(false);
          this.particles[i].y = this.height + 10;
        }
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}

let weatherSystem = null;

/* =================================================================
   2. Slider Input Binding & Event Handling
   ================================================================= */
function initializeSliders() {
  FEATURE_KEYS.forEach(key => {
    const slider = document.getElementById(key);
    const badge = document.getElementById(`val_${key}`);

    if (slider && badge) {
      // Sync initial badge value
      badge.textContent = parseFloat(slider.value).toFixed(slider.step.includes('.') ? 1 : 0);

      slider.addEventListener('input', () => {
        badge.textContent = parseFloat(slider.value).toFixed(slider.step.includes('.') ? 1 : 0);
        
        // Remove preset chip active highlight if manually adjusted
        presetChips.forEach(chip => chip.classList.remove('active'));

        if (realtimeToggle.checked) {
          triggerDebouncedPredict();
        }
      });
    }
  });

  // Reset defaults button
  document.getElementById('resetBtn').addEventListener('click', () => {
    setPresetValues({
      temperature: 26.0,
      humidity: 45.0,
      wind_speed: 12.0,
      precipitation: 0.0,
      cloud_cover: 15.0,
      atmospheric_pressure: 1018.0,
      visibility: 16.0,
      uv_index: 8.0
    });
    presetChips.forEach(chip => chip.classList.remove('active'));
    executePredict();
  });

  // Form submission
  weatherForm.addEventListener('submit', (e) => {
    e.preventDefault();
    executePredict();
  });
}

function getSliderValues() {
  const payload = {};
  FEATURE_KEYS.forEach(key => {
    const slider = document.getElementById(key);
    payload[key] = parseFloat(slider.value);
  });
  return payload;
}

function setPresetValues(values) {
  FEATURE_KEYS.forEach(key => {
    if (key in values) {
      const slider = document.getElementById(key);
      const badge = document.getElementById(`val_${key}`);
      if (slider && badge) {
        slider.value = values[key];
        badge.textContent = parseFloat(values[key]).toFixed(slider.step.includes('.') ? 1 : 0);
      }
    }
  });
}

function triggerDebouncedPredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    executePredict();
  }, 120);
}

/* =================================================================
   3. Predict API Call & UI Rendering
   ================================================================= */
async function executePredict() {
  const payload = getSliderValues();
  const predictBtn = document.getElementById('predictBtn');
  
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    if (result.status === 'success') {
      renderPredictionResult(result);
    } else {
      console.error('Prediction API error:', result.message);
    }
  } catch (err) {
    console.error('Failed to contact prediction service:', err);
  }
}

function renderPredictionResult(data) {
  const predClass = data.prediction;
  const meta = data.metadata || {};
  const theme = meta.theme || 'sunny';

  currentWeatherType = predClass;
  currentTheme = theme;

  // Update canvas background weather mode
  if (weatherSystem) {
    weatherSystem.setMode(predClass);
  }

  // Update body theme class
  bodyEl.className = `theme-${theme}`;

  // Update Main Hero Prediction
  document.getElementById('weatherIcon').textContent = meta.icon || '🌡️';
  document.getElementById('predictedWeatherName').textContent = predClass;
  document.getElementById('confidenceVal').textContent = `${data.confidence.toFixed(1)}%`;
  document.getElementById('weatherHeadline').textContent = meta.headline || '';
  document.getElementById('weatherDescription').textContent = meta.description || '';

  // Update Severity Badge
  const severityBadge = document.getElementById('severityBadge');
  severityBadge.textContent = meta.severity || 'Normal Conditions';
  severityBadge.setAttribute('data-level', meta.severity_level || 'low');

  // Update Lifestyle Advisories
  document.getElementById('clothingAdvice').textContent = meta.clothing || '';
  document.getElementById('activityAdvice').textContent = meta.outdoor_activity || '';
  document.getElementById('precautionsAdvice').textContent = meta.precautions || '';

  // Update Probability Bars
  renderProbabilityBars('probListContainer', data.probabilities, predClass);
}

function renderProbabilityBars(containerId, probabilities, topClass) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = '';
  Object.entries(probabilities).forEach(([cls, prob]) => {
    const isActive = cls === topClass;
    const row = document.createElement('div');
    row.className = `prob-row ${isActive ? 'active' : ''}`;
    row.innerHTML = `
      <div class="prob-row-meta">
        <span>${cls}</span>
        <span>${prob.toFixed(1)}%</span>
      </div>
      <div class="prob-meter-bg">
        <div class="prob-meter-fill" style="width: ${Math.max(1, prob)}%"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

/* =================================================================
   4. Presets Handling
   ================================================================= */
let presetsData = {};

async function initializePresets() {
  try {
    const resp = await fetch('/api/presets');
    const json = await resp.json();
    if (json.status === 'success') {
      presetsData = json.presets;
    }
  } catch (e) {
    console.error('Failed to load presets:', e);
  }

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      const presetKey = chip.getAttribute('data-preset');
      if (presetsData[presetKey]) {
        setPresetValues(presetsData[presetKey].values);
        executePredict();
      }
    });
  });
}

/* =================================================================
   5. Live City Weather Telemetry & ML Detection
   ================================================================= */
function initializeLiveWeather() {
  const liveForm = document.getElementById('liveSearchForm');
  const cityInput = document.getElementById('cityInput');
  const geoBtn = document.getElementById('geoBtn');
  const quickCityChips = document.querySelectorAll('.city-chip');

  liveForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const city = cityInput.value.trim();
    if (city) {
      fetchLiveCityWeather(`city=${encodeURIComponent(city)}`);
    }
  });

  quickCityChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const city = chip.getAttribute('data-city');
      cityInput.value = city;
      fetchLiveCityWeather(`city=${encodeURIComponent(city)}`);
    });
  });

  geoBtn.addEventListener('click', () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.');
      return;
    }
    const liveLoading = document.getElementById('liveLoading');
    liveLoading.classList.remove('hidden');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        fetchLiveCityWeather(`lat=${lat}&lon=${lon}`);
      },
      (err) => {
        liveLoading.classList.add('hidden');
        alert(`Geolocation error: ${err.message}. Defaulting to city search.`);
      }
    );
  });
}

async function fetchLiveCityWeather(queryParams) {
  const loadingEl = document.getElementById('liveLoading');
  const errorEl = document.getElementById('liveError');
  const resultEl = document.getElementById('liveResultContainer');
  const errorMsg = document.getElementById('liveErrorMessage');

  loadingEl.classList.remove('hidden');
  errorEl.classList.add('hidden');
  resultEl.classList.add('hidden');

  try {
    const res = await fetch(`/api/live_weather?${queryParams}`);
    const data = await res.json();
    loadingEl.classList.add('hidden');

    if (data.status === 'success') {
      renderLiveCityResult(data);
      resultEl.classList.remove('hidden');
    } else {
      errorMsg.textContent = data.message || 'Unable to fetch weather for this location.';
      errorEl.classList.remove('hidden');
    }
  } catch (err) {
    loadingEl.classList.add('hidden');
    errorMsg.textContent = 'Network error contacting live weather service.';
    errorEl.classList.remove('hidden');
  }
}

function renderLiveCityResult(data) {
  const loc = data.location;
  const tel = data.telemetry;
  const ml = data.ml_prediction;
  const meta = ml.metadata || {};

  document.getElementById('liveCityName').innerHTML = `<i class="fa-solid fa-map-pin"></i> ${loc.city}`;
  document.getElementById('liveCoords').innerHTML = `Lat: ${loc.latitude.toFixed(4)} &bull; Lon: ${loc.longitude.toFixed(4)} &bull; Timezone: ${loc.timezone}`;

  // Telemetry grid cards
  const teleGrid = document.getElementById('liveTelemetryGrid');
  teleGrid.innerHTML = `
    <div class="tele-card">
      <span class="tele-label">Temperature</span>
      <span class="tele-value">${tel.temperature.toFixed(1)} °C</span>
    </div>
    <div class="tele-card">
      <span class="tele-label">Humidity</span>
      <span class="tele-value">${tel.humidity.toFixed(0)} %</span>
    </div>
    <div class="tele-card">
      <span class="tele-label">Wind Speed</span>
      <span class="tele-value">${tel.wind_speed.toFixed(1)} km/h</span>
    </div>
    <div class="tele-card">
      <span class="tele-label">Precipitation</span>
      <span class="tele-value">${tel.precipitation.toFixed(1)} mm</span>
    </div>
    <div class="tele-card">
      <span class="tele-label">Cloud Cover</span>
      <span class="tele-value">${tel.cloud_cover.toFixed(0)} %</span>
    </div>
    <div class="tele-card">
      <span class="tele-label">Pressure</span>
      <span class="tele-value">${tel.atmospheric_pressure.toFixed(1)} hPa</span>
    </div>
    <div class="tele-card">
      <span class="tele-label">Visibility</span>
      <span class="tele-value">${tel.visibility.toFixed(1)} km</span>
    </div>
    <div class="tele-card">
      <span class="tele-label">UV Index</span>
      <span class="tele-value">${tel.uv_index.toFixed(1)}</span>
    </div>
  `;

  // ML Detection Output
  document.getElementById('liveConfidence').textContent = `${ml.confidence.toFixed(1)}% ML Confidence`;
  document.getElementById('liveWeatherIcon').textContent = meta.icon || '⛅';
  document.getElementById('liveWeatherName').textContent = ml.detected_weather;
  document.getElementById('liveHeadline').textContent = meta.headline || '';
  document.getElementById('liveDescription').textContent = meta.description || '';

  renderProbabilityBars('liveProbList', ml.probabilities, ml.detected_weather);
}

/* =================================================================
   6. Model Diagnostics & Confusion Matrix
   ================================================================= */
async function loadDiagnostics() {
  if (modelMetricsCache) {
    renderDiagnostics(modelMetricsCache);
    return;
  }

  try {
    const res = await fetch('/api/metrics');
    const json = await res.json();
    if (json.status === 'success') {
      modelMetricsCache = json.data;
      renderDiagnostics(modelMetricsCache);
    }
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

function renderDiagnostics(data) {
  document.getElementById('header-accuracy').textContent = `${data.accuracy}% Accuracy`;
  document.getElementById('diagAccuracy').textContent = `${data.accuracy}%`;
  document.getElementById('diagSamples').textContent = data.total_samples.toLocaleString();

  // Feature Importance List
  const featList = document.getElementById('featureImportanceList');
  featList.innerHTML = '';
  const entries = Object.entries(data.feature_importances);
  const maxImp = Math.max(...entries.map(e => e[1]));

  entries.forEach(([feat, imp]) => {
    const pct = (imp * 100).toFixed(2);
    const widthPct = ((imp / maxImp) * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'feat-row';
    row.innerHTML = `
      <div class="feat-meta">
        <span style="text-transform: capitalize;">${feat.replace('_', ' ')}</span>
        <span>${pct}%</span>
      </div>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width: ${widthPct}%"></div>
      </div>
    `;
    featList.appendChild(row);
  });

  // Confusion Matrix Table
  const table = document.getElementById('confusionTable');
  const classes = data.classes;
  const cm = data.confusion_matrix;

  let html = '<thead><tr><th>True \\ Pred</th>';
  classes.forEach(c => {
    const short = c.split('/')[0].trim();
    html += `<th>${short}</th>`;
  });
  html += '</tr></thead><tbody>';

  cm.forEach((row, rIdx) => {
    const shortTrue = classes[rIdx].split('/')[0].trim();
    html += `<tr><th>${shortTrue}</th>`;
    row.forEach((val, cIdx) => {
      const isDiag = rIdx === cIdx;
      const isZero = val === 0;
      html += `<td class="cm-cell ${isDiag ? 'diagonal' : ''} ${isZero ? 'zero' : ''}">${val}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody>';
  table.innerHTML = html;
}

/* =================================================================
   7. API Tester
   ================================================================= */
function initializeApiTester() {
  const sendBtn = document.getElementById('apiSendBtn');
  const payloadEditor = document.getElementById('apiPayload');
  const outputViewer = document.getElementById('apiOutput');

  sendBtn.addEventListener('click', async () => {
    try {
      const parsed = JSON.parse(payloadEditor.value);
      outputViewer.textContent = 'Calling /api/predict ...';

      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed)
      });
      const data = await resp.json();
      outputViewer.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      outputViewer.textContent = `Error: ${e.message}`;
    }
  });
}

/* =================================================================
   8. Tab Navigation
   ================================================================= */
function initializeTabs() {
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');

      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const pane = document.getElementById(`tab-${targetTab}`);
      if (pane) {
        pane.classList.add('active');
      }

      if (targetTab === 'analytics') {
        loadDiagnostics();
      } else if (targetTab === 'live') {
        // Trigger default search for London if not already loaded
        const container = document.getElementById('liveResultContainer');
        if (container.classList.contains('hidden')) {
          fetchLiveCityWeather('city=London');
        }
      }
    });
  });
}

/* =================================================================
   9. Multilingual Voice AI Agent Controller
   ================================================================= */
const VOICE_PROMPTS_BY_LANG = {
  'en': [
    "What is the weather in Tokyo?",
    "What should I wear today?",
    "Will it rain today?",
    "Explain current weather condition"
  ],
  'hi': [
    "टोक्यो का मौसम कैसा है?",
    "आज मुझे क्या पहनना चाहिए?",
    "क्या आज बारिश होगी?",
    "वर्तमान मौसम समझाओ"
  ],
  'es': [
    "¿Cómo está el clima en Tokio?",
    "¿Qué ropa debería usar hoy?",
    "¿Va a llover hoy?",
    "Explica el estado actual del clima"
  ],
  'fr': [
    "Quel temps fait-il à Tokyo ?",
    "Quels vêtements devrais-je porter ?",
    "Va-t-il pleuvoir aujourd'hui ?",
    "Explique les conditions météorologiques"
  ],
  'de': [
    "Wie ist das Wetter in Tokio?",
    "Was soll ich heute anziehen?",
    "Wird es heute regnen?",
    "Erkläre die aktuelle Wetterlage"
  ],
  'zh': [
    "东京现在的天气怎么样？",
    "我今天应该穿什么衣服？",
    "今天会下雨吗？",
    "请解释当前气象检测结果"
  ],
  'ja': [
    "東京の天気はどうですか？",
    "今日は何を着たらいいですか？",
    "今日雨は降りますか？",
    "現在の気象状態を解説して"
  ],
  'ar': [
    "كيف هو الطقس في طوكيو؟",
    "ماذا يجب أن أرتدي اليوم؟",
    "هل ستمطر اليوم؟",
    "اشرح لي حالة الطقس الحالية"
  ],
  'ru': [
    "Какая сейчас погода в Токио?",
    "Что мне сегодня надеть?",
    "Будет ли сегодня дождь?",
    "Объясни текущее состояние погоды"
  ],
  'pt': [
    "Como está o tempo em Tóquio?",
    "Que roupa devo vestir hoje?",
    "Vai chover hoje?",
    "Explique as condições do clima atual"
  ],
  'it': [
    "Che tempo fa a Tokyo?",
    "Cosa dovrei indossare oggi?",
    "Pioverà oggi?",
    "Spiega le attuali condizioni meteorologiche"
  ],
  'ko': [
    "도쿄의 현재 날씨는 어때요?",
    "오늘 어떤 옷을 입어야 하나요?",
    "오늘 비가 올까요?",
    "현재 기상 상태를 설명해주세요"
  ]
};

const BCP47_LANGUAGE_TAGS = {
  'en': 'en-US',
  'hi': 'hi-IN',
  'es': 'es-ES',
  'fr': 'fr-FR',
  'de': 'de-DE',
  'zh': 'zh-CN',
  'ja': 'ja-JP',
  'ar': 'ar-SA',
  'ru': 'ru-RU',
  'pt': 'pt-BR',
  'it': 'it-IT',
  'ko': 'ko-KR'
};

class VoiceAgentController {
  constructor() {
    this.modal = document.getElementById('voiceModal');
    this.openBtnNav = document.getElementById('navVoiceBtn');
    this.openBtnFab = document.getElementById('floatingVoiceBtn');
    this.closeBtn = document.getElementById('closeVoiceModal');
    
    this.langSelect = document.getElementById('voiceLanguageSelect');
    this.promptsContainer = document.getElementById('voicePromptChips');
    this.mainMicBtn = document.getElementById('mainMicBtn');
    this.mainMicIcon = document.getElementById('mainMicIcon');
    this.stopAudioBtn = document.getElementById('stopAudioBtn');
    this.visualizer = document.getElementById('visualizerContainer');
    this.statusText = document.getElementById('voiceStatusText');

    this.userBubble = document.getElementById('userBubble');
    this.userSpeechText = document.getElementById('userSpeechText');
    this.aiSpeechText = document.getElementById('aiSpeechText');

    this.textForm = document.getElementById('voiceTextInputForm');
    this.textInput = document.getElementById('voiceTextInput');

    this.currentLang = 'en';
    this.isRecording = false;
    this.isSpeaking = false;
    this.recognition = null;

    this.initSpeechRecognition();
    this.bindEvents();
    this.updatePromptChips(this.currentLang);
  }

  initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = false;
      this.recognition.lang = BCP47_LANGUAGE_TAGS[this.currentLang] || 'en-US';

      this.recognition.onstart = () => {
        this.isRecording = true;
        this.mainMicBtn.classList.add('recording');
        this.visualizer.classList.add('active');
        this.statusText.textContent = this.getListeningMessage(this.currentLang);
        this.statusText.className = 'voice-status listening';
      };

      this.recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        this.handleUserSpeechInput(transcript);
      };

      this.recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        this.stopRecordingState();
        this.statusText.textContent = `Mic status: ${event.error}. You can also type below!`;
        this.statusText.className = 'voice-status';
      };

      this.recognition.onend = () => {
        this.stopRecordingState();
      };
    } else {
      console.warn('Web Speech API (SpeechRecognition) is not supported in this browser environment.');
    }
  }

  bindEvents() {
    // Open & Close Modal
    const openModal = () => {
      this.modal.classList.remove('hidden');
      if ('speechSynthesis' in window) {
        window.speechSynthesis.getVoices();
      }
    };

    const closeModal = () => {
      this.modal.classList.add('hidden');
      this.stopSpeaking();
      this.stopRecording();
    };

    if (this.openBtnNav) this.openBtnNav.addEventListener('click', openModal);
    if (this.openBtnFab) this.openBtnFab.addEventListener('click', openModal);
    if (this.closeBtn) this.closeBtn.addEventListener('click', closeModal);

    // Close on backdrop click
    this.modal.addEventListener('click', (e) => {
      if (e.target === this.modal) closeModal();
    });

    // Language change
    this.langSelect.addEventListener('change', (e) => {
      this.currentLang = e.target.value;
      if (this.recognition) {
        this.recognition.lang = BCP47_LANGUAGE_TAGS[this.currentLang] || 'en-US';
      }
      this.updatePromptChips(this.currentLang);
      this.statusText.textContent = this.getReadyMessage(this.currentLang);
      this.statusText.className = 'voice-status';
    });

    // Main Mic Toggle
    this.mainMicBtn.addEventListener('click', () => {
      if (this.isRecording) {
        this.stopRecording();
      } else {
        this.startRecording();
      }
    });

    // Stop Audio / Mute
    this.stopAudioBtn.addEventListener('click', () => {
      this.stopSpeaking();
    });

    // Text Input Form
    this.textForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = this.textInput.value.trim();
      if (text) {
        this.textInput.value = '';
        this.handleUserSpeechInput(text);
      }
    });
  }

  startRecording() {
    this.stopSpeaking();
    if (this.recognition) {
      try {
        this.recognition.lang = BCP47_LANGUAGE_TAGS[this.currentLang] || 'en-US';
        this.recognition.start();
      } catch (e) {
        console.warn('Recognition start exception:', e);
      }
    } else {
      alert('Speech recognition is not supported in this browser. Please type your query in the text box below.');
    }
  }

  stopRecording() {
    if (this.recognition && this.isRecording) {
      try {
        this.recognition.stop();
      } catch (e) {}
    }
    this.stopRecordingState();
  }

  stopRecordingState() {
    this.isRecording = false;
    this.mainMicBtn.classList.remove('recording');
    if (!this.isSpeaking) {
      this.visualizer.classList.remove('active');
      this.statusText.textContent = this.getReadyMessage(this.currentLang);
      this.statusText.className = 'voice-status';
    }
  }

  updatePromptChips(lang) {
    const prompts = VOICE_PROMPTS_BY_LANG[lang] || VOICE_PROMPTS_BY_LANG['en'];
    this.promptsContainer.innerHTML = '';

    prompts.forEach(p => {
      const chip = document.createElement('button');
      chip.className = 'v-prompt-chip';
      chip.textContent = p;
      chip.addEventListener('click', () => {
        this.handleUserSpeechInput(p);
      });
      this.promptsContainer.appendChild(chip);
    });
  }

  getListeningMessage(lang) {
    const msgs = {
      'en': 'Listening... Speak your weather query now.',
      'hi': 'सुन रहा हूँ... अब अपना सवाल बोलिए।',
      'es': 'Escuchando... Di tu pregunta sobre el clima.',
      'fr': 'À l\'écoute... Posez votre question météo.',
      'de': 'Ich höre zu... Stellen Sie Ihre Wetterfrage.',
      'zh': '正在聆听... 请说出您的天气问题。',
      'ja': '聞き取り中... 天気について話してください。',
      'ar': 'جاري الاستماع... تحدث الآن.',
      'ru': 'Слушаю... Задайте вопрос о погоде.',
      'pt': 'Ouvindo... Faça sua pergunta sobre o clima.',
      'it': 'In ascolto... Fai la tua domanda sul meteo.',
      'ko': '듣고 있습니다... 날씨에 대해 말씀해주세요.'
    };
    return msgs[lang] || msgs['en'];
  }

  getReadyMessage(lang) {
    const msgs = {
      'en': 'Click the microphone or choose a prompt to start talking...',
      'hi': 'बात करने के लिए माइक पर क्लिक करें या कोई विकल्प चुनें...',
      'es': 'Haz clic en el micrófono o elige una sugerencia para comenzar...',
      'fr': 'Cliquez sur le micro ou choisissez une invite pour commencer...',
      'de': 'Klicken Sie auf das Mikrofon oder wählen Sie einen Vorschlag...',
      'zh': '点击麦克风或选择上方快捷提示开始交流...',
      'ja': 'マイクをクリックするか質問を選択してください...',
      'ar': 'انقر فوق الميكروفون أو اختر سؤالاً للبدء...',
      'ru': 'Нажмите на микрофон или выберите подсказку...',
      'pt': 'Clique no microfone ou escolha uma sugestão para falar...',
      'it': 'Clicca sul microfono o scegli un suggerimento per iniziare...',
      'ko': '마이크를 클릭하거나 추천 질문을 선택하세요...'
    };
    return msgs[lang] || msgs['en'];
  }

  async handleUserSpeechInput(queryText) {
    // Show user message bubble
    this.userBubble.classList.remove('hidden');
    this.userSpeechText.textContent = queryText;

    this.statusText.textContent = 'Thinking & analyzing weather model...';
    this.statusText.className = 'voice-status';

    // Gather screen state context
    const currentSliders = getSliderValues();
    const currentPrediction = currentWeatherType || 'Sunny / Clear';

    try {
      const resp = await fetch('/api/voice_assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          lang: this.currentLang,
          current_telemetry: currentSliders,
          current_prediction: currentPrediction
        })
      });

      const data = await resp.json();
      if (data.status === 'success') {
        const replyText = data.speech_text;
        this.aiSpeechText.textContent = replyText;

        // Auto trigger action if city was mentioned
        if (data.action === 'load_city' && data.target_city) {
          this.triggerCitySearch(data.target_city);
        }

        // Speak aloud
        this.speakText(replyText, this.currentLang);
      } else {
        this.aiSpeechText.textContent = data.message || 'Could not process voice query.';
      }
    } catch (err) {
      console.error('Voice Assistant API error:', err);
      this.aiSpeechText.textContent = 'Network error contacting Voice AI Agent.';
    }
  }

  triggerCitySearch(cityName) {
    // Switch to live city tab and search
    const liveTabBtn = document.querySelector('.tab-btn[data-tab="live"]');
    if (liveTabBtn) {
      liveTabBtn.click();
      const cityInput = document.getElementById('cityInput');
      if (cityInput) {
        cityInput.value = cityName;
        fetchLiveCityWeather(`city=${encodeURIComponent(cityName)}`);
      }
    }
  }

  speakText(text, lang) {
    if (!('speechSynthesis' in window)) return;

    this.stopSpeaking();

    const utterance = new SpeechSynthesisUtterance(text);
    const bcpTag = BCP47_LANGUAGE_TAGS[lang] || 'en-US';
    utterance.lang = bcpTag;

    // Try finding matching voice
    const voices = window.speechSynthesis.getVoices();
    const matchedVoice = voices.find(v => v.lang.startsWith(lang) || v.lang.replace('_', '-').startsWith(bcpTag));
    if (matchedVoice) {
      utterance.voice = matchedVoice;
    }

    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      this.isSpeaking = true;
      this.visualizer.classList.add('active');
      this.stopAudioBtn.classList.remove('hidden');
      this.statusText.textContent = 'Speaking...';
      this.statusText.className = 'voice-status speaking';
    };

    utterance.onend = () => {
      this.stopSpeaking();
    };

    utterance.onerror = () => {
      this.stopSpeaking();
    };

    window.speechSynthesis.speak(utterance);
  }

  stopSpeaking() {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    this.isSpeaking = false;
    this.stopAudioBtn.classList.add('hidden');
    if (!this.isRecording) {
      this.visualizer.classList.remove('active');
      this.statusText.textContent = this.getReadyMessage(this.currentLang);
      this.statusText.className = 'voice-status';
    }
  }
}

let voiceAgent = null;

/* =================================================================
   Application Bootstrap
   ================================================================= */
document.addEventListener('DOMContentLoaded', () => {
  weatherSystem = new WeatherParticleSystem(weatherCanvas);
  initializeSliders();
  initializePresets();
  initializeLiveWeather();
  initializeApiTester();
  initializeTabs();

  // Initialize Voice AI Agent
  voiceAgent = new VoiceAgentController();

  // Initial prediction
  executePredict();
  // Pre-load diagnostics
  loadDiagnostics();
});
