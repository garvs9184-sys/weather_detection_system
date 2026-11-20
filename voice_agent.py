"""
Multilingual Voice Agent Engine for AtmosphereAI
Supports natural language speech query processing in 12+ major global languages:
English, Hindi, Spanish, French, German, Chinese, Japanese, Arabic, Russian, Portuguese, Italian, Korean.
"""

import re
import requests
import numpy as np

SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'native': 'English', 'speech_code': 'en-US', 'flag': '🌐'},
    'hi': {'name': 'Hindi', 'native': 'हिन्दी', 'speech_code': 'hi-IN', 'flag': '🇮🇳'},
    'es': {'name': 'Spanish', 'native': 'Español', 'speech_code': 'es-ES', 'flag': '🇪🇸'},
    'fr': {'name': 'French', 'native': 'Français', 'speech_code': 'fr-FR', 'flag': '🇫🇷'},
    'de': {'name': 'German', 'native': 'Deutsch', 'speech_code': 'de-DE', 'flag': '🇩🇪'},
    'zh': {'name': 'Chinese', 'native': '中文', 'speech_code': 'zh-CN', 'flag': '🇨🇳'},
    'ja': {'name': 'Japanese', 'native': '日本語', 'speech_code': 'ja-JP', 'flag': '🇯🇵'},
    'ar': {'name': 'Arabic', 'native': 'العربية', 'speech_code': 'ar-SA', 'flag': '🇸🇦'},
    'ru': {'name': 'Russian', 'native': 'Русский', 'speech_code': 'ru-RU', 'flag': '🇷🇺'},
    'pt': {'name': 'Portuguese', 'native': 'Português', 'speech_code': 'pt-BR', 'flag': '🇧🇷'},
    'it': {'name': 'Italian', 'native': 'Italiano', 'speech_code': 'it-IT', 'flag': '🇮🇹'},
    'ko': {'name': 'Korean', 'native': '한국어', 'speech_code': 'ko-KR', 'flag': '🇰🇷'}
}

# Weather Class Localized Names
WEATHER_NAMES_I18N = {
    'Sunny / Clear': {
        'en': 'Sunny and Clear skies',
        'hi': 'धूप और साफ मौसम (Sunny & Clear)',
        'es': 'Soleado y Despejado',
        'fr': 'Ensoleillé et Ciel Dégagé',
        'de': 'Sonnig und Klarer Himmel',
        'zh': '晴朗无云',
        'ja': '快晴・晴れ',
        'ar': 'مشمس وصافٍ',
        'ru': 'Солнечно и Ясно',
        'pt': 'Ensolarado e Limpo',
        'it': 'Soleggiato e Sereno',
        'ko': '맑고 화창한 날씨'
    },
    'Partly Cloudy / Overcast': {
        'en': 'Partly Cloudy to Overcast',
        'hi': 'आंशिक रूप से बादलों से घिरा मौसम (Overcast)',
        'es': 'Parcialmente Nublado a Cubierto',
        'fr': 'Partiellement Nuageux à Couvert',
        'de': 'Teils Bewölkt bis Bedeckt',
        'zh': '多云至阴天',
        'ja': '曇りがち・薄曇り',
        'ar': 'غائم جزئياً إلى ملبد بالغيوم',
        'ru': 'Переменная облачность',
        'pt': 'Parcialmente Nublado a Encoberto',
        'it': 'Parzialmente Nuvoloso o Coperto',
        'ko': '구름 많고 흐림'
    },
    'Rainy / Drizzle': {
        'en': 'Rain and Drizzle',
        'hi': 'बारिश और फुहारें (Rainy / Drizzle)',
        'es': 'Lluvia y Llovizna continua',
        'fr': 'Pluie et Bruine continue',
        'de': 'Regen und Nieselregen',
        'zh': '降雨及细雨天气',
        'ja': '雨・霧雨',
        'ar': 'ممطر ورذاذ مستمر',
        'ru': 'Дождь и морось',
        'pt': 'Chuva e Garoa contínua',
        'it': 'Pioggia e Pioviggine',
        'ko': '비와 이슬비'
    },
    'Thunderstorm': {
        'en': 'Severe Thunderstorm',
        'hi': 'भीषण गरज-चमक के साथ तूफान (Thunderstorm)',
        'es': 'Tormenta Eléctrica Severa',
        'fr': 'Violent Orage avec Foudre',
        'de': 'Schweres Gewitter mit Sturmböen',
        'zh': '强雷暴及雷雨',
        'ja': '激しい雷雨・嵐',
        'ar': 'عاصفة رعدية شديدة مع رياح قوية',
        'ru': 'Сильная гроза со шквалом',
        'pt': 'Tempestade Elétrica Severa',
        'it': 'Forte Temporale con Fulmini',
        'ko': '강한 뇌우 및 폭풍'
    },
    'Snowy / Blizzard': {
        'en': 'Snowfall and Blizzard conditions',
        'hi': 'बर्फबारी और बर्फीला तूफान (Snowy / Blizzard)',
        'es': 'Nevada y Ventisca gélida',
        'fr': 'Chutes de Neige et Blizzard',
        'de': 'Schneefall und Schneesturm',
        'zh': '降雪及暴风雪天气',
        'ja': '降雪・吹雪',
        'ar': 'تساقط الثلوج وعواصف ثلجية',
        'ru': 'Снегопад и метель',
        'pt': 'Nevasca e Condições Congelantes',
        'it': 'Nevicata e Tormenta di Neve',
        'ko': '눈보라 및 폭설'
    },
    'Foggy / Mist': {
        'en': 'Dense Fog and Reduced Visibility',
        'hi': 'घना कोहरा और धुंध (Foggy / Mist)',
        'es': 'Niebla Densa y Visibilidad Reducida',
        'fr': 'Brouillard Épais et Faible Visibilité',
        'de': 'Dichter Nebel mit Sichteinschränkung',
        'zh': '浓雾与低能见度天气',
        'ja': '濃霧・視界不良',
        'ar': 'ضباب كثيف وانخفاض في مدى الرؤية',
        'ru': 'Густой туман и плохая видимость',
        'pt': 'Nevoeiro Denso e Baixa Visibilidade',
        'it': 'Nebbia Fitta e Visibilità Ridotta',
        'ko': '짙은 안개 및 시야 저하'
    }
}

# Common Global Cities Lexicon (English & Multilingual mappings)
KNOWN_CITIES = {
    'tokyo': 'Tokyo', 'टोक्यो': 'Tokyo', '東京': 'Tokyo', 'とうきょう': 'Tokyo',
    'london': 'London', 'लंदन': 'London', 'londres': 'London', '伦敦': 'London', 'ロンドン': 'London',
    'paris': 'Paris', 'पेरिस': 'Paris', '巴黎': 'Paris', 'パリ': 'Paris',
    'mumbai': 'Mumbai', 'मुंबई': 'Mumbai', 'बम्बई': 'Mumbai', '孟买': 'Mumbai',
    'delhi': 'Delhi', 'दिल्ली': 'Delhi', 'नई दिल्ली': 'Delhi', 'new delhi': 'Delhi', '德里': 'Delhi',
    'new york': 'New York', 'न्यूयॉर्क': 'New York', 'nueva york': 'New York', '纽约': 'New York', 'ニューヨーク': 'New York',
    'dubai': 'Dubai', 'दुबई': 'Dubai', 'دبي': 'Dubai', '迪拜': 'Dubai',
    'reykjavik': 'Reykjavik', 'रेक्याविक': 'Reykjavik',
    'sydney': 'Sydney', 'सिडनी': 'Sydney', '悉尼': 'Sydney', 'シドニー': 'Sydney',
    'berlin': 'Berlin', 'बर्लिन': 'Berlin', '柏林': 'Berlin',
    'madrid': 'Madrid', 'मैड्रिड': 'Madrid', '马德里': 'Madrid',
    'rome': 'Rome', 'रोम': 'Rome', 'roma': 'Rome', '罗马': 'Rome',
    'toronto': 'Toronto', 'टोरंटो': 'Toronto', '多伦多': 'Toronto',
    'moscow': 'Moscow', 'मॉस्को': 'Moscow', 'москва': 'Moscow', '莫斯科': 'Moscow',
    'beijing': 'Beijing', 'बीजिंग': 'Beijing', '北京': 'Beijing',
    'singapore': 'Singapore', 'सिंगापुर': 'Singapore', '新加坡': 'Singapore',
    'bangalore': 'Bangalore', 'बेंगलुरु': 'Bangalore', 'bengaluru': 'Bangalore',
    'cairo': 'Cairo', 'काहिरा': 'Cairo', 'القاهرة': 'Cairo', '开罗': 'Cairo',
    'seoul': 'Seoul', 'सियोल': 'Seoul', '首尔': 'Seoul', 'ソウル': 'Seoul',
    'los angeles': 'Los Angeles', 'chicago': 'Chicago',
    'san francisco': 'San Francisco', 'boston': 'Boston'
}

def extract_city_from_query(query):
    """
    Extracts targeted city name from a speech transcription.
    """
    q_lower = query.lower().strip()
    
    # 1. Direct dictionary matches
    for k, v in KNOWN_CITIES.items():
        if k in q_lower:
            return v
            
    # 2. Regex patterns across languages
    patterns = [
        r'(?:weather in|forecast for|temperature in|clima en|tiempo en|météo à|temps à|wetter in|in)\s+([a-zA-Z\u00C0-\u024F\u0900-\u097F]+)',
        r'([a-zA-Z\u00C0-\u024F\u0900-\u097F]+)\s+(?:का मौसम|में मौसम|weather)',
        r'([^\s]+?)(?:的)?天气',
        r'([^\s]+?)の天気',
        r'طقس\s+([^\s]+)'
    ]
    for p in patterns:
        m = re.search(p, q_lower)
        if m:
            city_cand = m.group(1).strip()
            # Clean common filler words
            city_cand = re.sub(r'^(the|city|de|la|el|le|der|die|das)\s+', '', city_cand)
            if len(city_cand) > 2 and city_cand not in ['today', 'tomorrow', 'now', 'here']:
                return city_cand.title()

    return None

def detect_query_intent(query):
    """
    Detects user intent: 'city_weather', 'clothing', 'rain_check', 'explanation', 'greeting', 'status'.
    """
    q = query.lower()
    
    # Greeting
    if any(w in q for w in ['hello', 'hi', 'hey', 'नमस्ते', 'hola', 'bonjour', 'hallo', '你好', 'こんにちは', 'مرحبا', 'привет', 'olá', 'ciao', '안녕']):
        return 'greeting'
        
    # Clothing / What to wear
    if any(w in q for w in ['wear', 'cloth', 'attire', 'jacket', 'coat', 'कपड़े', 'पहनूं', 'ropa', 'vestir', 'vêtements', 'porter', 'kleidung', '穿什么', '着る', 'ارتداء', 'одежда', 'roupa', 'indossare', '옷']):
        return 'clothing'
        
    # Rain check / Will it rain
    if any(w in q for w in ['rain', 'umbrella', 'बारिश', 'छाता', 'lluvia', 'paraguas', 'pluie', 'parapluie', 'regen', 'regnen', '下雨', '雨', 'مطر', 'дождь', 'chuva', 'pioggia', '비']):
        return 'rain_check'
        
    # Explanation
    if any(w in q for w in ['why', 'explain', 'how does', 'क्यों', 'समझाइए', 'por qué', 'pourquoi', 'warum', '为什么', 'なぜ', 'لماذا', 'почему', 'por que', 'perché', '왜']):
        return 'explanation'
        
    # City weather check
    if extract_city_from_query(query):
        return 'city_weather'
        
    return 'status'

def generate_voice_response(query, lang_code, current_telemetry, current_prediction, live_weather_fn, ml_predict_fn):
    """
    Processes incoming voice transcription and produces localized speech text and UI actions.
    """
    lang = lang_code if lang_code in SUPPORTED_LANGUAGES else 'en'
    intent = detect_query_intent(query)
    city = extract_city_from_query(query)
    
    # Case 1: City Query
    if city:
        live_result = live_weather_fn(city)
        if live_result and live_result.get('status') == 'success':
            loc = live_result['location']['city']
            tel = live_result['telemetry']
            pred = live_result['ml_prediction']['detected_weather']
            conf = live_result['ml_prediction']['confidence']
            temp = round(tel['temperature'], 1)
            humidity = round(tel['humidity'])
            
            pred_localized = WEATHER_NAMES_I18N.get(pred, {}).get(lang, pred)
            
            speech_templates = {
                'en': f"In {loc}, the temperature is currently {temp}°C with {humidity}% humidity. Our machine learning model detects {pred_localized} with {conf:.0f}% confidence.",
                'hi': f"{loc} में इस समय तापमान {temp}°C और नमी {humidity}% है। हमारे मशीन लर्निंग मॉडल ने {conf:.0f}% विश्वास के साथ {pred_localized} की पुष्टि की है।",
                'es': f"En {loc}, la temperatura actual es de {temp}°C con {humidity}% de humedad. Nuestro modelo de machine learning detecta {pred_localized} con {conf:.0f}% de confianza.",
                'fr': f"À {loc}, la température actuelle est de {temp}°C avec {humidity}% d'humidité. Notre modèle d'apprentissage automatique détecte {pred_localized} avec {conf:.0f}% de confiance.",
                'de': f"In {loc} beträgt die Temperatur aktuell {temp}°C bei {humidity}% Luftfeuchtigkeit. Unser Machine-Learning-Modell erkennt {pred_localized} mit {conf:.0f}% Konfidenz.",
                'zh': f"{loc}当前气温为{temp}摄氏度，湿度为{humidity}%。我们的机器学习模型以{conf:.0f}%的置信度检测为：{pred_localized}。",
                'ja': f"{loc}の現在の気温は{temp}度、湿度は{humidity}%です。機械学習モデルは信頼度{conf:.0f}%で「{pred_localized}」と判定しました。",
                'ar': f"في {loc}، تبلغ درجة الحرارة حالياً {temp} درجة مئوية مع رطوبة {humidity}%. يكشف نموذج الذكاء الاصطناعي عن {pred_localized} بنسبة ثقة {conf:.0f}%.",
                'ru': f"В городе {loc} сейчас температура {temp}°C и влажность {humidity}%. Модель машинного обучения с уверенностью {conf:.0f}% определяет {pred_localized}.",
                'pt': f"Em {loc}, a temperatura atual é de {temp}°C com {humidity}% de umidade. Nosso modelo de machine learning detecta {pred_localized} com {conf:.0f}% de confiança.",
                'it': f"A {loc} la temperatura è di {temp}°C con umidità al {humidity}%. Il nostro modello di intelligenza artificiale rileva {pred_localized} con confidenza del {conf:.0f}%.",
                'ko': f"{loc}의 현재 기온은 {temp}도이며 습도는 {humidity}%입니다. 머신러닝 모델이 {conf:.0f}% 신뢰도로 {pred_localized} 상태를 감지했습니다."
            }
            
            reply = speech_templates.get(lang, speech_templates['en'])
            return {
                'speech_text': reply,
                'display_text': reply,
                'lang': lang,
                'action': 'load_city',
                'target_city': city,
                'live_data': live_result
            }

    # Case 2: Greeting
    if intent == 'greeting':
        greetings = {
            'en': "Hello! I am AtmosphereAI Voice Agent. You can ask me about live weather in any city, clothing advice, or atmospheric conditions in your preferred language.",
            'hi': "नमस्ते! मैं एटमॉस्फियर-एआई वॉइस एजेंट हूँ। आप मुझसे किसी भी शहर का लाइव मौसम, कपड़ों की सलाह या मौसम की स्थिति अपनी भाषा में पूछ सकते हैं।",
            'es': "¡Hola! Soy el Asistente de Voz de AtmosphereAI. Puedes preguntarme sobre el clima en cualquier ciudad, qué ropa usar o las condiciones atmosféricas en tu idioma.",
            'fr': "Bonjour ! Je suis l'agent vocal AtmosphereAI. Demandez-moi la météo en direct dans n'importe quelle ville, des conseils vestimentaires ou les conditions atmosphériques.",
            'de': "Hallo! Ich bin der AtmosphereAI Sprachassistent. Sie können mich nach dem Live-Wetter in jeder Stadt, Kleidungsempfehlungen oder Wettervorhersagen fragen.",
            'zh': "您好！我是 AtmosphereAI 智能语音助手。您可以随时用您偏好的语言向我询问任何城市的天气、着装建议或气象检测结果。",
            'ja': "こんにちは！AtmosphereAI 音声エージェントです。世界中の都市の天気、服装のアドバイス、気象予測についていつでも声でお尋ねください。",
            'ar': "مرحباً! أنا المساعد الصوتي لـ AtmosphereAI. يمكنك سؤالي عن طقس أي مدينة في العالم، ونصائح الملابس، والتنبؤات الجوية بلغتك المفضلة.",
            'ru': "Здравствуйте! Я голосовой помощник AtmosphereAI. Вы можете спросить меня о погоде в любом городе, подходящей одежде или прогнозе на вашем языке.",
            'pt': "Olá! Sou o assistente de voz do AtmosphereAI. Você pode me perguntar sobre o clima ao vivo em qualquer cidade, dicas de roupas ou previsões meteorológicas.",
            'it': "Ciao! Sono l'assistente vocale di AtmosphereAI. Puoi chiedermi il meteo in tempo reale di qualsiasi città, consigli sull'abbigliamento o condizioni atmosferiche.",
            'ko': "안녕하세요! AtmosphereAI 음성 비서입니다. 전 세계 도시의 실시간 날씨, 옷차림 추천, 기상 예측에 대해 언제든 음성으로 물어보세요."
        }
        reply = greetings.get(lang, greetings['en'])
        return {'speech_text': reply, 'display_text': reply, 'lang': lang, 'action': 'none'}

    # Case 3: Clothing / Attire Advice
    pred_current = current_prediction or 'Sunny / Clear'
    temp_current = current_telemetry.get('temperature', 24.0) if current_telemetry else 24.0
    
    if intent == 'clothing':
        if 'Rain' in pred_current or 'Thunderstorm' in pred_current:
            advice_dict = {
                'en': "It is wet outside! Make sure to take a waterproof raincoat, sturdy umbrella, and water-resistant boots.",
                'hi': "बाहर बारिश का मौसम है! कृपया वाटरप्रूफ रेनकोट, मजबूत छाता और सुरक्षित जूते पहनें।",
                'es': "¡Está lloviendo afuera! Lleva un impermeable resistente al agua, un paraguas firme y calzado impermeable.",
                'fr': "Il pleut dehors ! Prenez un imperméable imperméable, un parapluie résistant et des chaussures étanches.",
                'de': "Es regnet draußen! Bitte nehmen Sie eine Regenjacke, einen stabilen Regenschirm und wasserdichtes Schuhwerk mit.",
                'zh': "外面有雨！请记得携带防水雨衣、结实的雨伞和防水防滑鞋。",
                'ja': "外は雨が降っています！防水のレインコート、丈夫な傘、滑りにくい靴をご用意ください。",
                'ar': "هناك أمطار في الخارج! احرص على ارتداء معطف واقٍ من المطر واصطحاب مظلة وحذاء مضاد للماء.",
                'ru': "На улице дождь! Обязательно наденьте непромокаемую куртку, возьмите зонт и непромокаемую обувь.",
                'pt': "Está chovendo lá fora! Leve uma capa de chuva impermeável, guarda-chuva reforçado e sapatos resistentes.",
                'it': "Fuori piove! Ricordati di portare un impermeabile, un ombrello robusto e scarpe resistenti all'acqua.",
                'ko': "밖에 비가 내리고 있습니다! 방수 레인코트, 튼튼한 우산, 방수 신발을 착용하세요."
            }
        elif 'Snow' in pred_current or temp_current < 2:
            advice_dict = {
                'en': "Freezing temperatures detected! Dress warmly with thermal layers, a heavy insulated coat, gloves, and a warm beanie.",
                'hi': "बाहर कड़ाके की ठंड है! गर्म थर्मल इनर, भारी जैकेट, दस्ताने और गर्म ऊनी टोपी पहनकर ही बाहर निकलें।",
                'es': "¡Temperaturas bajo cero! Vístete con capas térmicas, un abrigo grueso acolchado, guantes y gorro de lana.",
                'fr': "Températures glaciales ! Portez des sous-vêtements thermiques, un manteau épais, des gants et un bonnet chaud.",
                'de': "Frostige Temperaturen! Ziehen Sie warme Thermokleidung, eine dicke Winterjacke, Handschuhe und eine Mütze an.",
                'zh': "当前温度极低！建议穿着保暖内衣、厚羽绒服、手套并佩戴毛线帽。",
                'ja': "氷点下の冷え込みです！暖かい防寒インナー、厚手のダウンコート、手袋、ニット帽を着用してください。",
                'ar': "درجات الحرارة منخفضة جداً! ارتدِ ملابس حرارية، معطفاً شتوياً سميكاً، قفازات، وقبعة صوفية دافئة.",
                'ru': "Очень холодно! Одевайтесь тепло: термобелье, плотный пуховик, перчатки и теплая шапка.",
                'pt': "Temperaturas muito frias! Vista roupas térmicas em camadas, casaco acolchoado pesado, luvas e gorro quente.",
                'it': "Temperature gelide! Indossa abbigliamento termico a strati, cappotto pesante, guanti e berretto di lana.",
                'ko': "영하의 매우 추운 날씨입니다! 보온 내의와 두꺼운 패딩 코트, 장갑, 털모자를 착용하세요."
            }
        else:
            advice_dict = {
                'en': f"With {temp_current:.0f}°C, comfortable breathable clothes, sunglasses, and sun protection are ideal.",
                'hi': f"वर्तमान तापमान {temp_current:.0f}°C है। हल्के आरामदायक कपड़े, धूप का चश्मा और सनस्क्रीन सबसे उपयुक्त हैं।",
                'es': f"Con una temperatura de {temp_current:.0f}°C, la ropa ligera y transpirable, gafas de sol y protección solar son ideales.",
                'fr': f"Avec {temp_current:.0f}°C, des vêtements légers et respirants, des lunettes de soleil et une crème solaire sont recommandés.",
                'de': f"Bei angenehmen {temp_current:.0f}°C ist leichte, atmungsaktive Kleidung mit Sonnenbrille und Sonnenschutz optimal.",
                'zh': f"当前气温为{temp_current:.0f}摄氏度，建议穿着轻便透气的衣物，佩戴墨镜并做好防晒。",
                'ja': f"現在の気温は{temp_current:.0f}度です。軽くて通気性の良い服装、サングラス、日焼け止めが最適です。",
                'ar': f"مع درجة حرارة {temp_current:.0f} مئوية، يُنصح بارتداء ملابس خفيفة مريحة واستخدام نظارات شمسية وواقي شمس.",
                'ru': f"При температуре {temp_current:.0f}°C отлично подойдёт легкая дышащая одежда, солнечные очки и защита от солнца.",
                'pt': f"Com {temp_current:.0f}°C, roupas leves e respiráveis, óculos de sol e protetor solar são ideais.",
                'it': f"Con una temperatura di {temp_current:.0f}°C, abiti leggeri e traspiranti, occhiali da sole e crema solare sono perfetti.",
                'ko': f"현재 기온은 {temp_current:.0f}도입니다. 편안하고 통기성 좋은 가벼운 옷차림과 선글라스 착용을 권장합니다."
            }
        reply = advice_dict.get(lang, advice_dict['en'])
        return {'speech_text': reply, 'display_text': reply, 'lang': lang, 'action': 'none'}

    # Case 4: Rain Check
    if intent == 'rain_check':
        precip = current_telemetry.get('precipitation', 0.0) if current_telemetry else 0.0
        humidity = current_telemetry.get('humidity', 50.0) if current_telemetry else 50.0
        
        if precip > 0.5 or 'Rain' in pred_current or 'Thunderstorm' in pred_current:
            rain_replies = {
                'en': f"Yes, precipitation is actively detected ({precip:.1f} mm with {humidity:.0f}% humidity). Keep an umbrella with you!",
                'hi': f"हाँ, वर्षा सक्रिय है ({precip:.1f} मिमी बारिश, {humidity:.0f}% आर्द्रता)। छाता साथ रखना बिल्कुल न भूलें!",
                'es': f"Sí, se detecta precipitación activa ({precip:.1f} mm y {humidity:.0f}% de humedad). ¡Lleva un paraguas contigo!",
                'fr': f"Oui, des précipitations actives sont détectées ({precip:.1f} mm avec {humidity:.0f}% d'humidité). Gardez un parapluie !",
                'de': f"Ja, Niederschlag ist aktiv ({precip:.1f} mm bei {humidity:.0f}% Feuchtigkeit). Bitte nehmen Sie einen Regenschirm mit!",
                'zh': f"是的，检测到当前有降水（降水量 {precip:.1f} 毫米，湿度 {humidity:.0f}%）。请随身带伞！",
                'ja': f"はい、雨が降っています（降水量 {precip:.1f} mm、湿度 {humidity:.0f}%）。傘を忘れずにお持ちください！",
                'ar': f"نعم، هناك تساقط للأمطار حالياً ({precip:.1f} ملم مع رطوبة {humidity:.0f}%). اصطحب مظلتك!",
                'ru': f"Да, наблюдаются осадки ({precip:.1f} мм при влажности {humidity:.0f}%). Обязательно возьмите зонт!",
                'pt': f"Sim, há precipitação ativa detectada ({precip:.1f} mm com {humidity:.0f}% de umidade). Leve um guarda-chuva!",
                'it': f"Sì, sono presenti precipitazioni attive ({precip:.1f} mm con {humidity:.0f}% di umidità). Porta l'ombrello con te!",
                'ko': f"네, 강수가 감지되었습니다 (강수량 {precip:.1f} mm, 습도 {humidity:.0f}%). 우산을 챙기세요!"
            }
        else:
            rain_replies = {
                'en': f"No rain detected right now. Precipitation is zero and relative humidity is {humidity:.0f}%.",
                'hi': f"फिलहाल बारिश की संभावना नहीं है। वर्षा 0 मिमी है और नमी {humidity:.0f}% है।",
                'es': f"No hay lluvia en este momento. La precipitación es cero y la humedad es del {humidity:.0f}%.",
                'fr': f"Pas de pluie détectée pour le moment. Précipitations nulles et humidité à {humidity:.0f}%.",
                'de': f"Aktuell kein Regen festgestellt. Niederschlag ist 0 mm bei {humidity:.0f}% Luftfeuchtigkeit.",
                'zh': f"目前没有检测到降雨。降水量为 0 毫米，相对湿度为 {humidity:.0f}%。",
                'ja': f"現在雨は観測されていません。降水量は0ミリ、湿度は{humidity:.0f}%です。",
                'ar': f"لا توجد أمطار حالياً. هطول الأمطار صفر ومعدل الرطوبة {humidity:.0f}%.",
                'ru': f"Дождя сейчас нет. Осадки 0 мм, а влажность составляет {humidity:.0f}%.",
                'pt': f"Nenhuma chuva detectada no momento. Precipitação zero e umidade de {humidity:.0f}%.",
                'it': f"Nessuna pioggia rilevata al momento. Precipitazioni a zero e umidità al {humidity:.0f}%.",
                'ko': f"현재 비가 내리지 않고 있습니다. 강수량은 0mm, 습도는 {humidity:.0f}%입니다."
            }
        reply = rain_replies.get(lang, rain_replies['en'])
        return {'speech_text': reply, 'display_text': reply, 'lang': lang, 'action': 'none'}

    # Case 5: Explanation & General Status
    pred_localized = WEATHER_NAMES_I18N.get(pred_current, {}).get(lang, pred_current)
    status_replies = {
        'en': f"AtmosphereAI machine learning model currently detects {pred_localized} based on the real-time atmospheric measurements. The ensemble model evaluated 120 decision trees with high confidence.",
        'hi': f"एटमॉस्फियर-एआई मॉडल ने वर्तमान वायुमंडलीय आंकड़ों के आधार पर {pred_localized} का पता लगाया है। 120 डिसिजन ट्रीज़ ने उच्च सटीकता के साथ यह परिणाम दिया है।",
        'es': f"El modelo de AtmosphereAI detecta actualmente {pred_localized} basado en las mediciones atmosféricas en tiempo real con alta confianza.",
        'fr': f"Le modèle AtmosphereAI détecte actuellement {pred_localized} sur la base des mesures atmosphériques en temps réel avec une haute confiance.",
        'de': f"Das AtmosphereAI Modell erkennt basierend auf den Echtzeit-Atmosphäredaten aktuell {pred_localized} mit hoher Konfidenz.",
        'zh': f"基于当前实时气象遥测数据，AtmosphereAI 机器学习模型高置信度检测出：{pred_localized}。",
        'ja': f"リアルタイムの気象データに基づき、AtmosphereAI 機械学習モデルは高い信頼度で「{pred_localized}」を判定しています。",
        'ar': f"يكشف نموذج AtmosphereAI حالياً عن {pred_localized} بناءً على القياسات الجوية الحالية بنسبة ثقة عالية.",
        'ru': f"Модель AtmosphereAI на основе текущих метеорологических данных с высокой точностью определяет: {pred_localized}.",
        'pt': f"O modelo AtmosphereAI detecta atualmente {pred_localized} com base nas medições atmosféricas em tempo real com alta precisão.",
        'it': f"Il modello AtmosphereAI rileva attualmente {pred_localized} sulla base delle misurazioni atmosferiche in tempo reale.",
        'ko': f"실시간 대기 관측 데이터를 기반으로 AtmosphereAI 머신러닝 모델이 높은 신뢰도로 {pred_localized} 상태를 판별했습니다."
    }
    reply = status_replies.get(lang, status_replies['en'])
    return {'speech_text': reply, 'display_text': reply, 'lang': lang, 'action': 'none'}
