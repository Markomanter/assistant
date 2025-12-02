# config.py

# ---------- Аудіо / запис ----------
SAMPLE_RATE = 16000        # Гц
RECORD_SECONDS = 5         # тривалість запису (секунди) — коротко для швидкості

# ---------- Whisper ----------
WHISPER_MODEL_NAME = "large-v3"   # швидка й достатньо точна small, medium, large
WHISPER_DEVICE = "cpu"         # працюємо на CPU
WHISPER_COMPUTE_TYPE = "int8"  # оптимально для CPU

# ---------- Ollama ----------
# базовий URL Ollama (локально)
OLLAMA_BASE_URL = "http://localhost:11434"
# постав тут реальну модель, з якою в тебе вже працює (наприклад "qwen2.5:3b", "qwen:4b", і т.д.)
#OLLAMA_MODEL = "qwen3:4b"
#OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_MODEL = "qwen3:8b"

# Легка router-модель для сортування запитів / вибору інструментів
OLLAMA_ROUTER_MODEL = "qwen3:0.6b"

# ---------- Hotkeys ----------
HOTKEY_RECORD = "f9"
HOTKEY_EXIT = "esc"

# ---------- TTS ----------
TTS_ENABLED = True          # щоб вимкнути озвучку — постав False
TTS_RATE = 190              # швидкість мовлення (збільшити — швидше)

# ---------- VAD (авто-кінець фрази) ----------
MAX_RECORD_SECONDS = 20       # максимум тривалості однієї репліки
VAD_THRESHOLD = 0.01          # чутливість до голосу (чим менше, тим чутливіше)
VAD_SILENCE_SECONDS = 1.2     # скільки секунди тиші вважати кінцем фрази




