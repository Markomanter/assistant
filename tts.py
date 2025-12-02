# tts.py
import os
import subprocess
import tempfile

import sounddevice as sd
import soundfile as sf
import pyttsx3

import config

# Папка, де лежить tts.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Шлях до piper.exe (аналогічно до piper_test.py, але відносно tts.py)
PIPER_EXE = os.path.join(BASE_DIR, "piper", "piper.exe")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Моделі Piper (шляхи як у попередній версії tts.py)
PIPER_UK_MODEL = os.path.join(MODELS_DIR, "uk-UA-lada-medium.onnx")
#PIPER_EN_MODEL = os.path.join(MODELS_DIR, "en_US-amy-medium.onnx")
#PIPER_EN_MODEL = os.path.join(BASE_DIR, "en_US-lessac-medium.onnx")
PIPER_EN_MODEL = os.path.join(MODELS_DIR, "en_US-ryan-high.onnx")


def _get_piper_model(lang: str) -> str:
    """
    Повертає шлях до моделі Piper залежно від мови.
    uk* -> українська модель, інакше -> англійська.
    """
    lang = (lang or "en").lower()
    if lang.startswith("uk"):
        model_path = PIPER_UK_MODEL
    else:
        model_path = PIPER_EN_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Не знайдено файл моделі Piper: {model_path}")

    return model_path


def _speak_with_piper(text: str, lang: str = "uk") -> bool:
    """
    Озвучка через Piper CLI (piper.exe), як у piper_test.py.
    Повертає True, якщо все пройшло успішно.
    """
    text = (text or "").strip()
    if not text:
        return True  # нема що озвучувати, але й помилки немає

    if not os.path.exists(PIPER_EXE):
        print(f"❌ Piper не знайдено за шляхом: {PIPER_EXE}")
        return False

    try:
        model_path = _get_piper_model(lang)
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        return False

    print(f"[TTS] Використовую Piper CLI, lang={lang}, model={os.path.basename(model_path)}")

    # Тимчасовий WAV-файл для виводу Piper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        PIPER_EXE,
        "--model", model_path,
        "--output_file", out_path,
    ]

    try:
        # Запускаємо Piper і передаємо текст через stdin (як у piper_test.py)
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # ховаємо зайвий вивід
            stderr=subprocess.PIPE,
        )

        _, stderr = process.communicate(input=text.encode("utf-8"))

        if process.returncode != 0:
            print(f"❌ Помилка Piper: {stderr.decode('utf-8', errors='ignore')}")
            return False

        # Читаємо готовий WAV і програємо
        data, samplerate = sf.read(out_path, dtype="int16")
        print(f"[TTS] Piper WAV shape={data.shape}, samplerate={samplerate}")

        sd.play(data, samplerate)
        sd.wait()

        print("[TTS] Piper: відтворення завершено.")
        return True

    except FileNotFoundError:
        print(f"❌ Piper не знайдено (FileNotFoundError): {PIPER_EXE}")
        return False
    except Exception as e:
        print(f"⚠️ Помилка Piper TTS: {e}")
        return False
    finally:
        # Прибираємо тимчасовий файл
        try:
            os.remove(out_path)
        except Exception:
            pass


def _speak_with_pyttsx3(text: str) -> None:
    """Резервний варіант — старий добрий pyttsx3/SAPI."""
    try:
        print("[TTS] Використовую pyttsx3...")
        engine = pyttsx3.init()
        engine.setProperty("rate", config.TTS_RATE)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        print("[TTS] pyttsx3: відтворення завершено.")
    except Exception as e:
        print("⚠️ Помилка TTS (pyttsx3):", e)


def speak(text: str, lang: str = "uk") -> None:
    """
    Загальна функція TTS:
    - спочатку пробуємо Piper через piper.exe (CLI),
    - якщо не вийшло — fallback на pyttsx3.
    """
    if not getattr(config, "TTS_ENABLED", True):
        return

    text = (text or "").strip()
    if not text:
        return

    # Спочатку пробуємо Piper (CLI-рішення з piper_test.py)
    if _speak_with_piper(text, lang=lang):
        return

    # Якщо Piper не спрацював — fallback на pyttsx3
    _speak_with_pyttsx3(text)
