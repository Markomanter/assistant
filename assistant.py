# assistant.py

import time
import keyboard
import atexit
import subprocess

import config
from stt import record_audio, transcribe_audio
from tts import speak
from llm import ask_ollama_smart as ask_ollama

import warnings
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)




def normalize_lang(text: str, detected_lang: str | None) -> str:
    """
    Нормалізуємо мову:
    - якщо є кирилиця → 'uk'
    - інакше використовуємо detected_lang або 'en'
    """
    text = text or ""
    detected = (detected_lang or "").lower()

    uk_chars = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    has_cyrillic = any(ch in uk_chars for ch in text)

    if has_cyrillic:
        if detected != "uk":
            print(f"🔤 Whisper визначив мову як '{detected_lang}', але знайдена кирилиця → вважаю 'uk'.")
        return "uk"

    if not detected:
        print("⚠ Whisper не повернув код мови, вважаю 'en'.")
        return "en"

    return detected


def handle_interaction():
    """Один цикл: запис → розпізнавання → LLM → TTS."""
    t0 = time.perf_counter()

    # 1) Запис
    audio = record_audio()
    t1 = time.perf_counter()
    print(f"⏱ Запис зайняв: {t1 - t0:.2f} с")

    # 2) Розпізнавання
    text, lang = transcribe_audio(audio)
    t2 = time.perf_counter()
    print(f"⏱ Розпізнавання зайняло: {t2 - t1:.2f} с")

    if not text:
        print("⚠ Нічого не розпізнано, спробуй ще раз.")
        return

    original_lang = normalize_lang(text, lang)

    # 3) Відповідь моделі (з контекстом, всередині ask_ollama)
    reply = ask_ollama(text, user_lang=original_lang)
    t3 = time.perf_counter()
    print(f"⏱ Відповідь моделі зайняла: {t3 - t2:.2f} с")

    # 4) Вивід
    print("\n=============================")
    print("Ти сказав:")
    print(text)
    print(f"(Мова: {original_lang})")
    print("\nАсистент (фінальна відповідь):")
    print(reply)
    print("=============================\n")

    # 5) Озвучка
    if config.TTS_ENABLED:
        print("🔊 Озвучую відповідь...")
        t4_start = time.perf_counter()
        try:
            speak(reply, lang=original_lang)
        except Exception as e:
            print("⚠️ Помилка на етапі TTS:", e)
        t4_end = time.perf_counter()
        print(f"⏱ Озвучка зайняла: {t4_end - t4_start:.2f} с")
    else:
        t4_end = t3

    total = time.perf_counter() - t0
    print(f"✅ Повний цикл зайняв: {total:.2f} с")
    print(f"Натисни {config.HOTKEY_RECORD.upper()} ще раз, або {config.HOTKEY_EXIT.upper()} для виходу.")


def cleanup_ollama_model():
    """
    Викликається автоматично при завершенні програми.
    Пробує зупинити (вивантажити) поточну модель з пам'яті через `ollama stop`.
    """
    model = getattr(config, "OLLAMA_MODEL", None)
    if not model:
        return

    try:
        print(f"[cleanup] Пробую зупинити модель {model} через 'ollama stop'...")
        subprocess.run(["ollama", "stop", model], check=False)
        print(f"[cleanup] Модель {model} зупинена (якщо була запущена).")
    except Exception as e:
        print(f"[cleanup] Не вдалося зупинити модель {model}: {e}")


# Реєструємо хук очищення при виході
atexit.register(cleanup_ollama_model)


def main():
    print("Голосовий ассистент запущений.")
    print(f"Натисни {config.HOTKEY_RECORD.upper()}, щоб записати голос (≈{config.RECORD_SECONDS} сек).")
    print(f"Натисни {config.HOTKEY_EXIT.upper()}, щоб вийти.\n")

    while True:
        if keyboard.is_pressed(config.HOTKEY_EXIT):
            print("👋 Вихід.")
            break

        if keyboard.is_pressed(config.HOTKEY_RECORD):
            # невелика затримка, щоб уникнути дублювань натискань
            time.sleep(0.2)
            try:
                handle_interaction()
            except Exception as e:
                print("❌ Сталася помилка в циклі взаємодії:", e)

        time.sleep(0.05)  # щоб не грузити CPU


if __name__ == "__main__":
    main()
