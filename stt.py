# stt.py

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import config

print(f"Завантажую модель faster-whisper ({config.WHISPER_MODEL_NAME})…")

whisper_model = WhisperModel(
    config.WHISPER_MODEL_NAME,
    device=config.WHISPER_DEVICE,              # з config
    compute_type=config.WHISPER_COMPUTE_TYPE,  # "int8" для швидкості
)


def _rms(x: np.ndarray) -> float:
    """Середньоквадратичне значення (грубо — гучність фрейму)."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def record_audio() -> np.ndarray:
    """
    Слухаємо мікрофон, поки:
    - не зʼявиться голос (гучність > VAD_THRESHOLD),
    - а потім не буде тиші VAD_SILENCE_SECONDS підряд.

    Так асистент сам розуміє, коли ти закінчив говорити.
    """
    sr = config.SAMPLE_RATE
    FRAME_DURATION = 0.2  # тривалість одного фрейму в секундах
    frame_samples = int(sr * FRAME_DURATION)

    vad_threshold = getattr(config, "VAD_THRESHOLD", 0.01)
    vad_silence_seconds = getattr(config, "VAD_SILENCE_SECONDS", 0.8)
    max_record_seconds = getattr(config, "MAX_RECORD_SECONDS", 20)

    print("🎙 Слухаю мікрофон... Говори, і я зупинюся, коли буде пауза.")

    audio_chunks: list[np.ndarray] = []
    started = False
    silence_time = 0.0
    total_time = 0.0

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
        while True:
            frame, _ = stream.read(frame_samples)  # shape: (frame_samples, 1)
            frame = frame.reshape(-1)
            total_time += FRAME_DURATION

            level = _rms(frame)

            if not started:
                # Чекаємо, поки зʼявиться голос
                if level > vad_threshold:
                    started = True
                    print("🎙 Виявив голос, записую...")
                    audio_chunks.append(frame.copy())
                else:
                    # ще тиша до початку — пропускаємо
                    continue
            else:
                # вже записуємо
                audio_chunks.append(frame.copy())

                if level < vad_threshold:
                    silence_time += FRAME_DURATION
                    if silence_time >= vad_silence_seconds:
                        print("⏹ Виявлено паузу, зупиняю запис.")
                        break
                else:
                    # знову голос — обнуляємо таймер тиші
                    silence_time = 0.0

                if total_time >= max_record_seconds:
                    print("⏹ Досягнуто MAX_RECORD_SECONDS, зупиняю запис.")
                    break

    if not audio_chunks:
        print("⚠ Не отримав жодного звуку.")
        return np.zeros(0, dtype="float32")

    audio = np.concatenate(audio_chunks).astype("float32")
    return audio


def transcribe_audio(audio: np.ndarray):
    """
    Приймає numpy-масив float32 [N] 16kHz і відправляє в faster-whisper.
    Повертає (text, lang_code).
    """
    if audio is None or audio.size == 0:
        return "", "unknown"

    # На всякий випадок зведемо до 1D
    if audio.ndim > 1:
        audio = audio[:, 0]

    if config.SAMPLE_RATE != 16000:
        # У тебе SAMPLE_RATE = 16000, тому цей блок можна ігнорувати.
        # Якщо коли-небудь поміняєш — тут треба буде додати ресемплінг.
        pass

    print("🧠 Розпізнаю текст...")

    segments, info = whisper_model.transcribe(
        audio,
        beam_size=3,
        language=None,  # авто-визначення
    )

    text_chunks = [seg.text for seg in segments]
    text = " ".join(text_chunks).strip()
    lang = (info.language or "unknown").lower()

    print(f"📝 Розпізнаний текст: {text!r}")
    print(f"🌐 Визначена мова: {lang}")
    return text, lang
