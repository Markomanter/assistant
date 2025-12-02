# translate.py
from typing import Literal

from transformers import MarianMTModel, MarianTokenizer

# Моделі Helsinki-NLP для uk<->en
UK_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-uk-en"
EN_UK_MODEL_NAME = "Helsinki-NLP/opus-mt-en-uk"


class MarianTranslator:
    def __init__(self):
        # Завантажуємо обидві моделі один раз при імпорті модуля
        self.uk_en_tokenizer = MarianTokenizer.from_pretrained(UK_EN_MODEL_NAME)
        self.uk_en_model = MarianMTModel.from_pretrained(UK_EN_MODEL_NAME)

        self.en_uk_tokenizer = MarianTokenizer.from_pretrained(EN_UK_MODEL_NAME)
        self.en_uk_model = MarianMTModel.from_pretrained(EN_UK_MODEL_NAME)

    def _translate_batch(self, texts, direction: Literal["uk_en", "en_uk"]) -> list[str]:
        if direction == "uk_en":
            tok = self.uk_en_tokenizer
            model = self.uk_en_model
        else:
            tok = self.en_uk_tokenizer
            model = self.en_uk_model

        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=512, num_beams=2)
        decoded = tok.batch_decode(outputs, skip_special_tokens=True)
        # Прибираємо зайві пробіли
        return [d.strip() for d in decoded]

    def translate_uk_en(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        return self._translate_batch([text], "uk_en")[0]

    def translate_en_uk(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        return self._translate_batch([text], "en_uk")[0]


# Глобальний інстанс (завантажується один раз)
translator = MarianTranslator()


def translate(text: str, src: str, dst: str) -> str:
    """
    Універсальний вхід:
    - src='uk', dst='en' -> uk->en
    - src='en', dst='uk' -> en->uk
    Інші варіанти наразі повертають text як є.
    """
    s = (src or "").lower()
    d = (dst or "").lower()

    if s.startswith("uk") and d.startswith("en"):
        return translator.translate_uk_en(text)
    if s.startswith("en") and d.startswith("uk"):
        return translator.translate_en_uk(text)
    
    print(f"[MT] Unsupported direction {src}->{dst}, повертаю оригінал.")

    # Поки що тільки uk<->en підтримуємо
    return text
