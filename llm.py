# llm.py

import httpx
import config
from db import save_turn
from translate import translate as mt_translate
from web_tools import web_search, format_results_for_llm
import json




# Модель для "сортувальника" / router-а (може бути легша за основну)
ROUTER_MODEL = getattr(config, "OLLAMA_ROUTER_MODEL", config.OLLAMA_MODEL)

OLLAMA_GENERATE_URL = config.OLLAMA_BASE_URL.rstrip("/") + "/api/generate"




def _split_think_and_answer(text: str) -> tuple[str, str]:
    """
    Розділяє сирий текст моделі на:
    - think_text: те, що всередині <think>...</think> (якщо є)
    - answer: те, що після </think> (те, що треба показувати/озвучувати користувачу)
    """
    if not text:
        return "", ""

    raw = text.strip()
    think = ""
    answer = raw

    open_tag = "<think>"
    close_tag = "</think>"

    if close_tag in raw:
        # шукаємо межі think-блоку
        start = raw.find(open_tag)
        end = raw.rfind(close_tag)

        if start != -1 and end > start:
            think = raw[start + len(open_tag):end].strip()
        else:
            # якщо <think> немає, але є </think> — беремо все перед </think>
            think = raw[:end].strip()

        answer = raw[end + len(close_tag):].strip()

    # прибираємо можливий ```...``` навколо відповіді
    if answer.startswith("```"):
        parts = answer.split("```")
        if len(parts) >= 3:
            answer = parts[-1].strip()

    return think, answer


def _generate_ollama(prompt: str, model: str | None = None) -> str:
    """
    Виклик /api/generate до Ollama, повертає СИРИЙ текст відповіді (може містити <think>).
    """
    if model is None:
        model = config.OLLAMA_MODEL

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # за бажанням можна зафіксувати контекст:
        # "options": {"num_ctx": 2048},
    }

    print("🤖 Запитую модель через Ollama (/api/generate)...")

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OLLAMA_GENERATE_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    raw = (data.get("response") or "").strip()
    return raw


def translate_text(text: str, src: str, dst: str) -> str:
    """
    Перекладає text з мови src в мову dst, використовуючи
    окрему легку модель перекладу (не Ollama).
    """
    return mt_translate(text, src=src, dst=dst)



def ask_ollama(
    user_text: str,
    user_lang: str | None = None,
    web_context: str | None = None,
) -> str:
    """
    Головна функція для асистента.

    Якщо web_context не None — він буде доданий до промпту як
    блок з результатами веб-пошуку, але:
    - user_text для БД не змінюється,
    - переклад працює тільки поверх user_text, web_context не перекладаємо.
    """
    lang = (user_lang or "unknown").lower()
    is_uk = lang.startswith("uk")

    # 1. Готуємо текст для моделі (англійською)
    user_text_en = None
    model_input = user_text

    if is_uk:
        print("🔁 Переклад запиту UK → EN для моделі...")
        user_text_en = translate_text(user_text, src="uk", dst="en")
        print(f"🔁 UK → EN: {user_text_en!r}")
        model_input = user_text_en

    # 2. Основний system-prompt: модель думає і відповідає АНГЛІЙСЬКОЮ
    web_block = ""
    if web_context:
        web_block = (
            "\n\nHere are some web search results that may be relevant:\n"
            f"{web_context}\n"
            "When answering, rely primarily on these results if they are relevant,\n"
            "and say if something is still uncertain.\n"
        )

    system_prompt = (
        "You are a helpful AI assistant.\n"
        "- You ALWAYS think and answer in English.\n"
        f"- The original user language code was: {lang}.\n"
        "- You MAY use <think>...</think> for internal reasoning,\n"
        "  but the final answer for the user MUST be written AFTER the </think> tag,\n"
        "  in clean English.\n"
        "- The final answer should be concise (1–3 sentences) unless the question requires more.\n\n"
        f"{web_block}\n"
        "User message (in English):\n"
        f"{model_input}\n\n"
        "Assistant:"
    )

    raw = _generate_ollama(system_prompt)
    think, answer_en = _split_think_and_answer(raw)

    # ... (далі залишаємо твою логіку THINK, переклад EN→UK, save_turn, return final_reply)

    # 3. THINK MODE в консолі
    if think:
        print("\n🧠 THINK MODE (внутрішні роздуми моделі):")
        print(think)
        print("🧠 END THINK\n")

    if not answer_en:
        answer_en = raw
        print("⚠️ Не знайдено явного </think>, використовую всю відповідь як фінальну (EN).")

    print(f"💬 Відповідь моделі (EN, до перекладу): {answer_en!r}")

    # 4. Формуємо фінальну відповідь мовою користувача + рядки для БД
    final_reply = answer_en
    user_text_to_save = user_text
    assistant_reply_to_save = answer_en

    if is_uk:
        print("🔁 Переклад відповіді EN → UK...")
        answer_uk = translate_text(answer_en, src="en", dst="uk")
        print(f"🔁 EN → UK: {answer_uk!r}")
        final_reply = answer_uk

        # у БД: спочатку українською, потім переклад
        if user_text_en:
            user_text_to_save = f"{user_text}\n\n[EN]\n{user_text_en}"
        else:
            user_text_to_save = user_text

        assistant_reply_to_save = f"{answer_uk}\n\n[EN]\n{answer_en}"

    # 5. Зберігаємо в базу даних
    try:
        save_turn(
            user_text=user_text_to_save,
            user_lang=lang,
            assistant_think=think,
            assistant_reply=assistant_reply_to_save,
        )
    except Exception as e:
        print(f"⚠️ Не вдалося зберегти розмову в БД: {e}")

    # 6. Повертаємо фінальну відповідь (її побачиш у консолі й почуєш у TTS)
    return final_reply


def decide_need_web(user_text: str, user_lang: str | None) -> tuple[bool, str]:
    """
    Вирішує, чи потрібен веб-пошук.
    Логіка класифікації винесена в окрему легку router-модель (Qwen3:0.6b).
    """
    system_prompt = """
    You are a classifier that decides whether a web search is needed.

    Return STRICT JSON with keys:
    - "need_web": true or false
    - "search_query": string (may be empty if need_web is false)

    Use web search when the question is about:
    - current or recent events (news, politics, war, etc.),
    - current prices, availability, product lists,
    - weather right now or forecast,
    - live sports results, timetables, schedules,
    - anything that clearly depends on up-to-date external data.

    Do NOT request web search for:
    - general knowledge that does not change often,
    - programming questions,
    - math and logic,
    - advice that does not need exact current facts.

    Respond with JSON only. Do NOT add any extra text.
    """
    lang = (user_lang or "unknown").lower()
    prompt = f"{system_prompt}\n\nUser language: {lang}\nUser question:\n{user_text}"

    # Тут ми явно юзаємо ROUTER_MODEL (Qwen3:0.6b), а не основну модель.
    raw = _generate_ollama(prompt, model=ROUTER_MODEL)

    try:
        data = json.loads(raw.strip())
        need_web = bool(data.get("need_web"))
        search_query = str(data.get("search_query") or "").strip()
        return need_web, search_query
    except Exception:
        # якщо модель не повернула JSON — просто не робимо веб
        print(f"⚠️ decide_need_web: не вдалося розпарсити JSON: {raw!r}")
        return False, ""

def ask_ollama_smart(user_text: str, user_lang: str | None = None) -> str:
    """
    Обгортка над ask_ollama, яка:
    1) вирішує, чи потрібен веб-пошук;
    2) якщо потрібен — робить пошук і додає web_context у промпт;
    3) інакше працює як звичайний ask_ollama.
    """
    need_web, search_query = decide_need_web(user_text, user_lang)

    if not need_web:
        print("🌐 Веб-пошук не потрібен, відповідаю локально.")
        return ask_ollama(user_text, user_lang)

    if not search_query:
        search_query = user_text

    print(f"🌐 Роблю веб-пошук для запиту: {search_query!r}")
    results = web_search(search_query, max_results=5)

    if not results:
        print("🌐 Веб-пошук нічого не дав, відповідаю як звичайно (без інтернету).")
        return ask_ollama(user_text, user_lang)

    web_context = format_results_for_llm(results)

    return ask_ollama(user_text, user_lang, web_context=web_context)


def ask_ollama_with_web(user_text: str, user_lang: str | None) -> str:
    """
    Варіант запиту до LLM, який спочатку робить веб-пошук,
    а потім дає моделі контекст з результатами.
    """

    lang = (user_lang or "unknown").lower()
    is_uk = lang.startswith("uk")

    # 1. Формуємо запит для пошуку (можна прямо текст користувача)
    search_query = user_text.strip()

    print(f"🌐 Роблю веб-пошук для запиту: {search_query!r}")
    results = web_search(search_query, max_results=5)

    if not results:
        print("🌐 Веб-пошук нічого не дав, відповідаю як звичайно (без інтернету).")
        # fallback — звичайний ask_ollama
        return ask_ollama(user_text, user_lang)

    web_context = format_results_for_llm(results)

    # 2. Готуємо інструкцію для моделі
    #    (щоб вона опиралась на результати пошуку)
    target_lang_name = "English"  # бо модель думає англійською

    prompt = f"""
    You are an AI assistant that answers using web search results.

    User question:
    \"\"\"{user_text}\"\"\"

    Web search results (may contain noise, but also the answer):
    \"\"\"{web_context}\"\"\"

    Your task:
    - Look through the web results and EXTRACT concrete factual information relevant to the question.
    - If the question is about current weather, you MUST try to extract:
    - current temperature,
    - feels-like temperature (if available),
    - weather condition (e.g. cloudy, rain),
    - wind and humidity (if mentioned).
    - Quote NUMERIC values exactly as they appear in the snippets.
    - If you cannot find specific numbers, clearly say: "I could not find exact current values, only general information."

    Answer in {target_lang_name}.
    Give a single, concise paragraph with the extracted data.
    """

    # 3. Викликаємо “сиру” генерацію через Ollama
    #    (у тебе вже є внутрішня функція _generate_ollama)
    raw = _generate_ollama(prompt)

    # 4. Розділяємо think / answer, як ти вже робиш в ask_ollama
    think, answer = _split_think_and_answer(raw)

    # Можна при бажанні зберегти в БД окремо, але для простоти просто повернемо відповідь
    return answer.strip() or raw.strip()
