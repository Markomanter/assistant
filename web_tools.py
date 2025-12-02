# web_tools.py
from typing import List, Dict
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Повертає список результатів пошуку.
    Кожен результат має ключі: title, href, body.
    """
    query = (query or "").strip()
    if not query:
        return []

    print(f"[web] ➜ Пошук: {query!r}")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    print(f"[web] ✓ Отримано результатів: {len(results)}")
    for i, r in enumerate(results[:3], 1):
        print(f"[web] {i}: {r.get('title')!r} | {r.get('href')!r}")

    return results


def format_results_for_llm(results: list[dict]) -> str:
    """
    Робить компактний текстовий блок з результатів,
    який зручно підсунути в промпт моделі.
    """
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        url = r.get("href") or ""
        snippet = r.get("body") or ""
        lines.append(f"[{i}] {title}\nURL: {url}\n{snippet}")

    return "\n\n".join(lines)
