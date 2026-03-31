"""
LLM result cache - lưu kết quả theo key = title|content|description
để tránh gọi LLM lại cho dữ liệu trùng.
"""
import json
import os
import hashlib
from typing import Optional, Any
from src.settings import PROJECT_DIR

CACHE_FILE = os.path.join(PROJECT_DIR, "data", "llm_cache.json")


def _make_key(title: str, content: str, description: str) -> str:
    raw = f"{title.strip()}|{content.strip()}|{description.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_cached(cache: dict, title: str, content: str, description: str) -> Optional[Any]:
    key = _make_key(title, content, description)
    return cache.get(key)


def set_cached(cache: dict, title: str, content: str, description: str, value: Any) -> None:
    key = _make_key(title, content, description)
    cache[key] = value
