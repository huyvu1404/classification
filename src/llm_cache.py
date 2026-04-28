"""
LLM result cache - lưu kết quả theo key = title|content|description
để tránh gọi LLM lại cho dữ liệu trùng.
Cache tự động reset vào 12h khuya mỗi ngày.
"""
import json
import os
import hashlib
from datetime import datetime, time
from typing import Optional, Any
from src.settings import PROJECT_DIR

CACHE_FILE = os.path.join(PROJECT_DIR, "data", "llm_cache.json")
CACHE_METADATA_KEY = "__cache_metadata__"


def _make_key(title: str, content: str, description: str) -> str:
    raw = f"{title.strip()}|{content.strip()}|{description.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _should_reset_cache(cache: dict) -> bool:
    """Kiểm tra xem cache có cần reset không (nếu đã qua 12h khuya)."""
    if not cache or CACHE_METADATA_KEY not in cache:
        return False
    
    last_reset = cache[CACHE_METADATA_KEY].get("last_reset")
    if not last_reset:
        return False
    
    try:
        last_reset_dt = datetime.fromisoformat(last_reset)
        now = datetime.now()
        
        # Tính thời điểm 12h khuya gần nhất
        midnight_today = datetime.combine(now.date(), time(0, 0, 0))
        
        # Nếu last_reset trước 12h khuya hôm nay và hiện tại đã qua 12h khuya
        if last_reset_dt < midnight_today:
            return True
        
        return False
    except Exception:
        return False


def _reset_cache() -> dict:
    """Tạo cache mới với metadata."""
    return {
        CACHE_METADATA_KEY: {
            "last_reset": datetime.now().isoformat(),
            "version": "1.0"
        }
    }


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            
            # Kiểm tra xem có cần reset không
            if _should_reset_cache(cache):
                print("🔄 Cache đã qua 12h khuya, đang reset...")
                return _reset_cache()
            
            return cache
        except Exception:
            return _reset_cache()
    return _reset_cache()


def save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    
    # Đảm bảo metadata luôn có
    if CACHE_METADATA_KEY not in cache:
        cache[CACHE_METADATA_KEY] = {
            "last_reset": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_cached(cache: dict, title: str, content: str, description: str) -> Optional[Any]:
    key = _make_key(title, content, description)
    # Bỏ qua metadata key
    if key == CACHE_METADATA_KEY:
        return None
    return cache.get(key)


def set_cached(cache: dict, title: str, content: str, description: str, value: Any) -> None:
    key = _make_key(title, content, description)
    # Không cho phép ghi đè metadata
    if key == CACHE_METADATA_KEY:
        return
    cache[key] = value
