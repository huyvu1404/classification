import re
import json
import numpy as np
import os
import unicodedata
import asyncio
from typing import Dict, List, Optional, Tuple
import aiohttp
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from src.models import loader
from src.settings import PROJECT_DIR
from src.utils import sanitize_excel_values
from src.llm_cache import load_cache, save_cache, get_cached, set_cached

load_dotenv()

LLM_RULES_PATH = os.path.join(PROJECT_DIR, "src/rules/llm-rules.json")
LABEL_RULE_PATH = os.path.join(PROJECT_DIR, "src/rules/label-rules.json")


class LabelClassifier:
    def __init__(self, project_name: str = "ShopeeFood", llm_rules_path: str = LLM_RULES_PATH, label_rule_path: str = LABEL_RULE_PATH):
        with open(llm_rules_path, "r", encoding="utf-8") as f:
            metadata = json.load(f).get("metadata", {})
        with open(label_rule_path, "r", encoding="utf-8") as f:
            self.kws_data = json.load(f)

        self.project_name = project_name
        self.default_labels = metadata.get("default_labels", {})
        self.confidence_threshold = metadata.get("confidence_threshold", 0.6)
        self.valid_labels: List[str] = []

        self.llm_api_url = os.getenv("LLM_API_URL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_model = os.getenv("LLM_MODEL", "")

        self.model = None
        self.label_encoder = None
        self._prompt_template: Optional[str] = None

    # ── Text helpers ──────────────────────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        text = unicodedata.normalize("NFC", str(text))
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r'[!?.,;:\"\'\(\)\[\]\{\}\/\\\|@#\$%\^\&\*\+=\-_~`<>]', " ", text)
        return re.sub(r"\s+", " ", text).lower().strip()

    def _extract_text(self, data: Dict) -> str:
        is_topic = "topic" in data.get("type", "").lower()
        title = data.get("title", "").strip()
        content = data.get("content", "").strip()
        description = data.get("description", "").strip()

        parts = []
        if title:
            parts.append(title)
        if content and content != title:
            parts.append(content)
        if is_topic and description and description not in (title, content):
            parts.append(description)
        return " | ".join(parts)

    # ── Label helpers ─────────────────────────────────────────────────────────

    def get_default_label(self) -> str:
        return self.default_labels.get(self.project_name, "Unknown")

    def _get_valid_labels(self) -> List[str]:
        if not self.valid_labels:
            project_kws = self.kws_data.get(self.project_name, {})
            if not project_kws and self.project_name == "SPX Express":
                project_kws = self.kws_data.get("SPX", {})
            self.valid_labels = list(project_kws.keys())
        return self.valid_labels

    # ── Rule matching ─────────────────────────────────────────────────────────

    def _check_site_rules(self, data: Dict) -> Tuple[Optional[str], Optional[str], float]:
        project_kws = self.kws_data.get(self.project_name)
        if not project_kws and self.project_name == "SPX Express":
            project_kws = self.kws_data.get("SPX")
        if not project_kws:
            return None, None, 0

        site_id = str(data.get("siteId", "")).strip()
        site_name = self._normalize_text(data.get("siteName", ""))
        channel = self._normalize_text(data.get("channel", ""))
        label_field = self._normalize_text(data.get("label", ""))
        author = self._normalize_text(data.get("author", ""))

        for label, rules in project_kws.items():
            for rule in rules if isinstance(rules, list) else [rules]:
                if "condition" in rule or "keywords" in rule:
                    continue
                for sid in rule.get("siteIds", rule.get("siteId", [])):
                    if site_id and site_id == str(sid).strip():
                        return label, "SITE ID", 0.95
                for sn in rule.get("siteNames", rule.get("siteName", [])):
                    nsn = self._normalize_text(sn)
                    if nsn and (nsn in site_name or site_name in nsn):
                        return label, "SITE NAME", 0.95
                for ch in rule.get("channels", []):
                    nch = self._normalize_text(ch)
                    if nch and nch in channel:
                        return label, "CHANNEL", 0.95
                for lb in rule.get("labels", []):
                    nlb = self._normalize_text(lb)
                    if nlb and nlb in label_field:
                        return label, "LABEL FIELD", 0.95
                for au in rule.get("authors", rule.get("author", [])):
                    nau = self._normalize_text(au)
                    if nau and nau in author:
                        return label, "AUTHOR", 0.95
        return None, None, 0

    def _check_keyword(self, data: Dict) -> Tuple[Optional[str], Optional[str], float]:
        project_kws = self.kws_data.get(self.project_name)
        if not project_kws and self.project_name == "SPX Express":
            project_kws = self.kws_data.get("SPX")
        if not project_kws:
            return None, None, 0

        normalized_text = self._normalize_text(self._extract_text(data))
        buzz_type = data.get("type", "").lower()
        site_name = data.get("siteName", "").lower()

        for label, rules in project_kws.items():
            for rule in rules if isinstance(rules, list) else [rules]:
                keywords = rule.get("keywords", [])
                if not keywords:
                    continue
                condition = rule.get("condition", {})
                if "type" in condition and condition["type"].lower() not in buzz_type:
                    continue
                if "siteName" in condition:
                    req = condition["siteName"].lower()
                    if req and req != site_name:
                        continue
                for kw in keywords:
                    if self._normalize_text(kw) in normalized_text:
                        return label, "KEYWORD MATCHING", 0.95
        return None, None, 0

    # ── Model ─────────────────────────────────────────────────────────────────

    def load_pretrained_model(self):
        try:
            self.model, self.label_encoder = loader(self.project_name)
        except Exception as e:
            print(f"Could not load pretrained model for {self.project_name}: {e}")

    def model_based_classification(self, text: str) -> Tuple[Optional[str], float]:
        if self.model is None or self.label_encoder is None:
            return None, 0.0
        try:
            normalized = self._normalize_text(text)
            prediction = self.model.predict([normalized])[0]
            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba([normalized])[0]
                confidence = float(np.max(proba))
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function([normalized])
                if isinstance(scores, np.ndarray):
                    raw = float(scores[0]) if scores.ndim == 1 else float(scores[0].max())
                    confidence = float(1 / (1 + np.exp(-raw)))
            return self.label_encoder.inverse_transform([prediction])[0], confidence
        except Exception as e:
            print(f"Model prediction error: {e}")
            return None, 0.0

    # ── LLM ───────────────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str, session: aiohttp.ClientSession) -> Optional[str]:
        try:
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 20,
                "stream": False,
            }
            async with session.post(
                f"{self.llm_api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result["choices"][0]["message"]["content"].strip()
                return None
        except Exception:
            return None

    def _load_prompt_template(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        path = os.path.join(PROJECT_DIR, "src/prompts/classifier", f"{self.project_name}.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._prompt_template = f.read()
        except FileNotFoundError:
            print(f"Prompt not found: {path}")
            self._prompt_template = ""
        return self._prompt_template

    def _build_json_data(self, data: Dict) -> str:
        buzz_type = "comment" if "comment" in data.get("type", "").lower() else "topic"
        return json.dumps({
            "type": buzz_type,
            "title": data.get("title", ""),
            "content": data.get("content", ""),
            "description": data.get("description", ""),
            "siteName": data.get("siteName", ""),
        }, ensure_ascii=False, indent=4)

    async def llm_classification(self, data: Dict, session: aiohttp.ClientSession, cache: dict = None) -> Tuple[Optional[str], float]:
        if not self.llm_api_url or not data:
            return None, 0.0
        template = self._load_prompt_template()
        if not template:
            return None, 0.0

        title = data.get("title", "")
        content = data.get("content", "")
        description = data.get("description", "")

        if cache is not None:
            cached = get_cached(cache, title, content, description)
            if cached is not None:
                return cached  # (label, confidence)

        answer = await self._call_llm(template.format(data=self._build_json_data(data)), session)
        if not answer:
            return None, 0.0

        confidence = 0.85
        raw = answer.strip()
        if "|" in raw:
            parts = raw.split("|", 1)
            raw = parts[0].strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                pass

        if self.project_name == "Giao Hàng Nhanh" and "NGƯỜI GỬI HÀNG" in raw.upper():
            raw = "NGƯỜI MUA HÀNG"

        raw_upper = raw.upper()
        for label in self._get_valid_labels():
            label_upper = label.upper()
            if label_upper == raw_upper or label_upper in raw_upper:
                result = (label, confidence)
                if cache is not None:
                    set_cached(cache, title, content, description, result)
                return result

        if cache is not None:
            set_cached(cache, title, content, description, (None, 0.0))
        return None, 0.0

    # ── Classification pipeline ───────────────────────────────────────────────

    def classify_sync_rules(self, data: Dict) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Returns (label, method, confidence) or (None, None, None) if LLM needed."""
        text = self._extract_text(data)

        label, method, conf = self._check_site_rules(data)
        if label and conf > self.confidence_threshold:
            return label, method, conf

        label, method, conf = self._check_keyword(data)
        if label and conf > self.confidence_threshold:
            return label, method, conf

        if not text:
            return self.get_default_label(), "NoText", 0.0

        return None, None, None

    async def classify_async(self, data: Dict, session: aiohttp.ClientSession, cache: dict = None) -> Tuple[str, str, float]:
        label, method, conf = self.classify_sync_rules(data)
        if label is not None:
            return label, method, conf

        label, conf = await self.llm_classification(data, session, cache)
        if label and conf >= self.confidence_threshold:
            return label, "LLM", conf

        text = self._extract_text(data)
        label, conf = self.model_based_classification(text)
        if label and conf >= self.confidence_threshold:
            return label, "PretrainedModel", conf

        return self.get_default_label(), "LowConfidence", 0.0

    def classify(self, data: Dict) -> Tuple[str, str, float]:
        """Sync wrapper around classify_async."""
        async def _run():
            async with aiohttp.ClientSession() as session:
                return await self.classify_async(data, session)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_run())


# ── Batch helpers ─────────────────────────────────────────────────────────────

async def _classify_batch(classifier: LabelClassifier, batch: List[Tuple], session: aiohttp.ClientSession, cache: dict = None) -> List[Tuple]:
    async def _classify_one(idx, data):
        try:
            label, method, conf = await classifier.classify_async(data, session, cache)
            return (idx, label, method, conf)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            return (idx, "Unknown", "Error", 0.0)

    return await asyncio.gather(*[_classify_one(idx, data) for idx, data in batch])


def _row_to_dict(row) -> Dict:
    def safe(col):
        val = getattr(row, col, "")
        return str(val) if pd.notna(val) else ""
    return {
        "id": safe("Id"), "topic": safe("Topic"),
        "title": safe("Title"), "content": safe("Content"),
        "description": safe("Description"), "type": safe("Type"),
        "siteName": safe("SiteName"), "author": safe("Author"),
        "siteId": safe("SiteId"), "channel": safe("Channel"),
        "label": safe("Labels1"),
        "parentId": safe("ParentId"),
    }



async def classify_category(
    df: pd.DataFrame,
    project_name: str,
    batch_size: int = 20,
    max_concurrent: int = 10,
    tqdm_func=None,
    log_func=None,
) -> pd.DataFrame:
    """Classification pipeline:
    - Phase 1: sync rules cho tất cả rows
    - Phase 2: LLM cho tất cả rows chưa có kết quả (không inherit)
    """
    required = ["Id", "Title", "Content", "Description", "Type", "SiteName"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: missing columns {missing}")

    if tqdm_func is None:
        tqdm_func = tqdm
    if log_func is None:
        log_func = print

    classifier = LabelClassifier(project_name=project_name)
    classifier.load_pretrained_model()
    classifier._get_valid_labels()

    results: Dict[int, Tuple[str, str, float]] = {}
    needs_llm_indices: List[int] = []

    log_func(f"📋 Tổng số dòng dữ liệu: **{len(df)}**")

    # Phase 1: sync rules
    for row in tqdm_func(df.itertuples(), total=len(df), desc="⚙️ Sync Rules"):
        idx = row.Index
        data = _row_to_dict(row)
        label, method, conf = classifier.classify_sync_rules(data)
        if label is not None:
            results[idx] = (label, method, conf)
        else:
            needs_llm_indices.append(idx)

    rule_assigned = len(results)
    log_func(f"✅ Đã gán qua rule: **{rule_assigned}** dòng — Cần LLM: **{len(needs_llm_indices)}** dòng")

    # Phase 2: LLM cho tất cả rows chưa có kết quả
    cache = load_cache()

    if needs_llm_indices:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def classify_one(idx: int, session: aiohttp.ClientSession):
            async with semaphore:
                try:
                    data = _row_to_dict(df.iloc[idx])
                    label, method, conf = await classifier.classify_async(data, session, cache)
                    return (idx, label, method, conf)
                except Exception as e:
                    print(f"Error on row {idx}: {e}")
                    return (idx, classifier.get_default_label(), "Error", 0.0)

        pbar = tqdm_func(total=len(needs_llm_indices), desc="🤖 LLM")
        async with aiohttp.ClientSession() as session:
            # Batch processing để tránh tạo quá nhiều tasks cùng lúc
            for i in range(0, len(needs_llm_indices), batch_size):
                batch_indices = needs_llm_indices[i:i+batch_size]
                tasks = [classify_one(idx, session) for idx in batch_indices]
                
                for coro in asyncio.as_completed(tasks):
                    idx, label, method, conf = await coro
                    results[idx] = (label, method, conf)
                    pbar.update(1)
        pbar.close()
        save_cache(cache)
    else:
        log_func("ℹ️ Không có rows nào cần LLM.")

    # Assign back
    df = df.copy()
    df["Label"] = df.index.map(lambda i: results.get(i, (classifier.get_default_label(), "Unknown", 0.0))[0])
    
    return sanitize_excel_values(df)
