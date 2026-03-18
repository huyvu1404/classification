import re
import json
import os
import aiohttp
import asyncio
import emoji
import unicodedata
import nest_asyncio
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from src.settings import PROJECT_DIR
from src.utils import sanitize_excel_values

load_dotenv()
nest_asyncio.apply()


def _load_prompt(prompt_name: str, **kwargs) -> str:
    path = os.path.join(PROJECT_DIR, "src/prompts/detector", f"{prompt_name}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().format(**kwargs)
    except FileNotFoundError:
        print(f"Prompt not found: {path}")
        return ""
    except Exception as e:
        print(f"Error loading prompt {prompt_name}: {e}")
        return ""


class Detector:
    INTEREST_KEYWORDS = [
        "giá bao nhiêu", "mua ở đâu", "order", "đặt hàng",
    ]

    SEEDING_KEYWORDS = [
        "đáng mua", "nên mua", "phải mua", "mua ngay", "đáng thử", "nên thử",
        "chất lượng tốt", "chất lượng cao", "sản phẩm tốt", "sản phẩm chất",
        "đáng tiền", "xứng đáng", "giá tốt", "giá rẻ", "giá hợp lý",
        "rẻ mà ngon", "bổ rẻ", "tiết kiệm",
        "dùng rất tốt", "dùng ổn", "dùng ok", "trải nghiệm tốt", "hài lòng",
        "không thất vọng", "đáng để thử",
        "mua đi", "mua thôi", "mua nha", "mua đi bạn", "ai chưa mua thì mua",
        "nên sắm", "đáng sắm", "mình đã mua", "mình dùng rồi",
        "tốt hơn", "ngon hơn", "chất hơn", "đỉnh hơn", "hơn hẳn",
        "hot", "trend", "đang hot", "đang viral",
        "ai cũng mua", "bán chạy", "cháy hàng", "mọi người đều", "ai cũng khen",
    ]

    QUALITY_KEYWORDS = [
        "ngon", "tuyệt", "tốt", "chất lượng", "xuất sắc", "tuyệt vời", "đỉnh",
        "ok", "oke", "thơm", "béo", "mịn", "đậm đà",
        "dở", "tệ", "kém", "không ngon", "không tốt", "tồi", "sữa bị chua",
        "nhạt", "loãng", "tanh", "hôi", "không đáng", "thất vọng", "kém chất lượng",
    ]

    def __init__(self, project_name: str = "Vinamilk"):
        self.project_name = project_name
        self.project_keywords = self._load_project_keywords()
        self.llm_api_url = os.getenv("LLM_API_URL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_model = os.getenv("LLM_MODEL", "")
        self._confidence_threshold = 0.5
    
    def _load_project_keywords(self) -> List[str]:
        path = os.path.join(PROJECT_DIR, "src/keywords/keywords.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get(self.project_name, [])
        except Exception as e:
            print(f"Could not load keywords: {e}")
            return []

    # ── Text helpers ──────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        
        text = unicodedata.normalize("NFC", str(text))
        
        text = emoji.replace_emoji(text, replace="")
        
        text = re.sub(r"https?://\S+|www\.\S+", "", str(text))
        
        text = re.sub(r'[!?.,;:\"\'\(\)\[\]\{\}\/\\\|@#\$%\^\&\*\+=\-_~`<>]', " ", text)
        
        return re.sub(r"\s+", " ", text).lower().strip()

    def _get_str(self, row: pd.Series, col: str) -> str:
        val = row.get(col, "")
        return str(val).strip() if col in row.index and not pd.isna(val) else ""

    def _merge_text(self, row: pd.Series) -> str:
        parts = []
        for col in ("Title", "Content", "Description"):
            val = self._get_str(row, col)
            if val and val not in parts:
                parts.append(val)
        return " ".join(parts)

    def _is_comment(self, row: pd.Series) -> bool:
        return "comment" in self._get_str(row, "Type").lower()

    def _is_empty(self, text: str) -> bool:
        text = self._normalize(text)
        return not text.strip()

    # ── Keyword matching ──────────────────────────────────────────────────────

    def _find_matches(self, text: str, keywords: List[str]) -> List[str]:
        norm = self._normalize(text)
        return [kw for kw in keywords if (nkw := self._normalize(kw)) and (nkw in norm or nkw.replace(" ", "") in norm)]

    def _check_keywords(self, row: pd.Series, keywords: List[str], kw_type: str = "general") -> bool:
        is_comment = self._is_comment(row)
        if is_comment and kw_type in ("interest", "seeding", "quality"):
            content_matches = self._find_matches(self._get_str(row, "Content"), keywords)
            if not content_matches:
                return False
            title_matches = self._find_matches(self._get_str(row, "Title"), self.project_keywords)
            if not title_matches:
                return False
            return True
        
        text = self._get_str(row, "Content") if (is_comment and kw_type == "project") else self._merge_text(row)
        matches = self._find_matches(text, keywords)
        return bool(matches)
    
    def _check_author(self, author: str) ->List[str]:
        if self.project_name.lower().strip() in author.lower().strip():
            return True, 0.95
        
    async def _call_llm(self, prompt: str, session: aiohttp.ClientSession) -> bool:
        try:
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 10,
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
                    answer = result["choices"][0]["message"]["content"].strip().lower()
                    return "yes" in answer
                return False
        except Exception:
            return False

    async def _llm_detect(self, row: pd.Series, session: aiohttp.ClientSession) -> Optional[str]:
        """Single LLM call returning is_relevant."""
        
        comment = self._get_str(row, "Content")
        
        data = json.dumps({
            "content": self._get_str(row, "Title"),
            "comment": self._get_str(row, "Content")
        }, ensure_ascii=False)
        
        name_tag_prompt = _load_prompt("name_tag_detection", comment=comment)
        relevance_prompt = _load_prompt("content_relevance", project=self.project_name, data=data)
        
        if not name_tag_prompt and not relevance_prompt:
            return None
        
        include_name_tag = await self._call_llm(name_tag_prompt, session)
        if include_name_tag:
            return "Yes"
        
        is_relevant = await self._call_llm(relevance_prompt, session)
        if is_relevant:
            return "Yes"
        
        return None

    def _detect_sync(self, row: pd.Series) -> Tuple[Optional[str], str]:
        """Returns (result, rule, explanation) or (None, '', '') if LLM needed."""

        sentiment = self._get_str(row, "Sentiment").lower()
        if sentiment in ("negative", "positive"):
            return "Yes", "Rule 1: Sentiment"

        if "trò chơi nhỏ" in self._get_str(row, "Labels1").lower():
            return "Yes", "Rule 2: Labels1"

        channel = self._get_str(row, "Channel")
        if channel in ("E-commerce", "News"):
            return "Yes", "Rule 3: Channel"

        if self._check_author(self._get_str(row, "Author")):
            return "Yes", "Rule 4: Author"
        
        if self._is_comment(row) and self._is_empty(self._get_str(row, "Content")):
            return "No", "Rule 5: Empty Content"

        for rule_num, kws, kw_type in (
            (6, self.project_keywords, "project"),
            (7, self.INTEREST_KEYWORDS, "interest"),
            (8, self.SEEDING_KEYWORDS, "seeding"),
            (9, self.QUALITY_KEYWORDS, "quality"),
        ):
            if not kws:
                continue
            matched = self._check_keywords(row, kws, kw_type)
            if matched:
                label = kw_type.capitalize() if rule_num > 5 else "Project"
                return "Yes", f"Rule {rule_num}: {label} Keywords"

        return None, ""

    # ── Async LLM detection ───────────────────────────────────────────────────

    async def _detect_row_llm(self, idx: int, row: pd.Series, session: aiohttp.ClientSession) -> Tuple[int, str, str]:
        title = self._get_str(row, "Title")
        content = self._get_str(row, "Content")

        if not content.strip():
            return idx, "No", "Empty Content"

        relevant = await self._llm_detect(row, session)
        if relevant is not None:
            title_matches = self._find_matches(title, self.project_keywords)
            if title_matches:
                return idx, relevant.capitalize(), "Rule 9: LLM Relevance"


        return idx, "No", "Can't Detect"

    # ── Public API ────────────────────────────────────────────────────────────

    async def detect_async(
        self,
        df: pd.DataFrame,
        use_llm: bool = False,
        batch_size: int = 20,
        max_concurrent: int = 10,
    ) -> pd.DataFrame:
        """Two-phase: sync rules first, then concurrent LLM batches."""
        # Use df index as key to guarantee correct row assignment
        results: Dict[int, Tuple[str, str]] = {}
        needs_llm: List[Tuple[int, pd.Series]] = []

        # Phase 1: sync rules
        for idx, row in df.iterrows():
            result, rule = self._detect_sync(row)
            if result is not None:
                results[idx] = (result, rule)
            else:
                needs_llm.append((idx, row))

        # Phase 2: concurrent LLM
        if needs_llm and use_llm:
            print(f"\nLLM: processing {len(needs_llm)} rows (batch={batch_size}, concurrent={max_concurrent})...")
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_batch(batch: List[Tuple[int, pd.Series]], session: aiohttp.ClientSession):
                async with semaphore:
                    out = []
                    for idx, row in batch:
                        try:
                            out.append(await self._detect_row_llm(idx, row, session))
                        except Exception as e:
                            print(f"Error on row {idx}: {e}")
                            out.append((idx, "No", "Error"))
                    return out

            async def run_llm():
                async with aiohttp.ClientSession() as session:
                    batches = [needs_llm[i:i + batch_size] for i in range(0, len(needs_llm), batch_size)]
                    with tqdm(total=len(needs_llm), desc="LLM") as pbar:
                        for coro in asyncio.as_completed([limited_batch(b, session) for b in batches]):
                            batch_results = await coro
                            for idx, result, rule in batch_results:
                                results[idx] = (result, rule)
                            pbar.update(len(batch_results))
            await run_llm()
        else:
            if needs_llm and not use_llm:
                print(f"LLM disabled. Marking {len(needs_llm)} rows as No.")
            for idx, _ in needs_llm:
                results[idx] = ("No", "No Match")

        # Assign back using df index — guaranteed correct mapping
        df = df.copy()
        df["Yes/No"] = df.index.map(lambda i: results[i][0])
        # df["Rule"] = df.index.map(lambda i: results[i][1])

        rule_stats: Dict[str, int] = {}
        for _, rule in results.values():
            rule_stats[rule] = rule_stats.get(rule, 0) + 1

        print("\nRule stats:", {k: v for k, v in rule_stats.items() if v})
        yes = (df["Yes/No"] == "Yes").sum()
        print(f"Yes: {yes} ({yes / len(df) * 100:.1f}%)  No: {len(df) - yes}")

        return sanitize_excel_values(df)

    def detect(self, df: pd.DataFrame, use_llm: bool = False, batch_size: int = 20, max_concurrent: int = 10) -> pd.DataFrame:
        """Sync wrapper around detect_async."""
        async def _run():
            return await self.detect_async(df, use_llm=use_llm, batch_size=batch_size, max_concurrent=max_concurrent)
        return asyncio.get_event_loop().run_until_complete(_run())


async def detect_relevant(
    df: pd.DataFrame,
    project_name: str = "Vinamilk",
    use_llm: bool = False,
    batch_size: int = 20,
    max_concurrent: int = 10,
) -> pd.DataFrame:
    return await Detector(project_name=project_name).detect_async(df, use_llm=use_llm, batch_size=batch_size, max_concurrent=max_concurrent)
