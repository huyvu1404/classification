import re
import json
import os
import unicodedata
import asyncio
import pandas as pd
from typing import List, Dict, Tuple
from src.settings import PROJECT_DIR
from src.utils import sanitize_excel_values
import aiohttp
AIOHTTP_AVAILABLE = True
from dotenv import load_dotenv
load_dotenv()

def load_prompt(prompt_name: str, **kwargs) -> str:
    """
    Load and format prompt from prompts folder
    
    Parameters:
    -----------
    prompt_name : str
        Name of the prompt file (without .txt extension)
    **kwargs : dict
        Variables to format in the prompt
    
    Returns:
    --------
    str
        Formatted prompt
    """
    prompt_path = os.path.join(PROJECT_DIR, "src/prompts", f"{prompt_name}.txt")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template.format(**kwargs)
    except FileNotFoundError:
        print(f"Prompt file not found: {prompt_path}")
        return ""
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return ""

class Detector:
    """
    Detector class to classify dataframe rows as Yes/No based on priority rules:
    1. Sentiment is Negative or Positive => Yes
    2. Labels1 is "Trò chơi nhỏ - trực tuyển" => Yes
    3. Channel is E-commerce or News => Yes
    4. Content is empty after removing whitespace, emojis, and links (only for comments) => No
    5. Text contains keywords from keywords folder => Yes
    6. Text contains interest keywords (inbox, ib, ibx, nhắn tin, ...) => Yes
    7. Text contains seeding keywords => Yes
    8. Text contains product quality keywords (ngon, dở, không ngon, ...) => Yes
    9. Text contains name tags @(name/id) or just names => Use LLM to detect => Yes
    10. Text is relevant to topic => Use LLM to detect => Yes
    """
    
    def __init__(self, project_name: str = "Vinamilk"):
        self.project_name = project_name
        self.keywords_path = os.path.join(PROJECT_DIR, "src/keywords/keywords.json")
        self.project_keywords = self._load_project_keywords()
        
        # LLM API configuration
        self.llm_api_url = os.getenv("LLM_API_URL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        
        # Define interest keywords (inbox, contact, etc.)
        self.interest_keywords = [
            "inbox", "ib", "ibx", "inb", "nhắn tin", "tin nhắn", "tn",
            "liên hệ", "chat", "message", "msg",
            "hỏi giá", "giá bao nhiêu",
            "mua ở đâu", "order", "đặt hàng"
        ]
        
        # Define seeding keywords - từ ngữ khen sản phẩm để tạo xu hướng mua
        self.seeding_keywords = [
            # Khen chung chung
            "đáng mua", "nên mua", "phải mua",
            "mua ngay", "đáng thử", "nên thử",
            "recommend", "gợi ý", "khuyên dùng",
            
            # Khen chất lượng
            "chất lượng tốt", "chất lượng cao",
            "sản phẩm tốt", "sản phẩm chất",
            "đáng tiền", "xứng đáng",
            
            # Khen giá trị
            "giá tốt", "giá rẻ", "giá hợp lý",
            "rẻ mà ngon", "bổ rẻ", "tiết kiệm",
            
            # Khen trải nghiệm
            "dùng rất tốt", "dùng ổn", "dùng ok",
            "trải nghiệm tốt", "hài lòng",
            "không thất vọng", "đáng để thử",
            
            # Khuyến khích mua
            "mua đi", "mua thôi", "mua nha", "mua đi bạn",
            "ai chưa mua thì mua", "nên sắm", "đáng sắm",
            "mình đã mua", "mình dùng rồi",
            
            # So sánh tích cực
            "tốt hơn", "ngon hơn", "chất hơn",
            "đỉnh hơn", "hơn hẳn",
            
            # Viral/trending
            "hot", "trend", "đang hot", "đang viral",
            "ai cũng mua", "bán chạy", "cháy hàng",
            "mọi người đều", "ai cũng khen"
        ]
        
        # Define product quality keywords
        self.quality_keywords = [
            # Positive
            "ngon", "tuyệt", "tốt", "chất lượng",
            "xuất sắc", "tuyệt vời", "đỉnh",
            "ok", "oke", "good", "great", "excellent", "amazing", "delicious",
            "thơm", "béo", "mịn", "đậm đà",
            
            # Negative
            "dở", "tệ", "kém", "không ngon",
            "không tốt", "tồi", "bad", "terrible", "awful",
            "nhạt", "loãng", "tanh", "hôi",
            "không đáng", "thất vọng", "kém chất lượng"
        ]
    
    def _load_project_keywords(self) -> List[str]:
        """Load keywords for the project from keywords.json"""
        try:
            with open(self.keywords_path, 'r', encoding='utf-8') as f:
                keywords_data = json.load(f)
                return keywords_data.get(self.project_name, [])
        except FileNotFoundError:
            print(f"Keywords file not found: {self.keywords_path}")
            return []
        except Exception as e:
            print(f"Error loading keywords: {e}")
            return []
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching
        - Keeps Vietnamese tones (giữ dấu thanh: à, ô, ư, etc.)
        - Removes punctuation (?, !, ., etc.)
        - Normalizes whitespace
        - Converts to lowercase
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove punctuation but keep Vietnamese characters with tones
        # Remove: ! ? . , ; : " ' ( ) [ ] { } / \ | @ # $ % ^ & * + = - _ ~ ` < >
        text = re.sub(r'[!?.,;:\"\'\(\)\[\]\{\}\/\\\|@#\$%\^\&\*\+=\-_~`<>]', ' ', text)
        
        # Normalize whitespace and convert to lowercase
        return re.sub(r"\s+", " ", text).lower().strip()
    
    def _merge_text_columns(self, row: pd.Series) -> str:
        """Merge Title, Content, Description into single text"""
        parts = []
        for col in ["Title", "Content", "Description"]:
            if col in row.index and row[col] and not pd.isna(row[col]):
                val = str(row[col]).strip()
                if val and val not in parts:
                    parts.append(val)
        return " ".join(parts)
    def _clean_content_for_empty_check(self, text: str) -> str:
        """
        Clean content by removing whitespace, emojis, and links
        Used to check if content is truly empty

        Parameters:
        -----------
        text : str
            Text to clean

        Returns:
        --------
        str
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""

        text = str(text)

        # Remove URLs (http, https, www)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove emojis and special unicode characters
        # Keep only: letters (including Vietnamese), numbers, basic punctuation
        text = re.sub(r'[^\w\s\.,!?\-àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _is_content_empty(self, row: pd.Series, is_comment: bool = False) -> bool:
        """
        Check if row content is empty after cleaning

        Parameters:
        -----------
        row : pd.Series
            Row from dataframe
        is_comment : bool
            If True, only check Content column. If False, check all columns.

        Returns:
        --------
        bool
            True if content is empty after cleaning
        """
        # For comments, only check Content column
        if is_comment:
            content = str(row.get("Content", "")) if "Content" in row.index and not pd.isna(row.get("Content")) else ""
            cleaned = self._clean_content_for_empty_check(content)
            return len(cleaned) == 0
        
        # For topics, check all text from Title, Content, Description
        text = self._merge_text_columns(row)
        cleaned = self._clean_content_for_empty_check(text)
        return len(cleaned) == 0

    
    def _get_text_for_keyword_type(self, row: pd.Series, keyword_type: str) -> str:
        """
        Get appropriate text based on keyword type and row type (comment/topic)
        
        Parameters:
        -----------
        row : pd.Series
            Row from dataframe
        keyword_type : str
            Type of keyword: 'project', 'interest', 'seeding', 'quality'
        
        Returns:
        --------
        str
            Text to check for keywords
        """
        # Check if it's a comment or topic
        is_comment = False
        if "Type" in row.index:
            row_type = str(row["Type"]).lower()
            is_comment = "comment" in row_type
        
        if is_comment:
            # For comments
            if keyword_type == "project":
                # Project keywords: check Content only
                content = str(row.get("Content", "")) if "Content" in row.index and not pd.isna(row.get("Content")) else ""
                return content
            else:
                # Interest, seeding, quality keywords: check Content + Title
                parts = []
                
                # Add Content
                if "Content" in row.index and row["Content"] and not pd.isna(row["Content"]):
                    parts.append(str(row["Content"]))
                
                # Add Title (to check for project keywords mentioned in comment title)
                if "Title" in row.index and row["Title"] and not pd.isna(row["Title"]):
                    parts.append(str(row["Title"]))
                
                return " ".join(parts)
        else:
            # For topics: use all fields (Title, Content, Description)
            return self._merge_text_columns(row)
    
    def _find_matched_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """
        Find which keywords matched in the text
        
        Parameters:
        -----------
        text : str
            Text to search in
        keywords : List[str]
            List of keywords to check
        
        Returns:
        --------
        List[str]
            List of matched keywords
        """
        matched = []
        text_normalized = self._normalize_text(text)
        
        for keyword in keywords:
            keyword_normalized = self._normalize_text(keyword)
            if not keyword_normalized:
                continue
            
            # Check exact match
            if keyword_normalized in text_normalized:
                matched.append(keyword)
                continue
            
            # Check without spaces
            keyword_no_space = keyword_normalized.replace(" ", "")
            # text_no_space = text_normalized.replace(" ", "")
            if keyword_no_space in text_normalized:
                matched.append(keyword)
                continue
            
        return matched
    
    def _contains_keywords(self, row: pd.Series, keywords: List[str], keyword_type: str = "general") -> tuple:
        """
        Check if row contains any keyword from the list
        
        For comments with interest/seeding/quality keywords:
        - Content must contain the keyword (interest/seeding/quality)
        - Title must contain project keyword
        - Both conditions must be met
        
        Parameters:
        -----------
        row : pd.Series
            Row from dataframe
        keywords : List[str]
            List of keywords to check
        keyword_type : str
            Type of keyword: 'project', 'interest', 'seeding', 'quality', 'general'
        
        Returns:
        --------
        tuple
            (bool, List[str]) - (has_match, matched_keywords)
        """
        # Check if it's a comment
        is_comment = False
        if "Type" in row.index:
            row_type = str(row["Type"]).lower()
            is_comment = "comment" in row_type
        
        # Special handling for comments with interest/seeding/quality keywords
        if is_comment and keyword_type in ["interest", "seeding", "quality"]:
            # Check Content for interest/seeding/quality keyword
            content = str(row.get("Content", "")) if "Content" in row.index and not pd.isna(row.get("Content")) else ""
            content_matched = self._find_matched_keywords(content, keywords)
            
            # If Content doesn't have keyword, return False
            if not content_matched:
                return (False, [])
            
            # Check Title for project keyword
            title = str(row.get("Title", "")) if "Title" in row.index and not pd.isna(row.get("Title")) else ""
            title_matched = self._find_matched_keywords(title, self.project_keywords)
            
            # Both conditions must be met
            if title_matched:
               
                all_matched = [f"Content: {kw}" for kw in content_matched] + [f"Title: {kw}" for kw in title_matched]
                return (True, all_matched)
            
            # Content has keyword but Title doesn't have project keyword
            return (False, [])
        
        # For other cases (topic or project keyword), use normal logic
        text = self._get_text_for_keyword_type(row, keyword_type)
        matched = self._find_matched_keywords(text, keywords)
        
        return (len(matched) > 0, matched)
    
    async def _call_llm(self, prompt: str, session: aiohttp.ClientSession) -> bool:
        """
        Generic LLM call function
        
        Parameters:
        -----------
        prompt : str
            Formatted prompt to send to LLM
        session : aiohttp.ClientSession
            Async HTTP session
        
        Returns:
        --------
        bool
            True if LLM returns "Yes", False otherwise
        """
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": False
            }
            
            async with session.post(
                f"{self.llm_api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result["choices"][0]["message"]["content"].strip().lower()
                    return "yes" in answer
                else:
                    return False
                
        except Exception:
            return False
    
    async def _detect_name_tags_llm(self, content: str, session: aiohttp.ClientSession) -> bool:
        """
        Use LLM to detect if content contains name tags or mentions people
        
        Parameters:
        -----------
        content : str
            Content text to analyze
        session : aiohttp.ClientSession
            Async HTTP session
        
        Returns:
        --------
        bool
            True if content contains name tags/mentions, False otherwise
        """

        if not content or pd.isna(content):
            return False
        
        # Load and format prompt
        prompt = load_prompt("name_tag_detection", comment=content)

        if not prompt:
            return False
        
        return await self._call_llm(prompt, session)
    
    async def _detect_content_relevance_llm(self, title: str, content: str, session: aiohttp.ClientSession) -> bool:
        """
        Use LLM to detect if content is relevant to title
        
        Parameters:
        -----------
        title : str
            Title text
        content : str
            Content text
        session : aiohttp.ClientSession
            Async HTTP session
        
        Returns:
        --------
        bool
            True if content is relevant to title, False otherwise
        """
        # Load and format prompt
        prompt = load_prompt("content_relevance", project=self.project_name, content=title, comment=content)
        if not prompt:
            return False
        
        return await self._call_llm(prompt, session)
    
    def _detect_row_sync(self, row: pd.Series) -> Tuple[str, bool, str, str]:
        """
        Detect a single row by applying rules 1-8 (synchronous rules).
        Returns (result, needs_llm, rule_name, explanation) where:
        - result: "Yes" or "No" or None (if needs LLM)
        - needs_llm: True if row needs LLM detection
        - rule_name: Name of the rule that matched
        - explanation: Detailed explanation with matched keywords
        
        Parameters:
        -----------
        row : pd.Series
            Single row from dataframe
        
        Returns:
        --------
        Tuple[str, bool, str, str]
            (result, needs_llm, rule_name, explanation)
        """
        # Rule 1: Sentiment is Negative or Positive
        if "Sentiment" in row.index:
            sentiment = row["Sentiment"]
            sentiment_list = ["negative", "positive"]
            
            if sentiment.strip().lower() in sentiment_list:
                explanation = f"Sentiment is {sentiment}"
                return ("Yes", False, "Rule 1: Sentiment", explanation)
        
        # Rule 2: Labels1 is "Trò chơi nhỏ - trực tuyển"
        if "Labels1" in row.index:
            labels1 = str(row["Labels1"]).strip()
            if "trò chơi nhỏ" in labels1.lower().strip():
                explanation = f"Labels1 is '{labels1}'"
                return ("Yes", False, "Rule 2: Labels1", explanation)
        
        # Rule 3: Channel is E-commerce or News
        if "Channel" in row.index:
            channel = row["Channel"]
            if channel in ["E-commerce", "News"]:
                explanation = f"Channel is {channel}"
                return ("Yes", False, "Rule 3: Channel", explanation)
        
        # Rule 4: Content is empty after removing whitespace, emojis, and links (only for comments)
        is_comment = False
        if "Type" in row.index:
            row_type = str(row["Type"]).lower()
            is_comment = "comment" in row_type
        
        if is_comment and self._is_content_empty(row, is_comment=True):
            explanation = "Content is empty after cleaning (only whitespace/emojis/links)"
            return ("No", False, "Rule 4: Empty Content", explanation)
        
        # Rule 5: Text contains project keywords
        if self.project_keywords:
            has_match, matched = self._contains_keywords(row, self.project_keywords, keyword_type="project")
            if has_match:
                matched_str = ", ".join(matched[:5])  # Limit to 5 keywords
                if len(matched) > 5:
                    matched_str += f" (and {len(matched)-5} more)"
                explanation = f"Matched project keywords: {matched_str}"
                return ("Yes", False, "Rule 5: Project Keywords", explanation)
        
        # Rule 6: Text contains interest keywords (inbox, ib, etc.)
        has_match, matched = self._contains_keywords(row, self.interest_keywords, keyword_type="interest")
        if has_match:
            matched_str = ", ".join(matched[:5])
            if len(matched) > 5:
                matched_str += f" (and {len(matched)-5} more)"
            explanation = f"Matched interest keywords: {matched_str}"
            return ("Yes", False, "Rule 6: Interest Keywords", explanation)
        
        # Rule 7: Text contains seeding keywords
        has_match, matched = self._contains_keywords(row, self.seeding_keywords, keyword_type="seeding")
        if has_match:
            matched_str = ", ".join(matched[:5])
            if len(matched) > 5:
                matched_str += f" (and {len(matched)-5} more)"
            explanation = f"Matched seeding keywords: {matched_str}"
            return ("Yes", False, "Rule 7: Seeding Keywords", explanation)
        
        # Rule 8: Text contains product quality keywords
        has_match, matched = self._contains_keywords(row, self.quality_keywords, keyword_type="quality")
        if has_match:
            matched_str = ", ".join(matched[:5])
            if len(matched) > 5:
                matched_str += f" (and {len(matched)-5} more)"
            explanation = f"Matched quality keywords: {matched_str}"
            return ("Yes", False, "Rule 8: Quality Keywords", explanation)

        return (None, True, "", "")
    
    async def _detect_rows_with_llm(self, rows_data: List[Tuple]) -> List[Tuple]:
        """
        Detect multiple rows using LLM in parallel
        Checks both Rule 9 (name tags) and Rule 10 (content relevance)
        
        Parameters:
        -----------
        rows_data : List[Tuple]
            List of (index, row, title, content) tuples
        
        Returns:
        --------
        List[Tuple]
            List of (index, result, rule_name, explanation) tuples
        """

        async with aiohttp.ClientSession() as session:
            results = []
            
            for idx, row, title, content in rows_data:
                # Rule 9: Check if Content has name tags AND Title has project keyword
                
                if content.strip():
                    name_tag_detected = await self._detect_name_tags_llm(content, session)
               
                    if name_tag_detected:
                        # Check if Title contains project keyword
                        title_matched = self._find_matched_keywords(title, self.project_keywords)
                        
                        if title_matched:
                            matched_str = ", ".join(title_matched[:3])
                            explanation = f"Content has name tags, Title matched: {matched_str}"
                            results.append((idx, "Yes", "Rule 9: Name Tags", explanation))
                            continue
                    
                    # Rule 10: Check if Content is relevant to Title
                    relevance_detected = await self._detect_content_relevance_llm(title, content, session)
                    
                    if relevance_detected:
                        results.append((idx, "Yes", "Rule 10: LLM Detection", "LLM detected Content is relevant to Title"))
                    else:
                        results.append((idx, "No", "No Match", "No rule matched"))
                else:
                    results.append((idx, "No", "No Match", "No rule matched"))
                   
            return results
       
    
    def detect(self, df: pd.DataFrame, use_llm: bool = False, batch_size: int = 10) -> pd.DataFrame:
        """
        Main detection method that applies all rules in priority order
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns: Title, Content, Description, Sentiment, Channel
        use_llm : bool
            Whether to use LLM for remaining rows (default: False)
        batch_size : int
            Number of rows to process in parallel for LLM detection (default: 10)
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with new columns "Yes/No", "Rule", and "Explanation"
        """
        df = df.copy()
        
        # Initialize columns as empty string
        df["Yes/No"] = ""
        df["Rule"] = ""
        df["Explanation"] = ""
        
        # Track statistics for each rule
        rule_stats = {
            "Rule 1: Sentiment": 0,
            "Rule 2: Labels1": 0,
            "Rule 3: Channel": 0,
            "Rule 4: Empty Content": 0,
            "Rule 5: Project Keywords": 0,
            "Rule 6: Interest Keywords": 0,
            "Rule 7: Seeding Keywords": 0,
            "Rule 8: Quality Keywords": 0,
            "Rule 9: Name Tags": 0,
            "Rule 10: LLM Detection": 0,
            "No Match": 0
        }
        
        # First pass: Apply synchronous rules (1-7)
        rows_needing_llm = []
        
        for idx, row in df.iterrows():
            result, needs_llm, rule_name, explanation = self._detect_row_sync(row)
            
            if result is not None:
                df.at[idx, "Yes/No"] = result
                df.at[idx, "Rule"] = rule_name
                df.at[idx, "Explanation"] = explanation
                
                # Track statistics
                if rule_name:
                    rule_stats[rule_name] += 1
            else:
                title = str(row.get("Title", "")) if "Title" in row.index and not pd.isna(row.get("Title")) else ""
                content = str(row.get("Content", "")) if "Content" in row.index and not pd.isna(row.get("Content")) else ""
                rows_needing_llm.append((idx, row, title, content))
        
        # Second pass: Apply LLM detection if enabled
        if use_llm and rows_needing_llm:
            print(f"\nProcessing {len(rows_needing_llm)} rows with LLM in batches of {batch_size}...")
            
            # Process in batches
            for i in range(0, len(rows_needing_llm), batch_size):
                batch = rows_needing_llm[i:i+batch_size]
                print(f"  Batch {i//batch_size + 1}/{(len(rows_needing_llm)-1)//batch_size + 1}...")
                
                # Run async detection for batch
                try:
                    # Get or create event loop
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    batch_results = loop.run_until_complete(self._detect_rows_with_llm(batch))
                    
                    # Check if batch_results is valid
                    if batch_results is None:
                        print(f"  Warning: Batch {i//batch_size + 1} returned None, marking as No Match")
                        batch_results = [(idx, "No", "No Match", "LLM call failed") for idx, _, _, _ in batch]
                    
                    # Debug: Print batch results count
                    print(f"  Batch returned {len(batch_results)} results")
                    
                    # Update dataframe with results
                    for idx, result, rule_name, explanation in batch_results:
                        df.at[idx, "Yes/No"] = result
                        df.at[idx, "Rule"] = rule_name
                        df.at[idx, "Explanation"] = explanation
                        rule_stats[rule_name] += 1
                        
                except Exception as e:
                    print(f"  Error processing batch {i//batch_size + 1}: {e}")
                    # Mark batch as failed
                    for idx, _, _, _ in batch:
                        df.at[idx, "Yes/No"] = "No"
                        df.at[idx, "Rule"] = "No Match"
                        df.at[idx, "Explanation"] = f"LLM error: {str(e)}"
                        rule_stats["No Match"] += 1
        else:
            if not use_llm:
                print(f"\nLLM detection disabled. Marking {len(rows_needing_llm)} remaining rows as 'No'")
            # Mark remaining rows as "No"
            for idx, _, _, _ in rows_needing_llm:
                df.at[idx, "Yes/No"] = "No"
                df.at[idx, "Rule"] = "No Match"
                df.at[idx, "Explanation"] = "No rule matched"
                rule_stats["No Match"] += 1
        
        # Print statistics
        print("\nRule Statistics:")
        for rule_name, count in rule_stats.items():
            if count > 0:
                print(f"  {rule_name}: {count} rows")
        
        # Print summary
        yes_count = (df["Yes/No"] == "Yes").sum()
        no_count = (df["Yes/No"] == "No").sum()
        print(f"\nDetection Summary:")
        print(f"  Yes: {yes_count} ({yes_count/len(df)*100:.1f}%)")
        print(f"  No: {no_count} ({no_count/len(df)*100:.1f}%)")
        columns = df.columns.to_list()
        df = df[columns[:-2]]
        df = sanitize_excel_values(df)
        return df


def detect_relevant(df: pd.DataFrame, project_name: str = "Vinamilk", use_llm: bool = False, batch_size: int = 10) -> pd.DataFrame:
    """
    Convenience function to detect relevant rows in dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    project_name : str
        Project name for keyword lookup (default: "Vinamilk")
    use_llm : bool
        Whether to use LLM for remaining rows (default: False)
    batch_size : int
        Number of rows to process in parallel for LLM detection (default: 10)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with "Yes/No" column added
    """
    detector = Detector(project_name=project_name)
    return detector.detect(df, use_llm=use_llm, batch_size=batch_size)
