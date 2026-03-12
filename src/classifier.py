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
from src.utils import sanitize_excel_values, clean_lower_text

load_dotenv()

LLM_RULES_PATH = os.path.join(PROJECT_DIR, "src/rules/llm-rules.json")
KEYWORD_PATH = os.path.join(PROJECT_DIR, "src/rules/label-kws.json")
SOURCE_MAPPING_PATH = os.path.join(PROJECT_DIR, "src/rules/source-mapping.json")

class LabelClassifier:
    def __init__(self, project_name: str = "ShopeeFood", llm_rules_path: str = LLM_RULES_PATH, kws_path: str = KEYWORD_PATH, source_mapping_path: str = SOURCE_MAPPING_PATH):
        """Initialize classifier with rules from JSON file"""
        with open(llm_rules_path, 'r', encoding='utf-8') as f:
            self.rules_data = json.load(f)
        with open(kws_path, 'r', encoding='utf-8') as f:
            self.kws_data = json.load(f)
        
        try:
            with open(source_mapping_path, 'r', encoding='utf-8') as f:
                self.source_mapping = json.load(f)
        except FileNotFoundError:
            self.source_mapping = {}
        
        self.llm_api_url = os.getenv("LLM_API_URL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_model = os.getenv("LLM_MODEL", "")
        self.classify_api_endpoint = os.getenv("CLASSIFY_API_ENDPOINT", "")
        self.project_name = project_name
        
        metadata = self.rules_data.get("metadata", {})
        self.default_labels = metadata.get("default_labels", {})
        self.confidence_threshold = metadata.get("confidence_threshold", 0.6)
        self.valid_labels = []

        self.model = None
        self.label_encoder = None

    def _normalize_text(self, text: str) -> str:
        """Normalize text: convert Unicode variants to normal, remove URLs, keep Vietnamese tones, remove punctuation, lowercase"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Normalize Unicode: convert bold, italic, and other variants to normal characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove punctuation but keep Vietnamese characters with tones
        text = re.sub(r'[!?.,;:\"\'\(\)\[\]\{\}\/\\\|@#\$%\^\&\*\+=\-_~`<>]', ' ', text)
        
        # Normalize whitespace and convert to lowercase
        return re.sub(r"\s+", " ", text).lower().strip()
    
    def _extract_text(self, data: Dict) -> str:
        """Extract and merge text from title, content, description"""
        text_parts = []
        
        is_topic = "topic" in data.get("type", "").lower()
        title = data.get("title", "").strip()
        content = data.get("content", "").strip()
        description = data.get("description", "").strip()
        
        if is_topic:
            if title:
                text_parts.append(title)
            if content and content != title:
                text_parts.append(content)
            if description and description not in [title, content]:
                text_parts.append(description)
        else:
            if title:
                text_parts.append(title)
            if content and content != title:
                text_parts.append(content)

        return " | ".join(text_parts)
    
    def get_default_label(self) -> str:
        """Get default label for project when confidence is low"""
        return self.default_labels.get(self.project_name, "Unknown")
    
    def get_project_rules(self) -> Optional[Dict]:
        """Get rules for specific project"""
        for project in self.rules_data.get("projects", []):
            if project["project_name"] == self.project_name:
                return project
        return None
    
    def _check_sitename(self, site_name: str) -> Optional[str]:
        """Check if SiteName matches predefined source mapping"""
        if not site_name or self.project_name not in self.source_mapping:
            return None
        site_name = self._normalize_text(site_name)
        for label_type, sources in self.source_mapping[self.project_name].items():
            for source in sources:
                source = self._normalize_text(source)
                if source in site_name or source.replace(" ", "") in site_name:
                    return label_type
        
        return None
    
    def _check_keyword(self, data: Dict):
        """Check for keyword matching with conditions"""
        project_kws = self.kws_data.get(self.project_name)
    
        if not project_kws:
            return None, None, 0
        
        # Extract text and metadata
        text = self._extract_text(data)
        normalized_text = self._normalize_text(text)
        
        buzz_type = data.get("type", "").lower()
        site_name = data.get("siteName", "").lower()
        
        # Check each label category
        for label, rules in project_kws.items():
            for rule in rules:
                condition = rule.get("condition", {})
                keywords = rule.get("keywords", [])
                
                # Check if conditions are met
                conditions_met = True
                
                # Check type condition
                if "type" in condition:
                    required_type = condition["type"].lower()
                    if required_type not in buzz_type:
                        conditions_met = False
                
                # Check siteName condition
                if "siteName" in condition:
                    required_site = condition["siteName"].lower()
                    # Empty string means any siteName is ok
                    if required_site and required_site != site_name:
                        conditions_met = False
                
                # If conditions are met, check keywords
                if conditions_met:
                    for kw in keywords:
                        normalized_kw = self._normalize_text(kw)
                        if normalized_kw in normalized_text:
                            return label, "KEYWORD MATCHING", 0.95
        
        return None, None, 0

    def _check_brand_indicators(self, site_name: str, buzz_type: str, author: str) -> bool:
        """Check if content is from official brand channels"""
        if "topic" in clean_lower_text(buzz_type):
            brand_indicators = [
                "ShopeeFood VN",
                "ShopeeFood Vietnam"
            ]
            
            site_name_lower = clean_lower_text(site_name)
            author_lower = clean_lower_text(author)
            
            for indicator in brand_indicators:
                indicator_lower = clean_lower_text(indicator)
                if (site_name_lower == indicator_lower or 
                    author_lower == indicator_lower or
                    site_name_lower.startswith(indicator_lower + " ") or
                    author_lower.startswith(indicator_lower + " ")):
                    return "Brand", "BrandIndicator", 1.0
        
        return None, None, 0
    

    def _get_valid_labels(self) -> list:
        
        project_rules = self.get_project_rules()
        for category in project_rules.get("label_categories", []):
            self.valid_labels.append(category.get("label_type", ""))
        return self.valid_labels


    def rule_based_classification(self, text: str) -> Optional[str]:
        """Apply rule-based classification using keywords"""
        project_rules = self.get_project_rules()
        if not project_rules:
            return None
        
        normalized_text = self._normalize_text(text)
        best_match = None
        max_keyword_count = 0
        
        for category in project_rules.get("label_categories", []):
            keywords = category.get("keywords", [])
            if not keywords:
                continue
            
            keyword_count = sum(1 for keyword in keywords if self._normalize_text(keyword) in normalized_text)
            
            if keyword_count > max_keyword_count:
                max_keyword_count = keyword_count
                best_match = category["label_type"]
        
        if max_keyword_count >= 2:
            return best_match
        
        return None
    
    def load_pretrained_model(self) -> Tuple[Optional[object], Optional[object]]:
        """Load pretrained model and label encoder for a project"""
        try:
            model, label_encoder = loader(self.project_name)
            self.model = model
            self.label_encoder = label_encoder
            print(f"Loaded pretrained model for project: {self.project_name}")
        except FileNotFoundError as e:
            print(f"No pretrained model found for project {self.project_name}: {e}")
        except Exception as e:
            print(f"Error loading pretrained model for {self.project_name}: {e}")

    def model_based_classification(self, text: str) -> Tuple[Optional[str], float]:
        """Use pretrained model for classification"""
        if self.model is None or self.label_encoder is None:
            return None, 0.0

        try:
            normalized_text = self._normalize_text(text)
            prediction = self.model.predict([normalized_text])[0]
            confidence = 0.0

            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function([normalized_text])
                if isinstance(scores, np.ndarray):
                    if scores.ndim == 1:
                        confidence = float(abs(scores[0]))
                    else:
                        confidence = float(scores[0].max())

            label = self.label_encoder.inverse_transform([prediction])[0]
            return label, confidence

        except Exception as e:
            print(f"Error during model prediction: {e}")
            return None, 0.0
    
    async def llm_classification(self, data: dict, session: aiohttp.ClientSession) -> tuple:
        """Use classification API endpoint with async aiohttp"""
        if not self.classify_api_endpoint:
            print("CLASSIFY_API_ENDPOINT not configured")
            return self.get_default_label(), 0.0
        
        try:
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "id": data.get("id", ""),
                "index": data.get("id", ""),
                "topic": data.get("topic", ""),
                "title": data.get("title", ""),
                "content": data.get("content", ""),
                "description": data.get("description", ""),
                "type": data.get("type", ""),
                "project": self.project_name.lower()
            }
            
            async with session.post(
                self.classify_api_endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if not result.get("success", False):
                        error_msg = result.get("error", "Unknown error")
                        print(f"Classification API returned error: {error_msg}")
                        return self.get_default_label(), 0.0
                    
                    labels = result.get("labels", [])
                    if not labels:
                        print("No labels returned from API")
                        return self.get_default_label(), 0.0
                    
                    # Find label with level 1
                    level1_label = None
                    for label_obj in labels:
                        if label_obj.get("level") == 1:
                            level1_label = label_obj.get("name", "")
                            break
                    
                    if not level1_label:
                        print("No level 1 label found in response")
                        return self.get_default_label(), 0.0
                    
                    # Default confidence for API classification
                    confidence = 0.7
                    
                    # Validate against project rules if available
                    project_rules = self.get_project_rules()
                    if project_rules:
                        label_lower = level1_label.lower()
                        for valid_label in self.valid_labels:
                            if valid_label.lower() == label_lower:
                                return valid_label, confidence
                            if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                                return valid_label, max(0.3, confidence - 0.2)
                    
                    return self.get_default_label(), confidence
                else:
                    print(f"Classification API error: {response.status}")
                    return self.get_default_label(), 0.0
                    
        except Exception as e:
            print(f"Classification API error: {e}")
            return self.get_default_label(), 0.0
    
    def classify_sync_rules(self, data: Dict) -> tuple:
        """
        Apply synchronous rules only (no LLM)
        Returns (label, method, confidence) or (None, None, None) if needs LLM
        """
        site_name = data.get("siteName", "")
        author = data.get("author", "")
        text = self._extract_text(data)

        # Rule 1: Brand Indicators
        label, method, confidence = self._check_brand_indicators(site_name, data.get("type", ""), author)
        if label and confidence > self.confidence_threshold:
            return label, method, confidence
        
        # Rule 2: SiteName Mapping
        if site_name:
            label = self._check_sitename(site_name)
            if label:
                return label, "SITE NAME", 0.95
        
        # Rule 3: Keyword Matching
        label, method, confidence = self._check_keyword(data)
        if label and confidence > self.confidence_threshold:
            return label, method, confidence

        # Rule 4: Empty text check
        if not text:
            return self.get_default_label(), "NoText", 0.0
        
        # Rule 5: Rule-based Classification
        label = self.rule_based_classification(text)
        if label is not None:
            return label, "RuleBased", 0.8
        
        # Rule 6: Model-based Classification (if available)
        # if self.model is not None:
        #     label, confidence = self.model_based_classification(text)
        #     if confidence >= self.confidence_threshold:
        #         return label, "PretrainedModel", confidence
        
        # Needs LLM
        return None, None, None
    
    async def classify_async(self, data: Dict, session: aiohttp.ClientSession) -> tuple:
        """Async classification method - returns (label, method, confidence)"""
        # Try sync rules first
        label, method, confidence = self.classify_sync_rules(data)
        if label is not None:
            return label, method, confidence
        
        # Need LLM classification
        label, confidence = await self.llm_classification(data, session)
        if label is not None and confidence >= self.confidence_threshold:
            return label, "LLM", confidence
        
        # Fallback to model with low confidence
        text = self._extract_text(data)
        label, confidence = self.model_based_classification(text)
        if label is not None:
            return label, "LowConfidence", confidence
        
        return self.get_default_label(), "LowConfidence", 0.0
    
    def classify(self, data: Dict) -> tuple:
        """Synchronous wrapper for classify_async (for backward compatibility)"""
        async def _classify():
            async with aiohttp.ClientSession() as session:
                return await self.classify_async(data, session)
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(_classify())


async def classify_batch_async(classifier: LabelClassifier, rows_data: List[Tuple], session: aiohttp.ClientSession) -> List[Tuple]:
    """Classify a batch of rows asynchronously"""
    tasks = []
    for idx, data in rows_data:
        task = classifier.classify_async(data, session)
        tasks.append((idx, task))
    
    results = []
    for idx, task in tasks:
        try:
            label, method, confidence = await task
            results.append((idx, label, method, confidence))
        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            results.append((idx, "Unknown", "Error", 0.0))
    
    return results


def classify_category(df: pd.DataFrame, project_name: str, batch_size: int = 10):
    """Classify dataframe rows - sync rules first, then batch LLM processing"""
    required_columns = ['Id', 'Topic', 'Title', 'Content', 'Description', 'Type', 'SiteName', 'Author']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Cảnh báo: Các cột sau không tồn tại trong file: {missing_columns}")
        print(f"Các cột hiện có: {list(df.columns)}")

    print("Đang khởi tạo classifier...")
    classifier = LabelClassifier(project_name=project_name)
    classifier.load_pretrained_model()
    classifier._get_valid_labels()
    # Prepare data
    rows_data = []
    for row in df.itertuples():
        idx = row.Index
        data = {
            "id": str(getattr(row, "Id", "")) if pd.notna(getattr(row, "Id", "")) else "",
            "topic": str(getattr(row, "Topic", "")) if pd.notna(getattr(row, "Topic", "")) else "",
            "title": str(getattr(row, "Title", "")) if pd.notna(getattr(row, "Title", "")) else "",
            "content": str(getattr(row, "Content", "")) if pd.notna(getattr(row, "Content", "")) else "",
            "description": str(getattr(row, "Description", "")) if pd.notna(getattr(row, "Description", "")) else "",
            "type": str(getattr(row, "Type", "")) if pd.notna(getattr(row, "Type", "")) else "",
            "siteName": str(getattr(row, "SiteName", "")) if pd.notna(getattr(row, "SiteName", "")) else "",
            "author": str(getattr(row, "Author", "")) if pd.notna(getattr(row, "Author", "")) else "",
        }
        rows_data.append((idx, data))

    results = [None] * len(df)
    rows_needing_llm = []

    # Phase 1: Apply sync rules to all rows
    print(f"Phase 1: Applying sync rules to {len(rows_data)} rows...")
    with tqdm(total=len(rows_data), desc="Sync Rules") as pbar:
        for idx, data in rows_data:
            label, method, confidence = classifier.classify_sync_rules(data)
            if label is not None:
                results[idx] = (label, method, confidence)
            else:
                rows_needing_llm.append((idx, data))
            pbar.update(1)

    # Phase 2: Batch process rows needing LLM
    if rows_needing_llm:
        print(f"\nPhase 2: Processing {len(rows_needing_llm)} rows with LLM (batch size {batch_size})...")
        
        async def process_llm_batches():
            async with aiohttp.ClientSession() as session:
                with tqdm(total=len(rows_needing_llm), desc="LLM Processing") as pbar:
                    for i in range(0, len(rows_needing_llm), batch_size):
                        batch = rows_needing_llm[i:i+batch_size]
                        batch_results = await classify_batch_async(classifier, batch, session)
                        
                        for idx, label, method, confidence in batch_results:
                            results[idx] = (label, method, confidence)
                        
                        pbar.update(len(batch))
        
        # Run async processing
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(process_llm_batches())
    else:
        print("\nNo rows need LLM processing!")

    labels, methods, confidences = zip(*results)
    df['Label'] = labels
    df['Method'] = methods
    df['Confidence'] = confidences
    df = sanitize_excel_values(df)

    return df
