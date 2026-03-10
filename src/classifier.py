import re
import json
import numpy as np
import os
import unicodedata
from typing import Dict, List, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.project_name = project_name
        
        metadata = self.rules_data.get("metadata", {})
        self.default_labels = metadata.get("default_labels", {})
        self.confidence_threshold = metadata.get("confidence_threshold", 0.6)
        
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
        
        for label_type, sources in self.source_mapping[self.project_name].items():
            if site_name in sources:
                return label_type
        
        return None
    
    def _check_mini_game(self, title: str, site_name: str, buzz_type: str):
        """Check for mini game content"""
        kws = "mini game"
        fanpage = ["shopee"]
        normalized_title = self._normalize_text(title)
        if site_name.lower() in fanpage and (kws in normalized_title or kws.replace(" ", "") in normalized_title):
            if "group" in buzz_type.lower():
                return "SELLER", "MINIGAME GROUP", 0.95
            elif "page" in buzz_type.lower():
                return "BUYER", "MINIGAME PAGE", 0.95 
            
        return None, None, 0

    def _check_keyword(self, text: str):
        """Check for keyword matching"""
        project_kws = self.kws_data.get(self.project_name)
        if not project_kws:
            return None, None, 0
        
        normalized_text = self._normalize_text(text)
        
        for label, kws in project_kws.items():
            for kw in kws:
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
    
    def llm_classification(self, buzz_type: str, content: str, site_name: str = "", author: str = "") -> tuple:
        """Use LLM to classify when rule-based fails"""
        project_rules = self.get_project_rules()
        if not project_rules:
            return "Unknown", 0.0
        
        label_definitions = []
        for category in project_rules.get("label_categories", []):
            label_type = category["label_type"]
            description = category["description"]
            rules = category.get("rules", "")
            examples = category.get("examples", [])
            
            label_def = f"{label_type}: {description}"
            if rules:
                label_def += f" | {rules}"
            if examples:
                label_def += f" | Examples: {'; '.join(examples[:2])}"
            
            label_definitions.append(label_def)
        
        context_info = ""
        if site_name or author:
            context_info = f"\n\nSOURCE CONTEXT:\n"
            if site_name:
                context_info += f"- SiteName: {site_name}\n"
            if author:
                context_info += f"- Author: {author}\n"
        
        additional_instructions = ""
        if self.project_name == "ShopeeFood":
            additional_instructions = """

** CRITICAL CLASSIFICATION RULES **

1. MERCHANT = Restaurant/shop OWNER posting about THEIR OWN business
   ✓ Uses: "quán mình", "shop tôi", "chúc quán mình"
   ✓ Posted from restaurant's official page
   ✗ NOT food reviews or customer recommendations
   
2. USER = Customers, food bloggers, reviewers, general public
   ✓ Food reviews with restaurant details, prices, addresses
   ✓ Personal experiences ordering/receiving food
   ✓ Sharing food experiences, recommendations
   ✓ Posts about ShopeeFood campaigns from customer perspective
   ✗ NOT from restaurant owner perspective
   
3. SHIPPER = Delivery drivers posting about their work
   ✓ From driver's perspective about delivery work
   ✓ Talking about income, orders, app issues
   ✗ NOT just mentioning delivery in passing
   
4. BRAND = Official ShopeeFood marketing only
   ✓ From official ShopeeFood VN channels
   ✗ NOT from community pages or user posts

DEFAULT: When unsure, classify as USER (not Merchant)"""
        
        instruction = """- Use <title>, <content>, and <description> to determine the author's perspective or intent.""" \
            if buzz_type == "topic" else \
        """- Use <topic> to get more context, <comment> to determine the commenter's perspective."""
        text_format = "(<title> | <content> | <description>)" if buzz_type == "topic" else "(<topic> | <comment>)"
        
        prompt = f"""You are a text classification expert.

Given text in format {text_format}. Your task is to assign EXACTLY ONE label to the given text.

*** LABELS ***
{chr(10).join([f"{i+1}. {ld}" for i, ld in enumerate(label_definitions)])}

{context_info}

*** INSTRUCTIONS ***

- Select the SINGLE most appropriate label.
- Focus on the AUTHOR'S perspective (the person writing the text).
{instruction}

{additional_instructions}
*** CONFIDENCE SCORE ***
Give a confidence score from 0 to 100:

90-100: Very clear match  
70-89: Strong match  
40-69: Possible match  
0-39: Weak or uncertain match

*** OUTPUT FORMAT ***
- ONLY return: <label>|<confidence>.
- DO NOT include explain.
- DO NOT include any characters.
"""
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            response = requests.post(
                f"{self.llm_api_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Text need to classify:\n{content}"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 20,
                    "stop": ["<think>", "\n"],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result["choices"][0]["message"]["content"].strip()
                
                if "<think>" in raw_response:
                    if "</think>" in raw_response:
                        raw_response = raw_response.split("</think>")[-1].strip()
                    else:
                        for valid_label in [cat["label_type"] for cat in project_rules.get("label_categories", [])]:
                            if valid_label in raw_response:
                                raw_response = valid_label
                                break
                
                label = raw_response
                confidence = 0.5
                
                if "|" in raw_response:
                    parts = raw_response.split("|")
                    label = parts[0].strip()
                    try:
                        confidence = float(parts[1].strip()) / 100.0
                    except (ValueError, IndexError):
                        confidence = 0.5
                else:
                    label = label.strip('"\'.,;: ')
                    label = label.split()[0] if label.split() else label
                
                valid_labels = [cat["label_type"] for cat in project_rules.get("label_categories", [])]
                
                if label in valid_labels:
                    return label, confidence
                
                label_lower = label.lower()
                for valid_label in valid_labels:
                    if valid_label.lower() == label_lower:
                        return valid_label, confidence
                    if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                        return valid_label, max(0.3, confidence - 0.2)
                
                print(f"LLM returned invalid label '{label}', using default label")
                return self.get_default_label(), 0.2
            else:
                print(f"LLM API error: {response.status_code}")
                return self.get_default_label(), 0.0
                
        except Exception as e:
            print(f"LLM classification error: {e}")
            return self.get_default_label(), 0.0
    
    def classify(self, data: Dict) -> tuple:
        """Main classification method - returns (label, method, confidence)"""
        site_name = data.get("siteName", "")
        author = data.get("author", "")
        buzz_type = data.get("type", "")
        title = data.get("title", "")
        text = self._extract_text(data)

        label, method, confidence = self._check_brand_indicators(site_name, buzz_type, author)
        if label and confidence > self.confidence_threshold:
            return label, method, confidence

        label, method, confidence = self._check_mini_game(title, site_name, buzz_type)
        if label and confidence > self.confidence_threshold:
            return label, method, confidence

        label, method, confidence = self._check_keyword(text)
        if label and confidence > self.confidence_threshold:
            return label, method, confidence

        if site_name:
            label = self._check_sitename(site_name)
            if label:
                return label, "SITE NAME", 0.95
              
        if not text:
            return self.get_default_label(), "NoText", 0.0
            
        label = self.rule_based_classification(text)
        if label is not None:
            return label, "RuleBased", 0.8
        
        label, confidence = self.llm_classification(buzz_type, text, site_name, author)
        if label is not None and confidence >= self.confidence_threshold:
            return label, "LLM", confidence
        
        label, confidence = self.model_based_classification(text)
        if confidence < self.confidence_threshold:
            default_label = self.get_default_label()
            return default_label, "LowConfidence", confidence
        
        return label, "PretrainedModel", confidence


def classify_category(df: pd.DataFrame, project_name: str, max_workers: int = 10):
    """Classify dataframe rows using multi-threading"""
    required_columns = ['Id', 'Topic', 'Title', 'Content', 'Description', 'Type', 'SiteName', 'Author']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Cảnh báo: Các cột sau không tồn tại trong file: {missing_columns}")
        print(f"Các cột hiện có: {list(df.columns)}")

    print("Đang khởi tạo classifier...")
    classifier = LabelClassifier(project_name=project_name)
    classifier.load_pretrained_model()

    def classify_row(row):
        idx = row.Index
        data = {
            "title": str(getattr(row, "Title", "")) if pd.notna(getattr(row, "Title", "")) else "",
            "content": str(getattr(row, "Content", "")) if pd.notna(getattr(row, "Content", "")) else "",
            "description": str(getattr(row, "Description", "")) if pd.notna(getattr(row, "Description", "")) else "",
            "type": str(getattr(row, "Type", "")) if pd.notna(getattr(row, "Type", "")) else "",
            "siteName": str(getattr(row, "SiteName", "")) if pd.notna(getattr(row, "SiteName", "")) else "",
            "author": str(getattr(row, "Author", "")) if pd.notna(getattr(row, "Author", "")) else "",
        }
        label, method, confidence = classifier.classify(data)
        return idx, label, method, confidence

    print(f"Đang phân loại với {max_workers} workers...")
    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(classify_row, row): row.Index
            for row in df.itertuples()
        }

        with tqdm(total=len(df), desc="Processing") as pbar:
            for future in as_completed(futures):
                try:
                    idx, label, method, confidence = future.result()
                    results[idx] = (label, method, confidence)
                except Exception as e:
                    idx = futures[future]
                    print(f"\nError processing row {idx}: {e}")
                    results[idx] = ("Unknown", "Error", 0.0)
                pbar.update(1)

    labels, methods, confidences = zip(*results)
    df['Label'] = labels
    # df['Method'] = methods
    # df['Confidence'] = confidences
    df = sanitize_excel_values(df)

    return df
