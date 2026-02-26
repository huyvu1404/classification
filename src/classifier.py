import asyncio
from pyexpat import model
import unicodedata
import re
import json
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv
from src.cms import get_keywords, login_cms
from src.models import loader
from src.settings import PROJECT_DIR

RULES_PATH = os.path.join(PROJECT_DIR, "src/label-rules.json")
SOURCE_MAPPING_PATH = os.path.join(PROJECT_DIR, "src/source-mapping.json")

load_dotenv()


import asyncio
import re
import unicodedata
import pandas as pd
from typing import Dict, List


class KeywordDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.topic_keywords: Dict[int, List[str]] = {}

    # ===================== TEXT UTILS =====================

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).lower().strip()

    @staticmethod
    def merge_text(row: pd.Series) -> str:
        if "comment" in str(row.get("Type", "")).lower():
            return str(row.get("Content", ""))

        parts = []
        for col in ("Title", "Content", "Description"):
            val = str(row.get(col, "")).strip()
            if val and val not in parts:
                parts.append(val)

        return " ".join(parts)

    # ===================== KEYWORDS =====================

    async def load_keywords(self):
        tokens = await login_cms()

        for topic_id in self.df["TopicId"].unique():
            kws = await get_keywords(tokens, topic_id)
            self.topic_keywords[topic_id] = [
                self.normalize(kw) for kw in kws if kw
            ]

    # ===================== CLASSIFICATION =====================

    def _match_keywords(self, text: str, keywords: List[str]) -> bool:
        for kw in keywords:
            if any(kw_part in text.lower() for kw_part in kw.split(" ")):
                return True
        return False

    async def classify(self) -> pd.DataFrame:
        # merge + normalize text once
        self.df["_text"] = (
            self.df.apply(self.merge_text, axis=1)
                   .map(self.normalize)
        )

        self.df["Yes/No"] = "No"

        await self.load_keywords()

        for topic_id, keywords in self.topic_keywords.items():
            if not keywords:
                continue

            mask = self.df["TopicId"] == topic_id
            self.df.loc[mask, "Yes/No"] = self.df.loc[mask, "_text"].map(
                lambda text: "Yes" if self._match_keywords(text, keywords) else "No"
            )

        return self.df.drop(columns="_text")


def classify_buzz_revelent(df):
    detector = KeywordDetector(df)
    return asyncio.run(detector.classify())


class LabelClassifier:
    def __init__(self, rules_path: str = RULES_PATH, source_mapping_path: str = SOURCE_MAPPING_PATH):
        """Initialize classifier with rules from JSON file"""
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules_data = json.load(f)
        
        # Load source mapping for SiteName-based classification
        try:
            with open(source_mapping_path, 'r', encoding='utf-8') as f:
                self.source_mapping = json.load(f)
        except FileNotFoundError:
            self.source_mapping = {}
        
        self.llm_api_url = os.getenv("LLM_API_URL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_model = os.getenv("LLM_MODEL", "")
        
        # Load metadata settings
        metadata = self.rules_data.get("metadata", {})
        self.default_labels = metadata.get("default_labels", {})
        self.confidence_threshold = metadata.get("confidence_threshold", 0.6)
        
        # Cache for pretrained models
        self.model = None
        self.label_encoder = None
        
    def extract_text(self, data: Dict) -> str:
        """Extract and merge text from title, content, description"""
        text_parts = []
        
        # Check if it's a topic or comment
        is_topic = data.get("type", "").lower() in ["fbgrouptopic", "topic"]
        
        if is_topic:
            # For topics, extract title, content, description
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            description = data.get("description", "").strip()
            
            # Add non-empty unique parts
            if title:
                text_parts.append(title)
            if content and content != title:
                text_parts.append(content)
            if description and description not in [title, content]:
                text_parts.append(description)
        else:
            # For comments, just use content
            content = data.get("content", "").strip()
            if content:
                text_parts.append(content)
        
        # Merge and clean text
        merged_text = " ".join(text_parts)
        # Remove URLs
        merged_text = re.sub(r'https?://\S+', '', merged_text)
        # Remove hashtags (optional, keep if needed for classification)
        # merged_text = re.sub(r'#\w+', '', merged_text)
        # Remove extra whitespace
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()
        
        return merged_text
    
    def get_default_label(self, project_name: str) -> str:
        """Get default label for project when confidence is low"""
        return self.default_labels.get(project_name, "Unknown")
    
    def get_project_rules(self, project_name: str) -> Optional[Dict]:
        """Get rules for specific project"""
        for project in self.rules_data.get("projects", []):
            if project["project_name"] == project_name:
                return project
        return None

    
    def check_source_mapping(self, site_name: str, project_name: str) -> Optional[str]:
        """Check if SiteName matches predefined source mapping (highest priority)"""
        if not site_name or project_name not in self.source_mapping:
            return None
        
        # Check each label category for matching source
        for label_type, sources in self.source_mapping[project_name].items():
            if site_name in sources:
                return label_type
        
        return None
    
    def detect_merchant_ownership(self, text: str, site_name: str) -> bool:
        """Detect if content shows restaurant/shop ownership perspective"""
        text_lower = text.lower()
        
        # Strong ownership indicators
        ownership_keywords = [
            "quán mình",
            "shop mình", 
            "shop tôi",
            "quán tôi",
            "shop chúng tôi",
            "quán chúng tôi",
            "cửa hàng mình",
            "cửa hàng chúng tôi",
            "nhà hàng mình",
            "bên mình",
            "em làm",
            "nhà em",
            "quán em"
        ]
        
        for keyword in ownership_keywords:
            if keyword in text_lower:
                return True
        
        # Check if SiteName is a restaurant/shop name (not a group or personal page)
        # Restaurant pages usually don't have these patterns
        non_merchant_patterns = [
            "hà nội",
            "sài gòn", 
            "ăn gì",
            "ở đâu",
            "cộng đồng",
            "hội",
            "group",
            "foody"
        ]
        
        site_name_lower = site_name.lower()
        has_non_merchant_pattern = any(pattern in site_name_lower for pattern in non_merchant_patterns)
        
        # If no ownership keywords and site name suggests it's not a merchant, return False
        if has_non_merchant_pattern:
            return False
        
        return False
    
    def check_brand_indicators(self, site_name: str, author: str, project_name: str) -> bool:
        """Check if content is from official brand channels"""
        if project_name == "ShopeeFood":
            # Official ShopeeFood channels - ONLY the main brand pages
            brand_indicators = [
                "ShopeeFood VN",
                "ShopeeFood Vietnam"
            ]
            
            # Check SiteName or Author - must be exact match or very close
            site_name_lower = site_name.lower()
            author_lower = author.lower()
            
            for indicator in brand_indicators:
                indicator_lower = indicator.lower()
                # Exact match or starts with the indicator
                if (site_name_lower == indicator_lower or 
                    author_lower == indicator_lower or
                    site_name_lower.startswith(indicator_lower + " ") or
                    author_lower.startswith(indicator_lower + " ")):
                    return True
        
        return False
    
    def rule_based_classification(self, text: str, project_name: str) -> Optional[str]:
        """Apply rule-based classification using keywords"""
        project_rules = self.get_project_rules(project_name)
        if not project_rules:
            return None
        
        text_lower = text.lower()
        best_match = None
        max_keyword_count = 0
        
        for category in project_rules.get("label_categories", []):
            keywords = category.get("keywords", [])
            if not keywords:
                continue
            
            # Count matching keywords
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            
            # If we find keywords, track the best match
            if keyword_count > max_keyword_count:
                max_keyword_count = keyword_count
                best_match = category["label_type"]
        
        # Return label if we have at least 2 keyword matches (threshold can be adjusted)
        if max_keyword_count >= 2:
            return best_match
        
        return None
    
    def load_pretrained_model(self, project_name: str) -> Tuple[Optional[object], Optional[object]]:
        """
        Load pretrained model and label encoder for a project
        Returns (model, label_encoder) or (None, None) if not available
        """
        # Check cache first
        try:
            model, label_encoder = loader(project_name)
            print
            self.model = model
            self.label_encoder = label_encoder
            print(f"Loaded pretrained model for project: {project_name}")
           
        except FileNotFoundError as e:
            print(f"No pretrained model found for project {project_name}: {e}")

        except Exception as e:
            print(f"Error loading pretrained model for {project_name}: {e}")


    def model_based_classification(self, text: str) -> Tuple[Optional[str], float]:
        if self.model is None or self.label_encoder is None:
            return None, 0.0

        try:
            prediction = self.model.predict([text])[0]

            confidence = 0.0

            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function([text])

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

    
    def llm_classification(self, text: str, project_name: str, site_name: str = "", author: str = "") -> tuple:
        """Use LLM to classify when rule-based fails. Returns (label, confidence)"""
        project_rules = self.get_project_rules(project_name)
        if not project_rules:
            return "Unknown", 0.0
        
        # Build concise label definitions optimized for Qwen
        label_definitions = []
        for category in project_rules.get("label_categories", []):
            label_type = category["label_type"]
            description = category["description"]
            rules = category.get("rules", "")
            examples = category.get("examples", [])
            
            # Combine description and rules for clarity
            label_def = f"{label_type}: {description}"
            if rules:
                label_def += f" | {rules}"
            
            # Add examples if available
            if examples:
                label_def += f" | Examples: {'; '.join(examples[:2])}"
            
            label_definitions.append(label_def)
        
        # Add context about source if available
        context_info = ""
        if site_name or author:
            context_info = f"\n\nSOURCE CONTEXT:\n"
            if site_name:
                context_info += f"- SiteName: {site_name}\n"
            if author:
                context_info += f"- Author: {author}\n"
        
        # Enhanced prompt with better instructions for ShopeeFood
        additional_instructions = ""
        if project_name == "ShopeeFood":
            additional_instructions = """

CRITICAL CLASSIFICATION RULES:

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
        
        # Optimized prompt for Qwen - more structured and direct
        prompt = f"""Classify the following text into ONE label for project "{project_name}".

LABELS:
{chr(10).join([f"{i+1}. {ld}" for i, ld in enumerate(label_definitions)])}
{context_info}
{additional_instructions}

TEXT:
{text}

INSTRUCTIONS:
- Read the text carefully and identify the AUTHOR'S ROLE/PERSPECTIVE
- Match content to the most appropriate label based on who is posting and their perspective
- Return ONLY the label name followed by confidence score (0-100)
- Format: <label>|<confidence>
- Example: User|85 or Merchant|70

ANSWER (format: label|confidence):"""
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add Authorization header if API key is provided
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            response = requests.post(
                f"{self.llm_api_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are a text classifier. Return ONLY the label name, nothing else. No thinking, no explanation."},
                        {"role": "user", "content": prompt}
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
                
                # Remove thinking tags if present (Qwen3 issue)
                if "<think>" in raw_response:
                    # Extract only the answer after thinking
                    if "</think>" in raw_response:
                        raw_response = raw_response.split("</think>")[-1].strip()
                    else:
                        # If thinking tag not closed, skip to first valid label word
                        for valid_label in [cat["label_type"] for cat in project_rules.get("label_categories", [])]:
                            if valid_label in raw_response:
                                raw_response = valid_label
                                break
                
                # Parse label and confidence
                label = raw_response
                confidence = 0.5  # Default medium confidence
                
                # Check if response contains confidence score (format: label|confidence)
                if "|" in raw_response:
                    parts = raw_response.split("|")
                    label = parts[0].strip()
                    try:
                        confidence = float(parts[1].strip()) / 100.0  # Convert to 0-1 range
                    except (ValueError, IndexError):
                        confidence = 0.5
                else:
                    # No confidence provided, estimate based on response clarity
                    # Clean up response - remove quotes, periods, extra text
                    label = label.strip('"\'.,;: ')
                    # Take only first word if multiple words
                    label = label.split()[0] if label.split() else label
                    
                    # If label matches exactly, higher confidence
                    valid_labels = [cat["label_type"] for cat in project_rules.get("label_categories", [])]
                    if label in valid_labels:
                        confidence = 0.7
                    else:
                        confidence = 0.4
                
                # Validate label exists in our rules
                valid_labels = [cat["label_type"] for cat in project_rules.get("label_categories", [])]
                if label in valid_labels:
                    return label, confidence
                
                # Try to find partial match (case-insensitive)
                label_lower = label.lower()
                for valid_label in valid_labels:
                    if valid_label.lower() == label_lower:
                        return valid_label, confidence
                    if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                        return valid_label, max(0.3, confidence - 0.2)  # Lower confidence for partial match
                
                # If no match found, return default label with low confidence
                print(f"LLM returned invalid label '{label}', using default label")
                return self.get_default_label(project_name), 0.2
            else:
                print(f"LLM API error: {response.status_code}")
                return self.get_default_label(project_name), 0.0
                
        except Exception as e:
            print(f"LLM classification error: {e}")
            return self.get_default_label(project_name), 0.0
    
    def classify(self, data: Dict) -> tuple:
        """Main classification method - returns (label, method, confidence)"""
        # Extract project/topic
        project_name = data.get("topic", "")
        
        # Extract SiteName and Author for source mapping check
        site_name = data.get("siteName", "")
        author = data.get("author", "")
        
        # Extract and merge text early for ownership detection
        text = self.extract_text(data)
        if project_name == "ShopeeFood":
        # PRIORITY 1: Check if it's official Brand content (100% confidence)
            if self.check_brand_indicators(site_name, author, project_name):
                return "Brand", "BrandIndicator", 1.0
            
            # PRIORITY 2: Check source mapping (Shipper/Merchant groups) (95% confidence)
            if site_name:
                label = self.check_source_mapping(site_name, project_name)
                if label:
                    return label, "SourceMapping", 0.95
            
            if not text:
                return self.get_default_label(project_name), "NoText", 0.0
            
            # PRIORITY 3: Check for strong merchant ownership indicators (90% confidence)
            if self.detect_merchant_ownership(text, site_name):
                return "Merchant", "OwnershipDetection", 0.9
            
        # PRIORITY 4: Try rule-based classification (80% confidence)
        label = self.rule_based_classification(text, project_name)
        
        if label is not None:
            return label, "RuleBased", 0.8
        
        # PRIORITY 5: Try pretrained model classification (confidence from model)
        label, confidence = self.model_based_classification(text)
        
        if label is not None and confidence >= self.confidence_threshold:
            return label, "PretrainedModel", confidence
        
        # PRIORITY 6: If model fails or low confidence, use LLM with confidence score
        # label, confidence = self.llm_classification(text, project_name, site_name, author)
        
        # If confidence is below threshold, use default label
        if confidence < self.confidence_threshold:
            default_label = self.get_default_label(project_name)
            return default_label, "LowConfidence", confidence
        
        return label, "LLM", confidence


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

def classify_buzz_category(df, max_workers: int = 10):

    required_columns = ['Id', 'Topic', 'Title', 'Content', 'Description', 'Type', 'SiteName']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Cảnh báo: Các cột sau không tồn tại trong file: {missing_columns}")
        print(f"Các cột hiện có: {list(df.columns)}")

    # Khởi tạo classifier
    print("Đang khởi tạo classifier...")
    classifier = LabelClassifier()
    classifier.load_pretrained_model(df["Topic"].iloc[0])  # Load model for the first topic (assuming all rows have same topic)
    # Hàm phân loại một bản ghi
    def classify_row(idx_row):
        idx, row = idx_row
        data = {
            "id": str(row.get('Id', '')) if pd.notna(row.get('Id')) else '',
            "topic": str(row.get('Topic', '')) if pd.notna(row.get('Topic')) else '',
            "title": str(row.get('Title', '')) if pd.notna(row.get('Title')) else '',
            "content": str(row.get('Content', '')) if pd.notna(row.get('Content')) else '',
            "description": str(row.get('Description', '')) if pd.notna(row.get('Description')) else '',
            "type": str(row.get('Type', '')) if pd.notna(row.get('Type')) else '',
            "siteName": str(row.get('SiteName', '')) if pd.notna(row.get('SiteName')) else '',
            "author": str(row.get('Author', '')) if pd.notna(row.get('Author')) else '',
        }
        
        label, method, confidence = classifier.classify(data)
        return idx, label, method, confidence
    
    # Phân loại song song
    print(f"Đang phân loại với {max_workers} workers...")
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(classify_row, (idx, row)): idx 
                  for idx, row in df.iterrows()}
        
        # Process completed tasks with progress bar
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
    
    # Sắp xếp kết quả theo index và thêm vào DataFrame
    labels = []
    methods = []
    confidences = []
    
    for idx in df.index:
        if idx in results:
            label, method, confidence = results[idx]
            labels.append(label)
            methods.append(method)
            confidences.append(confidence)
        else:
            labels.append("Unknown")
            methods.append("Error")
            confidences.append(0.0)
    
    # Thêm cột Label, Method và Confidence vào DataFrame
    df['Label'] = labels
   

    # Tạo tên file output nếu không được cung cấp

    # # Thống kê kết quả
    # print("\n=== THỐNG KÊ KẾT QUẢ ===")
    # print(f"Tổng số bản ghi: {len(df)}")
    
    # print("\n=== PHƯƠNG PHÁP PHÂN LOẠI ===")
    # method_counts = df['Method'].value_counts()
    # for method, count in method_counts.items():
    #     percentage = (count / len(df)) * 100
    #     print(f"  {method}: {count} ({percentage:.1f}%)")
    
    # print("\n=== CONFIDENCE STATISTICS ===")
    # print(f"Average Confidence: {df['Confidence'].mean():.2%}")
    # print(f"Median Confidence: {df['Confidence'].median():.2%}")
    # print(f"Min Confidence: {df['Confidence'].min():.2%}")
    # print(f"Max Confidence: {df['Confidence'].max():.2%}")
    
    # # Confidence distribution
    # low_conf = (df['Confidence'] < 0.9).sum()
    # med_conf = ((df['Confidence'] >= 0.9) & (df['Confidence'] < 0.95)).sum()
    # high_conf = (df['Confidence'] >= 0.95).sum()
    # print(f"\nConfidence Distribution:")
    # print(f"  Low (<90%): {low_conf} ({low_conf/len(df)*100:.1f}%)")
    # print(f"  Medium (90-95%): {med_conf} ({med_conf/len(df)*100:.1f}%)")
    # print(f"  High (≥95%): {high_conf} ({high_conf/len(df)*100:.1f}%)")
    
    # # print("\nPhân bố Label:")
    # # label_counts = df['Label'].value_counts()
    # # for label, count in label_counts.items():
    #     percentage = (count / len(df)) * 100
    #     print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # # So sánh Labels1 vs Label (nếu có cột Labels1)
    # # if 'Labels1' in df.columns:
    #     print("\n=== SO SÁNH LABELS1 (CHUẨN) VS LABEL ===")
        
    #     # Tính độ chính xác
    #     df['Match'] = df['Labels1'].str.strip().str.lower() == df['Label'].str.strip().str.lower()
    #     total = len(df)
    #     correct = df['Match'].sum()
    #     incorrect = total - correct
    #     accuracy = (correct / total) * 100
        
    #     print(f"Tổng số: {total}")
    #     print(f"Khớp: {correct} ({accuracy:.2f}%)")
    #     print(f"Sai lệch: {incorrect} ({100-accuracy:.2f}%)")
        
    #     # Chi tiết các trường hợp sai lệch
    #     if incorrect > 0:
    #         print("\n=== CHI TIẾT SAI LỆCH ===")
    #         mismatches = df[~df['Match']][['Id', 'Voice', 'Label', 'Method', 'SiteName', 'Title']]
            
    #         # Thống kê sai lệch theo Voice
    #         print("\nSai lệch theo Voice (chuẩn):")
    #         for voice_label in sorted(df['Voice'].unique()):
    #             voice_data = df[df['Voice'] == voice_label]
    #             voice_mismatches = voice_data[~voice_data['Match']]
    #             if len(voice_mismatches) > 0:
    #                 print(f"\n  {voice_label} (Chuẩn):")
    #                 predicted_counts = voice_mismatches['Label'].value_counts()
    #                 for pred_label, count in predicted_counts.items():
    #                     print(f"    → Dự đoán sai thành {pred_label}: {count} lần")
            
    #         # In ra một số ví dụ sai lệch
    #         print("\n=== MỘT SỐ VÍ DỤ SAI LỆCH ===")
    #         for idx, row in mismatches.head(10).iterrows():
    #             print(f"\n{'='*80}")
    #             print(f"ID: {row['Id']}")
    #             print(f"Voice (Chuẩn): {row['Voice']}")
    #             print(f"Label (Dự đoán): {row['Label']}")
    #             print(f"Method: {row['Method']}")
                
    #             # Get confidence if available
    #             if 'Confidence' in df.columns:
    #                 full_row = df.loc[idx]
    #                 print(f"Confidence: {full_row['Confidence']:.2%}")
                
    #             print(f"SiteName: {row['SiteName']}")
                
    #             # Lấy thông tin đầy đủ từ df
    #             full_row = df.loc[idx]
                
    #             title = str(full_row['Title']) if pd.notna(full_row['Title']) else ''
    #             content = str(full_row['Content']) if pd.notna(full_row['Content']) else ''
    #             description = str(full_row['Description']) if pd.notna(full_row['Description']) else ''
    #             type_val = str(full_row['Type']) if pd.notna(full_row['Type']) else ''
                
    #             print(f"Type: {type_val}")
                
    #             if title:
    #                 print(f"\nTitle:\n{title}")
    #             if content:
    #                 print(f"\nContent:\n{content}")
    #             if description:
    #                 print(f"\nDescription:\n{description}")
                
    #             print(f"{'='*80}")
    
    return df
