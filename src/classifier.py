import asyncio
from pyexpat import model
import unicodedata
import re
import json
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from src.get_kws import get_keywords, login_cms
from src.models import loader
from src.settings import PROJECT_DIR
from src.utils import sanitize_excel_values, clean_lower_text

load_dotenv()

RULES_PATH = os.path.join(PROJECT_DIR, "src/rules/label-rules.json")
SOURCE_MAPPING_PATH = os.path.join(PROJECT_DIR, "src/rules/source-mapping.json")
SPECIAL_PROJECT = "ShopeeFood"

class LabelClassifier:
    def __init__(self, project_name: str = "ShopeeFood", rules_path: str = RULES_PATH, source_mapping_path: str = SOURCE_MAPPING_PATH):
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
        self.project_name = project_name
        
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
        is_topic = "topic" in data.get("type", "").lower()
        
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
  
        merged_text = re.sub(r'https?://\S+', '', merged_text)
     
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()
        
        return merged_text
    
    def get_default_label(self) -> str:
        """Get default label for project when confidence is low"""
        return self.default_labels.get(self.project_name, "Unknown")
    
    def get_project_rules(self) -> Optional[Dict]:
        """Get rules for specific project"""
        for project in self.rules_data.get("projects", []):
            if project["project_name"] == self.project_name:
                return project
        return None

    
    def check_source_mapping(self, site_name: str) -> Optional[str]:
        """Check if SiteName matches predefined source mapping (highest priority)"""
        if not site_name or self.project_name not in self.source_mapping:
            return None
        
        # Check each label category for matching source
        for label_type, sources in self.source_mapping[self.project_name].items():
            if site_name in sources:
                return label_type
        
        return None
    
    def check_brand_indicators(self, site_name: str, buzz_type: str, author: str) -> bool:
        """Check if content is from official brand channels"""

        if "topic" in clean_lower_text(buzz_type):
            # Official ShopeeFood channels - ONLY the main brand pages
            brand_indicators = [
                "ShopeeFood VN",
                "ShopeeFood Vietnam"
            ]
            
            # Check SiteName or Author - must be exact match or very close
            site_name_lower = clean_lower_text(site_name)
            author_lower = clean_lower_text(author)
            
            for indicator in brand_indicators:
                indicator_lower = clean_lower_text(indicator)
                # Exact match or starts with the indicator
                if (site_name_lower == indicator_lower or 
                    author_lower == indicator_lower or
                    site_name_lower.startswith(indicator_lower + " ") or
                    author_lower.startswith(indicator_lower + " ")):
                    return True
        
        return False
    
    def rule_based_classification(self, text: str) -> Optional[str]:
        """Apply rule-based classification using keywords"""
        project_rules = self.get_project_rules()
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
    
    def load_pretrained_model(self) -> Tuple[Optional[object], Optional[object]]:
        """
        Load pretrained model and label encoder for a project
        Returns (model, label_encoder) or (None, None) if not available
        """
        # Check cache first
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

    
    def llm_classification(self, text: str, site_name: str = "", author: str = "") -> tuple:
        """Use LLM to classify when rule-based fails. Returns (label, confidence)"""
        project_rules = self.get_project_rules()
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
        if self.project_name == "ShopeeFood":
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
        prompt = f"""You are a text classifier. Return ONLY the label name, nothing else. No thinking, no explanation.
Your task is classify the following text into ONE label for project "{self.project_name}".

** LABELS **
{chr(10).join([f"{i+1}. {ld}" for i, ld in enumerate(label_definitions)])}
{context_info}
{additional_instructions}

** INSTRUCTIONS **
- Read the text carefully and identify the AUTHOR'S ROLE/PERSPECTIVE
- Match content to the most appropriate label based on who is posting and their perspective

** OUTPUT FORMAT **
- Return ONLY the label name followed by confidence score (0-100) in format: <label>|<confidence>
"""
        
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
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"""Text need to classify:
{text}"""}
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
                return self.get_default_label(), 0.2
            else:
                print(f"LLM API error: {response.status_code}")
                return self.get_default_label(), 0.0
                
        except Exception as e:
            print(f"LLM classification error: {e}")
            return self.get_default_label(), 0.0
    
    def classify(self, data: Dict) -> tuple:
        """Main classification method - returns (label, method, confidence)"""
        # Extract project/topic
        
        # Extract SiteName and Author for source mapping check
        site_name = data.get("siteName", "")
        author = data.get("author", "")
        buzz_type = data.get("type", "")
        # Extract and merge text early for ownership detection
        text = self.extract_text(data)
        if not text:
            return self.get_default_label(), "NoText", 0.0
        
        if clean_lower_text(self.project_name) == clean_lower_text(SPECIAL_PROJECT):
        # PRIORITY 1: Check if it's official Brand content (100% confidence)
            if self.check_brand_indicators(site_name, buzz_type, author):
                return "Brand", "BrandIndicator", 1.0
            
            # PRIORITY 2: Check source mapping (Shipper/Merchant groups) (95% confidence)
            if site_name:
                label = self.check_source_mapping(site_name)
                if label:
                    return label, "SourceMapping", 0.95
              
            # # PRIORITY 3: Check for strong merchant ownership indicators (90% confidence)
            # if self.detect_merchant_ownership(text, site_name):
            #     return "Merchant", "OwnershipDetection", 0.9
            
        # PRIORITY 4: Try rule-based classification (80% confidence)
        label = self.rule_based_classification(text)
        
        if label is not None:
            return label, "RuleBased", 0.8
        
        # PRIORITY 5: Try pretrained model classification (confidence from model)
        label, confidence = self.model_based_classification(text)
        
        if label is not None and confidence >= self.confidence_threshold:
            return label, "PretrainedModel", confidence
        
        # PRIORITY 6: If model fails or low confidence, use LLM with confidence score
        label, confidence = self.llm_classification(text, site_name, author)
        
        # If confidence is below threshold, use default label
        if confidence < self.confidence_threshold:
            default_label = self.get_default_label()
            return default_label, "LowConfidence", confidence
        
        return label, "LLM", confidence




def classify_category(df: pd.DataFrame, project_name: str, max_workers: int = 10):

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
    row = next(df.itertuples())
    print(row._fields)

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

    # unpack results
    labels, methods, confidences = zip(*results)

    df['Label'] = labels
    df['Method'] = methods
    df['Confidence'] = confidences

    df = sanitize_excel_values(df)

    return df

