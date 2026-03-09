import aiohttp
from typing import Dict, List
from src.settings import CMS_URL, CMS_REFERER, CMS_USER_NAME, CMS_PASSWORD, PROJECT_DIR
import time
import json
import os

AUTH_FILE_PATH = os.path.join(PROJECT_DIR, "auth/token_cache.json")
MANUAL_KEYWORDS_PATH = os.path.join(PROJECT_DIR, "src/keywords.json")


def load_cache(cache_file_path=AUTH_FILE_PATH):
    if not os.path.exists(cache_file_path):
        return {}

    with open(cache_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache, cache_file_path=AUTH_FILE_PATH):
    with open(cache_file_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_manual_keywords(keywords_file_path=MANUAL_KEYWORDS_PATH) -> Dict[str, List[str]]:
    """
    Load manual keywords from JSON file
    Returns dict with topic_name as key and list of keywords as value
    """
    if not os.path.exists(keywords_file_path):
        print(f"Manual keywords file not found: {keywords_file_path}")
        return {}
    
    try:
        with open(keywords_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading manual keywords: {e}")
        return {}

async def login_cms() -> Dict:
    token_cache = load_cache()
    if token_cache:
        if token_cache.get("expires_at", 0) - time.time() > 30 * 60:
            return token_cache
        
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,vi;q=0.8",
        "content-type": "application/json",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "referer": CMS_REFERER,
    }

    payload = {
        "operationName": "login",
        "variables": {
            "input": {
                "username": CMS_USER_NAME,
                "password": CMS_PASSWORD
            }
        },
        "query": """
        mutation login($input: LoginInput!) {
          login(input: $input) {
            status
            message
            refreshToken
            accessToken
            data {
              _id
              username
              firstName
              lastName
              email
              phone
              avatar
              permissions {
                group
                roles
              }
              status
              createdBy
              createdAt
              updatedBy
              updatedAt
            }
          }
        }
        """
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(
            CMS_URL,
            json=payload,
            timeout=20
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()


    login_data = result.get("data", {}).get("login")

    if not login_data:
        raise RuntimeError(f"Login failed: {result}")

    if login_data.get("message") != "Success":
        raise RuntimeError(
            f"Login failed: {login_data.get('message')}"
        )
    
    save_cache({
        "access_token": login_data["accessToken"],
        "refresh_token": login_data["refreshToken"],
        "expires_at": time.time() + 60 * 60 * 7
    })

    return {
        "access_token": login_data["accessToken"],
        "refresh_token": login_data["refreshToken"],
    }

async def get_keywords(token: Dict, topic_id: str, topic_name: str = "", project: str = "Vinamilk") -> List[str]:
    """
    Get keywords from both API and manual keywords file
    
    Parameters:
    -----------
    token : Dict
        Authentication token
    topic_id : str
        Topic ID for API call
    topic_name : str
        Topic name for manual keywords lookup
    
    Returns:
    --------
    List[str]
        Combined list of keywords from API and manual file (deduplicated)
    """
    keywords = []
    
    # Get keywords from API
    try:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,vi;q=0.8",
            "content-type": "application/json",
            "x-token": f"Bearer {token['access_token']}",
            "x-refresh-token": f"Bearer {token['refresh_token']}",
            "referer": CMS_REFERER,
        }

        payload = {
            "operationName": "topic",
            "variables": {
                "_id": topic_id
            },
            "query": """
            query topic($_id: ID!) {
              topic(_id: $_id) {
                status
                message
                data {
                  _id
                  name
                  mainKeys
                }
              }
            }
            """
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                CMS_URL,
                json=payload,
                timeout=20
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

        topic_data = result.get("data", {}).get("topic")
        if topic_data and topic_data.get("message") == "Success":
            topic_name = topic_data["data"].get("name", "")
            api_keywords = topic_data["data"].get("mainKeys", [])
            keywords.extend(api_keywords)
            
        else:
            print(f"Warning: Could not get keywords from API for topic_id {topic_id}")
    
    except Exception as e:
        print(f"Error getting keywords from API for topic_id {topic_id}: {e}")
    
    # Get manual keywords from file
    if topic_name:
        manual_keywords_dict = load_manual_keywords()
        if project == "Vinamilk":
            manual_keywords = manual_keywords_dict.get("vinamilk", [])
        else:
            manual_keywords = manual_keywords_dict.get(topic_name, [])
        
        if manual_keywords:
            keywords.extend(manual_keywords)
            print(f"Added {len(manual_keywords)} manual keywords for topic '{topic_name}'")
    
    # Deduplicate keywords (case-insensitive)
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)
    
    return unique_keywords