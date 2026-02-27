import aiohttp
from typing import Dict, List
from src.settings import CMS_URL, CMS_REFERER, CMS_USER_NAME, CMS_PASSWORD, PROJECT_DIR
import time
import json
import os

AUTH_FILE_PATH =  os.path.join(PROJECT_DIR, "auth/token_cache.json")


def load_cache(cache_file_path=AUTH_FILE_PATH):
    if not os.path.exists(cache_file_path):
        return {}

    with open(cache_file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def save_cache(cache, cache_file_path=AUTH_FILE_PATH):
    with open(cache_file_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

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

async def get_keywords(token: Dict, topic_id: str) -> List[str]:

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
    if not topic_data:
        raise RuntimeError(f"Get keywords failed: {result}")

    if topic_data.get("message") != "Success":
        raise RuntimeError(
            f"Get keywords failed: {topic_data.get('message')}"
        )

    return topic_data["data"].get("mainKeys", [])