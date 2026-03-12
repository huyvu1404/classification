"""
Simple test script for LLM call
Quick test to verify LLM API is working
"""
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()
from src.detector import Detector

async def test_simple_llm_call():
    """Test a simple LLM API call"""
    
#     # Get config from .env
#     llm_api_url = os.getenv("LLM_API_URL", "")
#     llm_api_key = os.getenv("LLM_API_KEY", "")
#     llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
#     print("=" * 60)
#     print("Simple LLM API Test")
#     print("=" * 60)
#     print(f"API URL: {llm_api_url}")
#     print(f"Model: {llm_model}")
#     print(f"API Key: {'Set' if llm_api_key else 'Not set'}")
#     print()
    
#     if not llm_api_url:
#         print("❌ Error: LLM_API_URL not configured")
#         print("\nPlease add to .env file:")
#         print("  LLM_API_URL=https://your-api-url.com/v1")
#         print("  LLM_API_KEY=your_api_key (optional)")
#         print("  LLM_MODEL=gpt-3.5-turbo")
#         return
    
#     # Simple test prompt
    test_prompt = """<|im_start|>system
Bạn là một chuyên gia phân loại nội dung.

Nhiệm vụ của bạn là kiểm tra xem trong nội dung bình luận của người dùng có nhắc đến hoặc tag tên của một người khác hay không.

*** QUY TẮC PHÂN TÍCH ***

1. Các trường hợp được xem là **Yes**:
- Có ký hiệu @ trước tên hoặc username
   Ví dụ:
   + "@huy.vu cái này hay nè"
   + "xem cái này nè @minh.nguyen"

- Có chứa tên, biệt danh:
   Ví dụ:
   + "Huy Vu cái này hay nè"
   + "Nguyen Minh xem thử"
   + "Ly Ly"
   + "Ngày mai đi thử Huy Vu"

2. Các trường hợp được xem là **No**:
- Nội dung không chứa username hoặc tên người cụ thể
- Nội dung chỉ chứa tên của tổ chức, công ty, doanh nghiệp, địa phương


*** ĐỊNH DẠNG ĐẦU RA ***
- Chỉ trả về đúng một dòng duy nhất chứa Yes hoặc No
- KHÔNG lặp lại yêu cầu
- KHÔNG giải thích lí do
- KHÔNG chứa bất kì kí tự nào khác

<|im_end|>
<|im_start|>user
Nội dung bình luận cần phân tích:
{comment}
"""
#     print(f"Test Prompt: {test_prompt}")
#     print("\nCalling LLM API...")
    
#     try:
#         headers = {
#             "Content-Type": "application/json"
#         }
        
#         if llm_api_key:
#             headers["Authorization"] = f"Bearer {llm_api_key}"
        
#         payload = {
#             "model": llm_model,
#             "messages": [
                
#                 {"role": "user", "content": test_prompt.format(comment="Bảo Bảo ")}
#             ],
#             "temperature": 0.1,
#             "max_tokens": 10,
#             "stream": False
#         }
        
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 f"{llm_api_url}/chat/completions",
#                 headers=headers,
#                 json=payload,
#                 timeout=aiohttp.ClientTimeout(total=30)
#             ) as response:
#                 print(f"Status Code: {response.status}")
                
#                 if response.status == 200:
#                     result = await response.json()
#                     answer = result["choices"][0]["message"]["content"]
                    
#                     print(f"\n✓ Success!")
#                     print(f"Response: {answer}")
#                     print(f"\nFull response:")
#                     print(result)
#                 else:
#                     error_text = await response.text()
#                     print(f"\n❌ Error: Status {response.status}")
#                     print(f"Response: {error_text}")
                    
#     except aiohttp.ClientError as e:
#         print(f"\n❌ Network Error: {e}")
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
    detector = Detector()
    comment = "SuSu"
    async with aiohttp.ClientSession() as session:
        res = await detector._detect_name_tags_llm(comment, session)
        print(res)
        print("++++++++++")
        # print("Test 2:", test_prompt.format(comment=comment))
        res2 = await detector._call_llm(prompt=test_prompt.format(comment=comment), session=session)
        print(res2)
if __name__ == "__main__":
    asyncio.run(test_simple_llm_call())
