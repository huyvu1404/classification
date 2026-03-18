"""
Simple test script for LLM call
Quick test to verify LLM API is working with parallel requests
"""
import asyncio
import aiohttp
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 10

test_prompt = """<|im_start|>system
Bạn là chuyên gia phân loại hành vi người dùng trên các cộng đồng thương mại điện tử Việt Nam. Nhiệm vụ: Xác định người đăng bài/comment là **SELLER** (người bán, chủ shop, đang kinh doanh) hay **BUYER** (người mua, người tiêu dùng thông thường).

### Nguyên tắc cốt lõi (ưu tiên tuyệt đối):
- SELLER: Góc nhìn từ phía **cung cấp sản phẩm/dịch vụ**, vận hành gian hàng, bán hàng.
- BUYER: Góc nhìn từ phía **tiêu dùng**, tìm mua, trải nghiệm mua sắm.

### Quy tắc đặc biệt cho MINIGAME:
→ **SELLER** nếu đăng trong **GROUP BÁN HÀNG rõ ràng** (tên group BẮT BUỘC chứa từ khóa liên quan tới bán hàng, kinh doanh, lập nghiệp, con buôn, seller, hội bán hàng... Ví dụ: Tâm sự con buôn, Hội seller Shopee, Lập nghiệp Shopee, Bán hàng với Shopee, Chia sẻ kinh nghiệm bán hàng).
  → **BUYER** nếu đăng trên **fanpage chính thức của Shopee** (ShopeeVN, Shopee, ShopeeOfficial, ShopeePayVN, ShopeeFoodVN...).
  → **BUYER** nếu đăng trong các group hỗ trợ game như "GIÚP TƯỚI CÂY SHOPEE", "NÔNG TRẠI SHOPEE", "SĂN XU SHOPEE", "Chơi game Shopee", "Tưới cây Shopee"... (nhóm này là cộng đồng người chơi săn xu, không phải bán hàng).
  → Nếu không thuộc các trường hợp trên → quay về phân tích content theo các dấu hiệu dưới.

### Dấu hiệu mạnh → SELLER:
- Giới thiệu, quảng cáo sản phẩm đang bán, link shop, "shop bên mình", "inbox để đặt hàng", "add Zalo".
- Quản lý shop: doanh số, phí sàn, hoàn hàng (góc shop), chạy ads, decor shop, up ảnh sản phẩm.
- Khiếu nại từ góc shop: đơn bị huỷ oan, vận chuyển thiệt hại cho shop.
- Đăng trong group bán hàng rõ ràng.

### Dấu hiệu mạnh → BUYER:
- Review, Seeding sản phẩm.
- Chương trình sale, giảm giá, deal hot trên các sàn thương mại điện tử.
- Hỏi mua, săn deal, xin link, than chất lượng, khoe deal săn được, "mới nhận hàng", "ship chậm quá".
- Khiếu nại từ góc người mua: hoàn tiền khó, hàng lỗi, shop lừa.
- Chia sẻ mã giảm giá chung, kinh nghiệm dùng app như người dùng.
- Đăng trong group buyer rõ ràng (Nghiện Shopee, Hội săn sale...).

### Các vùng giao thoa (xử lý sau minigame):
- Khiếu nại vận chuyển/hoàn hàng: "shop bị thiệt" → SELLER; "mình bị mất hàng/hoàn mãi không được" → BUYER.
- Affiliate/Shopee Video/KOL: thường BUYER (trừ khi rõ ràng quản lý shop + affiliate).
- Comment trong topic bán hàng: hỏi mua/tư vấn → BUYER; chia sẻ kinh nghiệm bán → SELLER.

### Định dạng đầu ra (TUYỆT ĐỐI):
- CHỈ trả về đúng 1 dòng: SELLER hoặc BUYER
- KHÔNG giải thích
- KHÔNG tự suy diễn mà phải dựa vào dữ liệu được cung cấp
- KHÔNG thêm bất kỳ nội dung nào khác.
<|im_end|>
<|im_start|>user
Dữ liệu JSON:
{data}
"""


def row_to_json(row):
    buzz_type = "comment" if "comment" in row["Type"].lower() else "topic"
    data = {
        "type": buzz_type,
        "title": row["Title"],
        "content": row["Content"],
        "description": row["Description"],
        "siteName": row["SiteName"]
    }
    return json.dumps(data, ensure_ascii=False, indent=4)


async def call_llm(session: aiohttp.ClientSession, json_str: str) -> tuple:
    """Call LLM API for a single row, returns (label, explain)"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": os.getenv("MODEL_NAME"),
        "messages": [
            {"role": "user", "content": test_prompt.format(data=json_str)}
        ],
        "temperature": 0.1,
        "max_tokens": 150,
        "stream": False
    }

    try:
        async with session.post(
            f"{os.getenv('LLM_API_URL')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                result = await response.json()
                label = result["choices"][0]["message"]["content"].strip()

                
                return label
            else:
                print(f"Error: status {response.status}")
                return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""


async def test_simple_llm_call():
    df = pd.read_excel("/Users/huyvu/Downloads/Seller Buyer Project Shopee (1).xlsx")
    df = df.reset_index(drop=True)  # ensure positional index matches row order

    # Pre-allocate results aligned to df index
    results = [""] * len(df)

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i + BATCH_SIZE]
            # Pair each task with its absolute position in df
            indexed_rows = [(pos, row) for pos, (_, row) in enumerate(batch.iterrows(), start=i)]
            tasks = [call_llm(session, row_to_json(row)) for _, row in indexed_rows]
            batch_results = await asyncio.gather(*tasks)

            for (pos, _), label in zip(indexed_rows, batch_results):
                results[pos] = label

            print(f"Processed {min(i + BATCH_SIZE, len(df))}/{len(df)} rows")

    df["llm_label"] = results
    df.to_excel("test.xlsx", index=False)
    print("Done. Saved to test.xlsx")


if __name__ == "__main__":
    asyncio.run(test_simple_llm_call())
