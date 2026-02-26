"""
Phân tích từ khóa xuất hiện nhiều nhất trong các dòng của cùng một group trong DataFrame
"""

import pandas as pd
from collections import Counter
import re
import unicodedata
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản: loại bỏ dấu, ký tự đặc biệt, chuyển về chữ thường
    """
    if not isinstance(text, str):
        return ""
    
    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    
    # Loại bỏ ký tự đặc biệt, giữ lại chữ và số
    text = re.sub(r"[^\w\s]", " ", text)
    
    return text.lower().strip()


def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """
    Trích xuất từ khóa từ văn bản
    """
    normalized = normalize_text(text)
    words = normalized.split()
    
    # Lọc từ có độ dài tối thiểu
    keywords = [word for word in words if len(word) >= min_length]
    
    return keywords


def analyze_group_keywords(
    df: pd.DataFrame,
    group_column: str,
    text_columns: List[str],
    top_n: int = 10,
    min_word_length: int = 2
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Phân tích từ khóa xuất hiện nhiều nhất cho mỗi group
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    group_column : str
        Tên cột dùng để group
    text_columns : List[str]
        Danh sách các cột chứa nội dung văn bản cần phân tích
    top_n : int
        Số lượng từ khóa top cần trả về
    min_word_length : int
        Độ dài tối thiểu của từ khóa
    
    Returns:
    --------
    Dict[str, List[Tuple[str, int]]]
        Dictionary với key là giá trị group, value là list các tuple (từ khóa, số lần xuất hiện)
    """
    results = {}
    
    # Group dữ liệu
    grouped = df.groupby(group_column)
    
    for group_name, group_df in grouped:
        all_keywords = []
        
        # Duyệt qua từng dòng trong group
        for _, row in group_df.iterrows():
            # Kết hợp nội dung từ các cột text
            combined_text = " ".join([
                str(row[col]) if pd.notna(row[col]) else ""
                for col in text_columns
            ])
            
            # Trích xuất từ khóa
            keywords = extract_keywords(combined_text, min_word_length)
            all_keywords.extend(keywords)
        
        # Đếm tần suất xuất hiện
        keyword_counts = Counter(all_keywords)
        
        # Lấy top N từ khóa
        top_keywords = keyword_counts.most_common(top_n)
        results[group_name] = top_keywords
    
    return results


def print_keyword_analysis(results: Dict[str, List[Tuple[str, int]]]):
    """
    In kết quả phân tích từ khóa
    """
    for group_name, keywords in results.items():
        print(f"\n{'='*60}")
        print(f"Group: {group_name}")
        print(f"{'='*60}")
        
        if keywords:
            print(f"{'Từ khóa':<30} {'Số lần xuất hiện':>20}")
            print(f"{'-'*60}")
            for keyword, count in keywords:
                print(f"{keyword:<30} {count:>20}")
        else:
            print("Không có từ khóa nào được tìm thấy")


def save_results_to_excel(
    results: Dict[str, List[Tuple[str, int]]],
    output_file: str
):
    """
    Lưu kết quả phân tích vào file Excel
    """
    data = []
    
    for group_name, keywords in results.items():
        for keyword, count in keywords:
            data.append({
                'Group': group_name,
                'Keyword': keyword,
                'Count': count
            })
    
    df_results = pd.DataFrame(data)
    df_results.to_excel(output_file, index=False)
    print(f"\nĐã lưu kết quả vào file: {output_file}")


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc dữ liệu
    df = pd.read_json("data/sample_data.json")
    
    # Phân tích từ khóa theo source
    results = analyze_group_keywords(
        df=df,
        group_column="source",
        text_columns=["title", "content", "description"],
        top_n=15,
        min_word_length=3
    )
    
    # In kết quả
    print_keyword_analysis(results)
    
    # Lưu kết quả vào Excel
    save_results_to_excel(results, "keyword_analysis_results.xlsx")
