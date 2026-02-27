import pandas as pd

def sanitize_excel_values(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: f"'{x}" if isinstance(x, str) and x.strip().startswith('=') else x
        )
    return df

def clean_lower_text(text: str):
    return text.lower().strip()