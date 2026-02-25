import unicodedata
import re

def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return text.lower().strip()


def merge_row(row, type):
    if "comment" in type.lower():
        return str(row["Content"]) + " | " + str(row["Description"])
    else:
        return str(row["Title"]) + " | " + str(row["Content"]) + " | " + str(row["Description"])


def classify_buzz_revelent(df, kws):

    df["merged"] = df.apply(lambda row: merge_row(row, row["Type"]), axis=1)
    df["Yes/No"] = "No"
    for topic_id in df["TopicId"].unique():
        topic_kws = kws.get(topic_id, [])
        print(f"Classifying TopicId {topic_id} with keywords: {topic_kws}")
        if not topic_kws:
            print(f"No keywords found for TopicId {topic_id}, skipping...")
            continue
        df.loc[df["TopicId"] == topic_id, "Yes/No"] = df.apply(
            lambda row: "Yes" if any(
                any(normalize_text(kw.split(" ")[i]) in normalize_text(row["merged"]) for i in range(len(kw.split(" "))))
                for kw in topic_kws
            ) else row["Yes/No"],
            axis=1
        )

    return df.drop(columns=["merged"])


def classify_buzz_category(df):
    pass