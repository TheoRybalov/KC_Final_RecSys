import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def get_path_in_project(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"../../{path}")

def process_text_data(df):
    df["text"] = df["text"].str.replace("\n", " ")
    stop_words_path = get_path_in_project("data/stop_words.csv")
    stop_words = pd.read_csv(stop_words_path)["word"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words=stop_words, max_features=30, max_df=0.95, min_df=0.01
    )
    tfidf = vectorizer.fit_transform(df["text"])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    return pd.concat([df, tfidf_df], axis=1)

def process_timestamp(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    return df.drop(columns="timestamp")

def merge_data(feed_df, user_df, post_df):
    return feed_df.merge(user_df, on="user_id", how="left") \
                  .merge(post_df, on="post_id", how="left")

def clean_final_data(df):
    return df.drop_duplicates(subset=["user_id", "post_id"]) \
             .drop(columns=["action", "exp_group"])
