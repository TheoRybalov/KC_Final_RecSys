import os
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
from sqlalchemy import create_engine
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.model_selection import TimeSeriesSplit


def get_postgres_engine():
    pg_user = os.getenv("POSTGRES_USER", " ")
    pg_password = os.getenv("POSTGRES_PASSWORD", " ")
    pg_host = os.getenv("POSTGRES_HOST", " ")
    pg_port = os.getenv("POSTGRES_PORT", " ")
    pg_db = os.getenv("POSTGRES_DATABASE", " ")

    engine = create_engine(
        f"postgresql://{pg_user}:{pg_password}"
        f"{pg_host}:{pg_port}/{pg_db}")
    
    return engine



def get_user_data(engine):
    querry = """
        SELECT * FROM public.user_data;
        """
    
    user_data_df = pd.read_sql(querry, con=engine)
    return user_data_df

def get_post_text_df(engine):
    querry = """
        SELECT * FROM public.post_text_df;
        """
    
    user_data = pd.read_sql(querry, con=engine)
    return user_data

def get_feed_data(engine):
    querry = """
        SELECT * FROM public.feed_data LIMIT 5000000;
        """
    
    user_data_df = pd.read_sql(querry, con=engine)
    return user_data_df




def merge_data(feed_data, user_data, post_text_df):
    merged_df = feed_data.merge(user_data, on="user_id", how="left")
    merged_df = merged_df.merge(post_text_df, on="post_id", how="left")
    return merged_df


def process_text_data(data: pd.DataFrame):

    data["text"] = data["text"].apply(lambda x: x.replace("\n", " "))

    stop_words = pd.read_csv("../data/stop_words.csv")["word"].values.tolist()

    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                max_features = 30,
                                max_df = 0.95,
                                min_df = 0.01)
    
    tfidf_matrix = vectorizer.fit_transform(data["text"])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns = vectorizer.get_feature_names_out())

    data = pd.concat([data, tfidf_df], axis = 1)
    data = data.drop(["text"], axis = 1)
    return data


def process_timestamp(data: pd.DataFrame):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour_of_day'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.day_of_week
    data.sort_values(by='timestamp')
    data = data.drop(["timestamp"], axis=1)
    return data



def clean_final_data(data: pd.DataFrame):
   data = data.drop_duplicates(subset=["user_id", "post_id"], keep="first")
   data = data.drop(["user_id", "post_id"], axis=1)
   data = data.drop(["action"], axis=1)
   return data

# def prepare_for_model(data: pd.DataFrame):

#     X = merged_df.drop(["target"], axis=1)
#     y = merged_df["target"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     categorial_cols = ["hour_of_day", "day_of_week", "city", "country", "os", "source", "topic"]
#     cat_features = [X_train.columns.get_loc(col) for col in categorial_cols]
#     print(X.columns)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   


# def train_model(data: pd.DataFrame):

#     X = merged_df.drop(["target"], axis=1)
#     y = merged_df["target"]












user_data = get_user_data(engine)
post_text_df = get_post_text_df(engine)
feed_data = get_feed_data(engine)

current_dir = os.path.dirname(__file__)
stop_words_path = os.path.join(current_dir, "../../data/stop_words.csv")

stop_words = pd.read_csv(stop_words_path)["word"].values.tolist()

vectorizer = TfidfVectorizer(stop_words=stop_words,
                             max_features = 30,
                             max_df = 0.95,
                             min_df = 0.01)


tfidf_matrix = vectorizer.fit_transform(post_text_df["text"])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns = vectorizer.get_feature_names_out())

post_data = pd.concat([post_text_df, tfidf_df], axis = 1)
post_data = post_data.drop(["text"], axis = 1)


feed_data['timestamp'] = pd.to_datetime(feed_data['timestamp'])
feed_data['hour_of_day'] = feed_data['timestamp'].dt.hour
feed_data['day_of_week'] = feed_data['timestamp'].dt.day_of_week
feed_data = feed_data.drop(["action"], axis=1)
feed_data = feed_data.drop(["timestamp"], axis=1)


merged_df = merge_data(feed_data, user_data, post_data)


output_dir = os.path.join(os.path.dirname(__file__), '../../data')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'merged_data.csv')
merged_df.to_csv(output_path, index=False, sep=";")

