import os
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
def get_path_in_project(path: str) -> str:
    current_dir = os.path.dirname(__file__)
    needed_path = os.path.join(current_dir, f"../../{path}")
    return needed_path


def transform_post_text(data: pd.DataFrame):

    data["text"] = data["text"].apply(lambda x: x.replace("\n", " "))

    stop_words_path = get_path_in_project("data/stop_words.csv")

    stop_words = pd.read_csv(stop_words_path)["word"].values.tolist()

    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                max_features = 30,
                                max_df = 0.95,
                                min_df = 0.01)
    
    tfidf_matrix = vectorizer.fit_transform(data["text"])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns = vectorizer.get_feature_names_out())

    data = pd.concat([data, tfidf_df], axis = 1)
    return data


def transform_timestamp(data: pd.DataFrame):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour_of_day'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.day_of_week
    data.sort_values(by='timestamp')
    data = data.drop(["timestamp"], axis=1)
    return data
