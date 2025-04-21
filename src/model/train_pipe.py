import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


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

def get_path_in_project(path: str) -> str:

    current_dir = os.path.dirname(__file__)
    needed_path = os.path.join(current_dir, f"../../{path}")
    return needed_path


def get_user_data(engine):
    query = """
        SELECT * FROM public.user_data;
        """
    conn = engine.connect().execution_options(stream_results=True)
    user_data_df = pd.read_sql(query, con=conn)
    conn.close()
    return user_data_df


def get_post_text_df(engine):
    query = """
        SELECT * FROM public.post_text_df;
        """
    conn = engine.connect().execution_options(stream_results=True)
    post_text_df = pd.read_sql(query, con=engine)
    conn.close()
    return post_text_df

def get_feed_data(engine):
    query = "SELECT * FROM public.feed_data LIMIT 1000000;"
    CHUNKSIZE = 200000

    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    
    # Предполагаем максимум чанков
    estimated_chunks = 1000000 // CHUNKSIZE + 1

    for chunk_dataframe in tqdm(pd.read_sql(query, conn, chunksize=CHUNKSIZE), total=estimated_chunks):
        chunks.append(chunk_dataframe)
    
    conn.close()
    return pd.concat(chunks, ignore_index=True)





def merge_data(feed_data, user_data, post_text_df):
    merged_df = feed_data.merge(user_data, on="user_id", how="left")
    merged_df = merged_df.merge(post_text_df, on="post_id", how="left")
    return merged_df


def process_text_data(data: pd.DataFrame):

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
   data = data.drop(["action"], axis=1)
   data = data.drop(["exp_group"], axis=1)
   return data

def prepare_for_model(data: pd.DataFrame):

    X = data.drop(["target"], axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorial_cols = ["hour_of_day", "day_of_week", "city", "country", "os", "source", "topic"]
    cat_features = [X_train.columns.get_loc(col) for col in categorial_cols]
    print(X.columns)


def save_data_to_sql(engine, data: pd.DataFrame, name):

    data.to_sql(name, con=engine, if_exists="replace", index = False)

def main():

    engine = get_postgres_engine()
    user_data = get_user_data(engine)
    post_text_df = get_post_text_df(engine)
    feed_data = get_feed_data(engine)

    post_text_df = process_text_data(post_text_df)

    feed_data = process_timestamp(feed_data)

    # save_data_to_sql(engine, user_data, 'fedorrybalov_lesson_22_user_data')
    # print("сохранили в базу user_data")

    # save_data_to_sql(engine, post_text_df, 'fedorrybalov_lesson_22_post_data')
    # print("сохранили в базу post_data")

    # save_data_to_sql(engine, feed_data, 'fedorrybalov_lesson_22_feed_data')
    # print("сохранили в базу feed_data")


    full_data = merge_data(feed_data, user_data, post_text_df)

    full_data = clean_final_data(full_data)

    

    # save_data_to_sql(engine, full_data, 'fedorrybalov_lesson_22_features')
    # print("сохранили в базу все")

    print(full_data.head())


    X = full_data.drop(["target", "user_id", "post_id"], axis=1)
    y = full_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorial_cols = ["hour_of_day", "day_of_week", "city", "country", "os", "source", "topic"]
    cat_features = [X_train.columns.get_loc(col) for col in categorial_cols]
    print(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(iterations = 1000,
                            depth = 6,
                            learning_rate= 0.1,
                            custom_metric='AUC',
                            eval_metric='AUC',
                            verbose = 100)

    model.fit(X_train, y_train, cat_features=cat_features)

    print("Accuracy: ", accuracy_score(y_test, model.predict(X_test)))
    print("Precision: ", precision_score(y_test, model.predict(X_test), average='weighted'))
    print("Recall:", recall_score(y_test, model.predict(X_test), average='weighted'))
    print("ROC_AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], ))

    model.save_model('catboost_model',
                           format="cbm")


if __name__ == "__main__":
    main()
