import os
import pickle
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("src\model\catboost_model.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)

    
    return model




def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    
    chunks = []
    
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features():

    user_query = """
        SELECT * FROM fedorrybalov_lesson_22_user_data;
        """
    
    user_df = batch_load_sql(user_query)
    

    post_text_query = """
        SELECT * FROM fedorrybalov_lesson_22_post_data;
        """
    
    post_text_df = batch_load_sql(post_text_query)
    
    feed_data_query = """
        SELECT * FROM fedorrybalov_lesson_22_feed_data LIMIT 10;
        """
    
    feed_data_df = batch_load_sql(feed_data_query)

    return user_df, post_text_df, feed_data_df





    
    
    

    


