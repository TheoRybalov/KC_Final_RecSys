import os
import pickle
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
import datetime
import numpy as np

from src.database.database import SessionLocal
from src.database.tables.table_user import User
from src.database.tables.table_post import Post
from src.database.tables.table_feed import Feed
from src.database.schema import UserGet, PostGet, FeedGet

from fastapi import FastAPI, Depends, HTTPException
from loguru import logger


app = FastAPI()
def get_db():
    with SessionLocal() as db:
        return db
    

@app.get("/user/{id}", response_model=UserGet)
def user_info(id, int, db: Session = Depends(get_db)):
    result = db.query(User).filter(User.user_id == id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Not found")
    else:
        return result
    

# @app.get("/post/{id}", response_model = PostGet)
# def post_into(id: int, db: Session = Depends(get_db)):
#     result =  db.query(Post).filter(Post.id == id).first()
#     if not result:
#         raise HTTPException(404, "user not found")
#     else:
#         return result

# @app.get("/user/{id}/feed", response_model = List[FeedGet])
# def user_feed_into(id: int, limit: int = 10,  db: Session = Depends(get_db)):
#     result =  (
#         db.query(Feed)
#         .filter(Feed.user_id == id)
#         .order_by(Feed.time.desc())
#         .limit(limit)
#         .all()
#     )
#     return result

# @app.get("/post/{id}/feed", response_model = List[FeedGet])
# def post_feed_into(id: int, limit: int = 10,  db: Session = Depends(get_db)):
#     result =  (
#         db.query(Feed)
#         .filter(Feed.post_id == id)
#         .order_by(Feed.time.desc())
#         .limit(limit)
#         .all()
#     )
#     return result

# @app.get("/post/recommendations/", response_model = List[PostGet])
# def get_recommended_feed(id: Optional[int] = None, limit: int = 10,  db: Session = Depends(get_db)):
#     result =  (
#         db.query(Post)
#         .select_from(Feed)
#         .filter(Feed.action == "like")
#         .join(Post, Post.id == Feed.post_id)
#         .group_by(Post.id)
#         .order_by(desc(func.count(Feed.post_id)))
#         .limit(limit)
#         .all()
#     )
    






def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("C:\\Users\\fedor\\KC_Final_RecSys\\catboost_model")
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

    logger.info("loading user_data")
    user_query = """
        SELECT * FROM fedorrybalov_lesson_22_user_data;
        """
    
    user_df = batch_load_sql(user_query)
    

    post_text_query = """
        SELECT * FROM fedorrybalov_lesson_22_post_data;
        """
    
    logger.info("loading post_data")
    post_text_df = batch_load_sql(post_text_query)
    
    feed_user_likes_query = """
        SELECT distinct user_id, post_id FROM feed_data WHERE action = 'like';
        """
    logger.info("loading user liked posts")
    feed_user_likes_df = batch_load_sql(feed_user_likes_query)

    return user_df, post_text_df, feed_user_likes_df


logger.info("loading features from DB")
USER_FEATURE, POST_FEATURE, FEED_LIKES_FEATURE = load_features()
logger.info("loading model")
model = load_models()

def recommended_posts(
		id: int, 
		time: datetime, 
		limit: int = 5):
    
    logger.info("preparing features for model")
    user_data = USER_FEATURE.loc[USER_FEATURE["user_id"] == id].drop(["user_id"], axis =1)

    post_data = POST_FEATURE.copy()

    user_post_data = user_data.merge(post_data, how = "cross").set_index("post_id")
    user_post_data = user_post_data.drop(["text"], axis=1)

    user_post_data["hour_of_day"] = time.hour
    user_post_data["day_of_week"] = time.weekday()

    cat_features = ["hour_of_day", "day_of_week", "city", "country", "os", "source", "topic"]

    all_features = cat_features +[col for col in user_post_data.columns if col not in cat_features]
    user_post_data = user_post_data[all_features]

    logger.info("making predictions...")
    predicts = model.predict_proba(user_post_data)[:, 1]

    user_post_data["predicts"] = predicts
    logger.info("predictions have been made")


    user_post_likes_idx = FEED_LIKES_FEATURE.loc[FEED_LIKES_FEATURE["user_id"] == id]["post_id"].values


    unliked_posts = user_post_data[~user_post_data.index.isin(user_post_likes_idx)]

    recommended_post_idx = unliked_posts.sort_values("predicts")[-limit:].index



    recommendations = post_data[post_data["post_id"].isin(recommended_post_idx)]


    print(recommendations[["post_id", "text", "topic"]].head())

    return [
        PostGet(**{
            "id": i,
            "text": post_data[post_data["post_id"] == i]["text"].values[0],
            "topic": post_data[post_data["post_id"] == i]["topic"].values[0]
        } ) for i in recommended_post_idx
    ]




print(recommended_posts(113947, datetime.datetime.now()))
print(recommended_posts(201, datetime.datetime.now()))
print(recommended_posts(200, datetime.datetime.now()))


# @app.get("/post/recommendations/", response_model=List[PostGet])
# def recommended_posts(id: int, time: datetime, limit: int = 10, db: Session = Depends(get_db)) -> List[PostGet]:
#     result = db.query(User).filter(User.user_id == id).first()
#     if not result:
#         raise HTTPException(status_code=404, detail="Not found")
#     else:
#         return recommended_posts(id, datetime.datetime.now(), 5)
    

     
    


    




    
    
    

    


