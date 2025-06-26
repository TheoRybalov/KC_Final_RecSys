from sqlalchemy import create_engine
import os
import pandas as pd
from tqdm import tqdm

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
    estimated_chunks = 1000000 // CHUNKSIZE + 1

    for chunk_dataframe in tqdm(pd.read_sql(query, conn, chunksize=CHUNKSIZE), total=estimated_chunks):
        chunks.append(chunk_dataframe)
    
    conn.close()
    return pd.concat(chunks, ignore_index=True)

