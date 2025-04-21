from sqlalchemy import create_engine
import os
import pandas as pd

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

def batch_load_sql(query: str, chunksize: int) -> pd.DataFrame:
    CHUNKSIZE = chunksize
    
    engine = get_postgres_engine()
    conn = engine.connect().execution_options(stream_results=True)
    
    chunks = []
    
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def save_to_sql(df: pd.DataFrame, table_name: str, engine):
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)