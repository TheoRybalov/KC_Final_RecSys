from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.database.sql_utils import get_postgres_engine, batch_load_sql
from src.data_preprocessing.feature_utils import process_text_data, process_timestamp


ENGINE = get_postgres_engine()
def load_user_data():
    query = """
        SELECT * FROM public.user_data;
        """
    user_data_df = batch_load_sql(query, chunksize=100000)
    return user_data_df

def load_post_data():
    query = """
        SELECT * FROM public.post_text_df;
        """
    
    post_data_df = batch_load_sql(query, chunksize=100000)
    return post_data_df

def load_feed_data():
    query = "SELECT * FROM public.feed_data LIMIT 1000000;"

    user_data_df = batch_load_sql(query, chunksize=200000)
    return user_data_df
    

def train_catboost(X, y, cat_features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        custom_metric='AUC',
        eval_metric='AUC',
        verbose=100
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("ROC_AUC:", roc_auc_score(y_test, y_proba))

    return model

def save_data_to_sql(engine, data: pd.DataFrame, name):

    data.to_sql(name, con=engine, if_exists="replace", index = False)

def main():

    print("Началось")

    user_data_df = load_user_data()
    post_data_df = process_text_data(load_post_data())
    feed_data_df = process_timestamp(load_feed_data())



    
    user_data_df =  user_data_df.drop(["exp_group"], axis =1)

    save_data_to_sql(ENGINE,  user_data_df, 'fedorrybalov_lesson_22_user_data')
    print("сохранили в базу user_data")

    save_data_to_sql(ENGINE, post_data_df, 'fedorrybalov_lesson_22_post_data')
    print("сохранили в базу post_data")



    features = feed_data_df.merge(user_data_df, on="user_id", how="left")
    features = features.merge(post_data_df, on="post_id", how="left")

    features = features.drop_duplicates(subset=["user_id", "post_id"])
    features = features.drop(["action"], axis=1)


    X = features.drop(["target", "user_id", "post_id"], axis=1)
    y = features["target"]

    cat_features = ["hour_of_day", "day_of_week", "city", "country", "os", "source", "topic"]

    all_features = cat_features +[col for col in X.columns if col not in cat_features]
    X = X[all_features]


    model = train_catboost(X, y, cat_features)

    model.save_model('catboost_model',
                           format="cbm")
    


    

    

if __name__ == '__main__':

    main()