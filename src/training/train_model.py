from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.database.sql_utils import get_postgres_engine, batch_load_sql
from src.data_preprocessing.feature_utils import process_text_data, process_timestamp
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
    

def train_catboost(X, y, cat_features_idx):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        custom_metric='AUC',
        eval_metric='AUC',
        verbose=100
    )

    model.fit(X_train, y_train, cat_features=cat_features_idx)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("ROC_AUC:", roc_auc_score(y_test, y_proba))

    return model



def main():

    print("Началось")

    user_data_df = load_user_data()
    post_data_df = process_text_data(load_post_data())
    feed_data_df = process_timestamp(load_feed_data())

    print(user_data_df.head())
    print(post_data_df.head())
    print(feed_data_df.head())


    features = feed_data_df.merge(user_data_df, on="user_id", how="left")
    features = features.merge(post_data_df, on="post_id", how="left")

    features = features.drop_duplicates(subset=["user_id", "post_id"])
    features = features.drop(["action", "exp_group"], axis=1)

    print(features.head())

    X = features.drop(["target", "user_id", "post_id"], axis=1)
    y = features["target"]

    categorial_cols = ["hour_of_day", "day_of_week", "city", "country", "os", "source", "topic"]
    cat_features = [X.columns.get_loc(col) for col in categorial_cols]

    model = train_catboost(X, y, cat_features)

if __name__ == '__main__':

    main()