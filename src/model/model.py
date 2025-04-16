import os
import pandas as pd
from sqlalchemy import create_engine
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit





csv_path = os.path.join(os.path.dirname(__file__), '../../data/merged_data.csv')

# Загружаем CSV
merged_df = pd.read_csv(csv_path, sep=";")


X = merged_df.drop(["target"], axis=1)
y = merged_df["target"]

scale_pos_weight = sum(y == 0) / sum(y == 1)
print(scale_pos_weight)

categorial_cols = ["hour_of_day", "day_of_week", "city", "country", "exp_group", "os", "source", "topic"]

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations = 1000,
                           depth = 6,
                           learning_rate= 0.1,
                           scale_pos_weight = scale_pos_weight)

model.fit(X_train, y_train, cat_features=categorial_cols)

print("Accuracy: ", accuracy_score(y_test, model.predict(X_test)))
print("Precision: ", precision_score(y_test, model.predict(X_test)))
print("Recall:", recall_score(y_test, model.predict(X_test)))
print("ROC_AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


output_dir = os.path.join(os.path.dirname(__file__), '../../src/model')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'catboost_model.cbm')

# Сохраняем модель с помощью pickle
model.save_model(output_path, format="cbm")
