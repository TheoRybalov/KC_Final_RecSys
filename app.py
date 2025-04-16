import os
import pickle
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

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

