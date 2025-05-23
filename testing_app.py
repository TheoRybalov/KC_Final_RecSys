import requests
from datetime import datetime

url = "http://127.0.0.1:8899/post/recommendations/"
# Параметры запроса
params = {
    "id": 2331,  # ID пользователя
    "time": datetime(2020, 5, 17), 
    "limit": 5
}

# Отправка GET-запроса
response = requests.get(url, params=params)

# Проверка статуса и вывод результата
if response.status_code == 200:
    recommendations = response.json()
    print("Рекомендованные посты:", recommendations)
else:
    print("Ошибка:", response.status_code, response.text)