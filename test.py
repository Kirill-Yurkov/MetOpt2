import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Загрузка данных
data = pd.read_csv('housing.csv')

# 2. Разделение на матрицу признаков и целевую переменную
X = data.drop(columns=['MEDV']).values  # Матрица признаков
y = data['MEDV'].values  # Целевая переменная

# 3. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.65, random_state=35
)

# 4. Проверка размеров
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")