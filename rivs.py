import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time

# Загрузка данных
data = pd.read_csv('MetOpt2/housing.csv')
X = data.drop(columns=['MEDV']).values  # Матрица признаков
y = data['MEDV'].values  # Целевая переменная

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.65, random_state=23)

# Нормализация данных
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
y_train = (y_train - y_train.mean()) / y_train.std()

# Инициализация параметров
alpha = np.zeros(X_train.shape[1])  # Начальные веса
grad = 2 * X_train.T @ (X_train @ alpha - y_train)  # Градиент
d = -grad  # Начальное направление
tolerance = 1e-6  # Допустимая погрешность
max_iter = 1000  # Максимальное число итераций

# Списки для сохранения метрик
losses = []
grad_norms = []
alpha_norms = []
times = []

# Время начала работы алгоритма
start_time = time()

for k in range(max_iter):
    # Сохранение текущих значений
    losses.append(np.linalg.norm(X_train @ alpha - y_train) ** 2)
    grad_norms.append(np.linalg.norm(grad))
    alpha_norms.append(np.linalg.norm(alpha))
    times.append(time() - start_time)
    
    # Проверка условия остановки
    if np.linalg.norm(grad) < tolerance:
        print(f"Сходимость достигнута за {k} итераций.")
        break
    
    # Вычисление оптимального шага beta
    Ad = 2 * X_train.T @ (X_train @ d)
    beta = -(grad.T @ d) / (d.T @ Ad)
    
    # Обновление весов
    alpha = alpha + beta * d
    
    # Вычисление нового градиента
    grad_new = 2 * X_train.T @ (X_train @ alpha - y_train)
    
    # Вычисление коэффициента gamma
    gamma = np.linalg.norm(grad_new) ** 2 / np.linalg.norm(grad) ** 2
    
    # Обновление направления
    d = -grad_new + gamma * d
    
    # Обновление градиента
    grad = grad_new

# Построение графиков
plt.figure(figsize=(15, 5))

# График убывания функции
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title("Убывание функции")
plt.xlabel("Итерации")
plt.ylabel("Значение функции")

# График нормы градиента
plt.subplot(1, 3, 2)
plt.plot(grad_norms)
plt.title("Норма градиента")
plt.xlabel("Итерации")
plt.ylabel("Норма градиента")

# График нормы вектора весов
plt.subplot(1, 3, 3)
plt.plot(alpha_norms)
plt.title("Норма вектора весов")
plt.xlabel("Итерации")
plt.ylabel("Норма весов")

plt.tight_layout()
plt.show()

# Вывод результатов
print(f"Количество итераций: {k}")
print(f"Время работы: {time() - start_time:.4f} секунд")