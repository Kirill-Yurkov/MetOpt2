import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Загрузка данных
data = pd.read_csv('housing.csv')
X = data.drop(columns=['MEDV']).values
y = data['MEDV'].values

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=23)

# Метод Полака-Рибьера
def polak_ribiere(X, y, alpha_init=None, max_iter=1000, tol=1e-6, learning_rate=0.01):
    # Инициализация параметров
    n_samples, n_features = X.shape
    if alpha_init is None:
        alpha = np.zeros(n_features)
    else:
        alpha = np.array(alpha_init)
    
    # История значений функционала
    history = []
    
    # Первая итерация
    grad_prev = None
    direction_prev = None
    
    for k in range(max_iter):
        # Вычисление текущего значения функционала
        residual = X @ alpha - y
        objective_value = 0.5 * np.linalg.norm(residual) ** 2
        history.append(objective_value)
        
        # Вычисление градиента
        grad = X.T @ residual
        
        # Проверка критерия остановки
        if np.linalg.norm(grad) < tol:
            print(f"Остановка: норма градиента {np.linalg.norm(grad)} < {tol}")
            break
        
        # Вычисление направления движения
        if k == 0:
            direction = -grad
        else:
            beta = (grad.T @ grad) / (grad_prev.T @ grad_prev)
            direction = -grad + beta * direction_prev
        
        # Обновление параметров
        alpha -= learning_rate * direction
        
        # Сохранение текущего градиента и направления для следующей итерации
        grad_prev = grad
        direction_prev = direction
    
    return alpha, history

# Запуск метода Полака-Рибьера
alpha_opt, history = polak_ribiere(X_train, y_train)

# Вывод результатов
print("Оптимальные параметры:", alpha_opt)
print("Значение функционала:", history[-1])