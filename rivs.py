import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Загрузка данных
data = pd.read_csv('MetOpt2/housing.csv')
X = data.drop(columns=['MEDV']).values  # Матрица признаков
y = data['MEDV'].values  # Целевая переменная

# Разделение данных на train и test (65/35) с random_state=23
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=23)

# Нормализация данных
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Определение функции потерь и градиента
def f(X, y, alpha):
    """Функция потерь: ||X @ alpha - y||^2"""
    return np.linalg.norm(X @ alpha - y) ** 2

def grad_f(X, y, alpha):
    """Градиент функции потерь"""
    return 2 * X.T @ (X @ alpha - y)

# Реализация метода Флетчера-Ривса
# Реализация метода Флетчера-Ривса
# Реализация метода Флетчера-Ривса
def fletcher_reeves(X, y, tol=1e-6, max_iter=2000):
    """
    Метод Флетчера-Ривса для минимизации функционала.
    
    Параметры:
    - X: матрица признаков
    - y: целевая переменная
    - tol: допустимая погрешность
    - max_iter: максимальное количество итераций
    
    Возвращает:
    - alpha: найденный вектор весов
    - loss_history: история значений функции потерь
    - grad_norm_history: история нормы градиента
    - alpha_norm_history: история нормы вектора весов
    - num_iterations: количество выполненных итераций
    - execution_time: время выполнения
    """
    n = X.shape[1]
    alpha = np.zeros(n)  # Начальное приближение
    grad = grad_f(X, y, alpha)
    p = -grad  # Начальное направление
    loss_history = [f(X, y, alpha)]
    grad_norm_history = [np.linalg.norm(grad)]
    alpha_norm_history = [np.linalg.norm(alpha)]
    
    start_time = time.time()
    num_iterations = 0  # Счетчик итераций
    
    for i in range(max_iter):
        # Линейный поиск для определения шага beta
        Ap = X @ p
        alpha_p = np.dot(grad, grad) / np.dot(p, X.T @ Ap)  # Корректное вычисление шага
        
        # Обновление alpha
        alpha_new = alpha + alpha_p * p
        
        # Вычисление нового градиента
        grad_new = grad_f(X, y, alpha_new)
        grad_norm = np.linalg.norm(grad_new)
        
        # Обновление истории
        loss_history.append(f(X, y, alpha_new))
        grad_norm_history.append(grad_norm)
        alpha_norm_history.append(np.linalg.norm(alpha_new))
        
        # Увеличение счетчика итераций
        num_iterations += 1
        
        # Проверка условия остановки
        if grad_norm < tol:
            break
        
        # Вычисление коэффициента Флетчера-Ривса
        beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
        
        # Обновление направления
        p = -grad_new + beta * p
        alpha = alpha_new
        grad = grad_new
    
    execution_time = time.time() - start_time
    return alpha, loss_history, grad_norm_history, alpha_norm_history, num_iterations, execution_time

# Запуск метода Флетчера-Ривса
alpha, loss_history, grad_norm_history, alpha_norm_history, num_iterations, execution_time = fletcher_reeves(X_train, y_train)

# Вывод результатов
print(f"Количество итераций: {num_iterations}")
print(f"Время выполнения: {execution_time:.4f} секунд")
print(f"Найденный вектор весов: {alpha}")

# Построение графиков
plt.figure(figsize=(18, 6))

# График убывания функции потерь
plt.subplot(1, 3, 1)
plt.plot(loss_history, label="Loss", color="blue")
plt.title("Убывание функции потерь")
plt.xlabel("Итерации")
plt.ylabel("Значение функции потерь")
plt.yscale("log")  # Логарифмический масштаб для наглядности
plt.legend()

# График нормы градиента
plt.subplot(1, 3, 2)
plt.plot(grad_norm_history, label="Gradient Norm", color="orange")
plt.title("Убывание нормы градиента")
plt.xlabel("Итерации")
plt.ylabel("Норма градиента")
plt.yscale("log")  # Логарифмический масштаб для наглядности
plt.legend()

# График нормы вектора весов
plt.subplot(1, 3, 3)
plt.plot(alpha_norm_history, label="Alpha Norm", color="green")
plt.title("Изменение нормы вектора весов")
plt.xlabel("Итерации")
plt.ylabel("Норма вектора весов")
plt.legend()

plt.tight_layout()
plt.show()