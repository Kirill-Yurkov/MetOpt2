import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# === 1. Загрузка данных ===
def load_data():
    df = pd.read_csv("housing.csv")
    X = df.drop(columns=["MEDV"]).values
    y = df["MEDV"].values.reshape(-1, 1)
    return X, y

X, y = load_data()

# === 2. Деление на train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=23)

# === 3. Масштабирование ===
y_train = y_train.ravel()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train shape:", y_train.shape)
# === 4. Определение общей функции потерь и градиента ===
def f(X, y, alpha):
    """Функция потерь: ||X @ alpha - y||^2"""
    return np.linalg.norm(X @ alpha - y) ** 2

def grad_f(X, y, alpha):
    """Градиент функции потерь"""
    grad = 2 * X.T @ (X @ alpha - y)
    return np.squeeze(grad) 

# === 5. Метод Хестениса–Штифеля ===
def hestenes_stiefel_cg(X, y, max_iter=100, tol=0.001):
    start_time = time.time()
    m = X.shape[1]  # Количество признаков
    alpha = np.zeros(m)  # Начальное приближение (вектор длины m)
    r = grad_f(X, y, alpha)  # Начальный градиент
    p = -r  # Начальное направление
    loss_values = []
    grad_norms = []
    alpha_norms = []
    iteration_count = 0  

    for k in range(max_iter):
        iteration_count += 1  
        
        # Проверка размерностей
        assert r.shape == (m,), f"r shape is {r.shape}, expected ({m},)"
        assert p.shape == (m,), f"p shape is {p.shape}, expected ({m},)"
        
        Ap = X.T @ (X @ p)  # Вычисление Ap
        if np.dot(p, Ap) == 0:
            print("Zero division error in alpha_step calculation.")
            break
        
        alpha_step = np.dot(r, r) / np.dot(p, Ap)  # Вычисление шага
        
        # Обновление alpha
        alpha_new = alpha + alpha_step * p
        
        # Вычисление нового градиента
        r_new = r + alpha_step * Ap
        grad_norm = np.linalg.norm(r_new)
        
        # Обновление истории
        loss_values.append(f(X, y, alpha_new))
        grad_norms.append(grad_norm)
        alpha_norms.append(np.linalg.norm(alpha_new))
        
        # Проверка условия остановки
        if grad_norm < tol:
            break
        
        # Вычисление коэффициента Хестениса–Штифеля
        beta = np.dot(r_new, (r_new - r)) / np.dot(p, (r_new - r))
        
        # Обновление направления
        p = -r_new + beta * p
        
        # Обновление переменных
        alpha = alpha_new
        r = r_new

    elapsed_time = time.time() - start_time
    return alpha, loss_values, grad_norms, alpha_norms, elapsed_time, iteration_count

def polak_ribiere_cg(X, y, tol=1e-6, max_iter=2000):
    n = X.shape[1]  # Количество признаков
    alpha = np.zeros(n)  # Начальное приближение (вектор длины n)
    grad = grad_f(X, y, alpha)  # Градиент функции потерь
    p = -grad  # Начальное направление (антиградиент)
    
    loss_history = [f(X, y, alpha)]  # История значений функции потерь
    grad_norm_history = [np.linalg.norm(grad)]  # История нормы градиента
    alpha_norm_history = [np.linalg.norm(alpha)]  # История нормы вектора весов
    
    start_time = time.time()
    num_iterations = 0  # Счетчик итераций

    for i in range(max_iter):
        # Линейный поиск для определения шага alpha_p
        Ap = X.T @ (X @ p)  # Вычисление Ap
        if np.dot(p, Ap) == 0:
            print("Zero division error in alpha_p calculation.")
            break
        alpha_p = np.dot(grad, grad) / np.dot(p, Ap)  # Вычисление шага
        
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
        
        # Вычисление коэффициента Полака-Рибьера
        beta = np.dot(grad_new, (grad_new - grad)) / np.dot(grad, grad)
        
        # Обновление направления
        p = -grad_new + beta * p
        
        # Обновление переменных
        alpha = alpha_new
        grad = grad_new

    execution_time = time.time() - start_time
    return alpha, loss_history, grad_norm_history, alpha_norm_history, num_iterations, execution_time

# === 7. Запуск методов и визуализация ===
def run_and_visualize(method, X_train, y_train, title_prefix=""):
    alpha_opt, losses, grad_norms, alpha_norms, iterations, execution_time = method(X_train, y_train)
    
    print(f"{title_prefix} Время выполнения алгоритма: {execution_time:.4f} секунд")
    print(f"{title_prefix} Количество итераций: {iterations}")
    
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    plt.plot(losses, label=f"{title_prefix} Loss", color="blue")
    plt.title(f"{title_prefix} Функция потерь $\\rho^2$")
    plt.xlabel("Итерация")
    plt.ylabel("Значение")
    plt.yscale("log")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(grad_norms, label=f"{title_prefix} Gradient Norm", color="orange")
    plt.title(f"{title_prefix} Норма градиента")
    plt.xlabel("Итерация")
    plt.ylabel("$\\|\\nabla \\rho^2\\|$")
    plt.yscale("log")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(alpha_norms, label=f"{title_prefix} Alpha Norm", color="green")
    plt.title(f"{title_prefix} Норма весов $\\|\\alpha\\|$")
    plt.xlabel("Итерация")
    plt.ylabel("Норма")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Запуск методов
print("=== Метод Хестениса–Штифеля ===")
run_and_visualize(hestenes_stiefel_cg, X_train_scaled, y_train, title_prefix="HS: ")

print("\n=== Метод Полака-Рибьера ===")
run_and_visualize(polak_ribiere_cg, X_train_scaled, y_train, title_prefix="PR: ")