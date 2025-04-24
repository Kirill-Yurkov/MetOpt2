import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time  # Для измерения времени выполнения

# === 1. Загрузка данных ===
df = pd.read_csv("MetOpt2/housing.csv")
X = df.drop(columns=["MEDV"]).values
y = df["MEDV"].values.reshape(-1, 1)

# === 2. Деление на train/test ===
X_train, _, y_train, _ = train_test_split(X, y, train_size=0.65, random_state=23)

# === 3. Масштабирование ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# === 4. Метод Хестениса–Штифеля ===
def hestenes_stiefel_cg(X, y, max_iter=100, tol=0.001):
    start_time = time.time()  # Начало измерения времени
    
    alpha = np.zeros((X.shape[1], 1))
    r = X.T @ (X @ alpha - y)
    p = -r
    loss_values = []
    grad_norms = []
    alpha_norms = []
    iteration_count = 0  

    for k in range(max_iter):
        iteration_count += 1  
        
        Ap = X.T @ (X @ p)
        alpha_step = (r.T @ r) / (p.T @ Ap)
        alpha = alpha + alpha_step * p

        r_new = r + alpha_step * Ap
        beta = (r_new.T @ (r_new - r)) / (p.T @ (r_new - r))

        p = -r_new + beta * p
        r = r_new

        loss = 0.5 * np.linalg.norm(X @ alpha - y)**2
        grad_norm = np.linalg.norm(r)
        alpha_norm = np.linalg.norm(alpha)

        loss_values.append(loss)
        grad_norms.append(grad_norm)
        alpha_norms.append(alpha_norm)

        if grad_norm < tol:
            break

    elapsed_time = time.time() - start_time  # Вычисление затраченного времени
    return alpha, loss_values, grad_norms, alpha_norms, elapsed_time, iteration_count

# === 5. Запуск и визуализация ===
alpha_opt, losses, grad_norms, alpha_norms, execution_time, iterations = hestenes_stiefel_cg(X_train_scaled, y_train)

print(f"Время выполнения алгоритма: {execution_time:.4f} секунд")
print(f"Количество итераций: {iterations}")

plt.figure(figsize=(16, 4))

plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title("Функция потерь $\\rho^2$")
plt.xlabel("Итерация")
plt.ylabel("Значение")

plt.subplot(1, 3, 2)
plt.plot(grad_norms)
plt.title("Норма градиента")
plt.xlabel("Итерация")
plt.ylabel("$\\|\\nabla \\rho^2\\|$")

plt.subplot(1, 3, 3)
plt.plot(alpha_norms)
plt.title("Норма весов $\\|\\alpha\\|$")
plt.xlabel("Итерация")
plt.ylabel("Норма")

plt.tight_layout()
plt.show()