from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import time

data = pd.read_csv('housing.csv')
X = data.drop(columns=['MEDV']).values
y = data['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.65, random_state=23)

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train = (y_train - y_train_mean) / y_train_std

def shiphel(X_train, y_train):
    alpha = np.zeros(X_train.shape[1])
    grad = 2 * X_train.T @ (X_train @ alpha - y_train)
    d = -grad
    losses = []
    grad_norms = []
    alpha_norms = []
    start_time = time.time()  

    for k in range(100):
        losses.append(np.linalg.norm(X_train @ alpha - y_train) ** 2)
        grad_norms.append(np.linalg.norm(grad))
        alpha_norms.append(np.linalg.norm(alpha))
        if np.linalg.norm(grad) < 0.001:
            break
        A = 2 * X_train.T @ X_train 
        gamma_k = -(grad.T @ d) / (d.T @ A @ d)
        alpha = alpha + gamma_k * d

        grad_new = 2 * X_train.T @ (X_train @ alpha - y_train)
        beta_k = (grad_new.T @ (grad_new - grad)) / (d.T @ (grad_new - grad))
        d = -grad_new + beta_k * d
        grad = grad_new

    end_time = time.time() 
    execution_time=end_time-start_time
    return alpha, losses, grad_norms, alpha_norms, execution_time, k

alpha_shiphel, losses_shiphel, grad_norms_shiphel, alpha_norms_shiphel, execution_time_shiphel, iterations_shiphel = shiphel(X_train=X_train, y_train=y_train)
print(f"Время выполнения алгоритма: {execution_time_shiphel:.4f} секунд")
print(f"Количество итераций: {iterations_shiphel}")
def visualize_plots_params(losses, grad_norms, alpha_norms):
    # Графики метрик
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title("Убывание функции")
    plt.xlabel("Итерации")
    plt.ylabel("Значение функции")
    plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.plot(grad_norms)
    plt.title("Норма градиента")
    plt.xlabel("Итерации")
    plt.ylabel("Норма градиента")
    plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.plot(alpha_norms)
    plt.title("Норма вектора весов")
    plt.xlabel("Итерации")
    plt.ylabel("Норма весов")
    plt.grid(True)
    plt.show()

visualize_plots_params(losses_shiphel, grad_norms_shiphel, alpha_norms_shiphel)

def calculate_mse(alpha):
    X_test_normalized = (X_test - X_mean) / X_std
    y_pred_normalized = X_test_normalized @ alpha
    y_pred = y_pred_normalized * y_train_std + y_train_mean

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

mse_shiphel, r2_shiphel, y_pred_shiphel = calculate_mse(alpha_shiphel)
print(f"Среднеквадратичная ошибка (MSE): {mse_shiphel:.4f}")
print(f"Коэффициент детерминации (R^2): {r2_shiphel:.4f}")

def visualize_predictions(y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Предсказания')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Идеальное совпадение")
    plt.xlabel("Истинные значения MEDV")
    plt.ylabel("Предсказанные значения MEDV")
    plt.title("Сравнение предсказаний и настоящих значений")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

visualize_predictions(y_pred_shiphel)

def rivs(X_train, y_train):
    alpha = np.zeros(X_train.shape[1])
    grad = 2 * X_train.T @ (X_train @ alpha - y_train)
    d = -grad
    losses = []
    grad_norms = []
    alpha_norms = []
    start_time = time.time()

    for k in range(100):
        losses.append(np.linalg.norm(X_train @ alpha - y_train) ** 2)
        grad_norms.append(np.linalg.norm(grad))
        alpha_norms.append(np.linalg.norm(alpha))
        if np.linalg.norm(grad) < 0.001:
            break

        gamma_k = (grad.T @ grad) / (d.T @ (2 * X_train.T @ X_train @ d))
        alpha = alpha + gamma_k * d
        grad_new = 2 * X_train.T @ (X_train @ alpha - y_train)
        beta_k = (grad_new.T @ grad_new) / (grad.T @ grad)
        d = -grad_new + beta_k * d
        grad = grad_new

    end_time = time.time()
    execution_time=end_time-start_time
    return alpha, losses, grad_norms, alpha_norms, execution_time, k

alpha_rivs, losses_rivs, grad_norms_rivs, alpha_norms_rivs, execution_time_rivs, iterations_rivs=rivs(X_train=X_train, y_train=y_train)

print(f" Время выполнения алгоритма: {execution_time_rivs:.4f} секунд")
print(f"Количество итераций: {iterations_rivs}")

visualize_plots_params(losses_rivs, grad_norms_rivs, alpha_norms_rivs)

mse_rivs, r2_rivs, y_pred_rivs =calculate_mse(alpha_rivs)

print(f"Среднеквадратичная ошибка (MSE): {mse_rivs:.4f}")
print(f"Коэффициент детерминации (R^2): {r2_rivs:.4f}")

visualize_predictions(y_pred_rivs)

tabledata=[['Хестенса-Штифеля', execution_time_shiphel, iterations_shiphel, mse_shiphel, r2_shiphel], ['Флетчера-Ривса', execution_time_rivs, iterations_rivs, mse_rivs, r2_rivs]]
pd.DataFrame(tabledata, columns=["Алгоритм","Время выполнения (с)", "Количество итераций", "Среднеквадратичная ошибка (MSE)", "Коэффициент детерминации"])