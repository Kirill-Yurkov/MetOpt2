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
        print(f"Сходимость достигнута за {k} итераций.")
        break

    gamma_k = (grad.T @ grad) / (d.T @ (2 * X_train.T @ X_train @ d))
    alpha = alpha + gamma_k * d
    grad_new = 2 * X_train.T @ (X_train @ alpha - y_train)
    beta_k = (grad_new.T @ grad_new) / (grad.T @ grad)
    d = -grad_new + beta_k * d
    grad = grad_new

end_time = time.time()
print(f" Время выполнения алгоритма: {end_time - start_time:.4f} секунд")

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

X_test_normalized = (X_test - X_mean) / X_std
y_pred_normalized = X_test_normalized @ alpha
y_pred = y_pred_normalized * y_train_std + y_train_mean

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Коэффициент детерминации (R^2): {r2:.4f}")

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
