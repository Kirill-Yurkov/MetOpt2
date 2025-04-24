import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# === 1. Загрузка данных ===
data = pd.read_csv("MetOpt2/housing.csv")
X = data.drop(columns=["MEDV"]).values
y = data["MEDV"].values.reshape(-1, 1)

# === 2. Разделение на train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.65, random_state=23
)

# === 3. Масштабирование ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. Метод Хестенса-Штифеля ===
def hestenes_stiefel_cg(X, y, max_iter=1000, tol=0.001):
    alpha = np.zeros((X.shape[1], 1))
    r = X.T @ (X @ alpha - y)
    p = -r
    losses = []
    grad_norms = []
    alpha_norms = []

    for k in range(max_iter):
        Ap = X.T @ (X @ p)
        alpha_step = (r.T @ r) / (p.T @ Ap)
        alpha = alpha + alpha_step * p

        r_new = r + alpha_step * Ap
        beta = (r_new.T @ (r_new - r)) / (p.T @ (r_new - r))

        p = -r_new + beta * p
        r = r_new

        loss = np.linalg.norm(X @ alpha - y) ** 2
        grad_norm = np.linalg.norm(r)
        alpha_norm = np.linalg.norm(alpha)

        losses.append(loss)
        grad_norms.append(grad_norm)
        alpha_norms.append(alpha_norm)

        if grad_norm < tol:
            break

    return alpha, losses, grad_norms, alpha_norms

# Обучение модели
alpha_opt, losses, grad_norms, alpha_norms = hestenes_stiefel_cg(X_train_scaled, y_train)

# === 5. Тестирование ===
y_pred = X_test_scaled @ alpha_opt
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# График истинных vs. предсказанных значений
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title("Истинные vs. Предсказанные значения")
plt.show()

# Графики процесса обучения
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title("Убывание функции потерь")
plt.xlabel("Итерации")
plt.ylabel("Значение")

plt.subplot(1, 3, 2)
plt.plot(grad_norms)
plt.title("Норма градиента")
plt.xlabel("Итерации")
plt.ylabel("Норма")

plt.subplot(1, 3, 3)
plt.plot(alpha_norms)
plt.title("Норма весов")
plt.xlabel("Итерации")
plt.ylabel("Норма")

plt.tight_layout()
plt.show()