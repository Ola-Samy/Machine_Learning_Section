
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

x = np.array([2, 4, 6, 8, 10], dtype=float)      
y = np.array([40, 50, 65, 80, 90], dtype=float)  
n = len(x)

x_bar, y_bar = x.mean(), y.mean()
print(f"Step 1  x_bar = {x_bar}, y_bar = {y_bar}")

dx, dy = x - x_bar, y - y_bar
Sxy, Sxx = np.sum(dx * dy), np.sum(dx ** 2)
print(f"Step 2  sum((x-x_bar)(y-y_bar)) = {Sxy},  sum((x-x_bar)^2) = {Sxx}")

b1 = Sxy / Sxx
b0 = y_bar - b1 * x_bar
print(f"Step 3  slope b1 = {b1},  intercept b0 = {b0}   ->  y_hat = {b0} + {b1}x")

y_hat = b0 + b1 * x
res = y - y_hat
print("\nStep 4  x   y   y_hat  residual")
for xi, yi, yh, r in zip(x, y, y_hat, res):
    print(f"        {xi:<3.0f} {yi:<3.0f} {yh:<6.1f} {r:+.1f}")

mae = np.mean(np.abs(res))
mse = np.mean(res ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum(res ** 2)
ss_tot = np.sum((y - y_bar) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\nStep 5  MAE = {mae:.2f}  MSE = {mse:.2f}  RMSE = {rmse:.3f}  R2 = {r2:.4f}")
print(f"        (SS_res = {ss_res}, SS_tot = {ss_tot})")

x_new = 7
print(f"\nStudent who studies {x_new} h -> predicted grade = {b0 + b1 * x_new}")

model = LinearRegression().fit(x.reshape(-1, 1), y)
pred = model.predict(x.reshape(-1, 1))
print("\n--- scikit-learn ---")
print("intercept:", model.intercept_, " slope:", model.coef_[0])
print("predict 7 h:", model.predict([[7]])[0])
print("MAE:", mean_absolute_error(y, pred), " MSE:", mean_squared_error(y, pred),
      " R2:", round(r2_score(y, pred), 4))


plt.figure(figsize=(6, 4))
plt.scatter(x, y, label="actual", zorder=3)
xs = np.linspace(0, 12, 50)
plt.plot(xs, b0 + b1 * xs, "r", label=f"y_hat = {b0:g} + {b1:g}x")
for xi, yi, yh in zip(x, y, y_hat):
    plt.plot([xi, xi], [yi, yh], "gray", linestyle=":")   # residuals
plt.xlabel("Study hours"); plt.ylabel("Exam grade"); plt.legend(); plt.grid(alpha=.3)
plt.title("Simple linear regression")
plt.tight_layout(); plt.savefig("linear_regression.png", dpi=120)
print("\nSaved linear_regression.png")
