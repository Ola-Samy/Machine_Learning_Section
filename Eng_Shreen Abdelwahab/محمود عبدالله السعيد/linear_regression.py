import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. DATASET SETUP (Study Hours vs Exam Grade)
# ============================================================

# بيانات الدراسة من المحاضرة (عدد ساعات الدراسة مقابل درجات الامتحان)
x = np.array([2, 4, 6, 8, 10])
y = np.array([40, 50, 65, 80, 90])

n = len(x)
print("=" * 60)
print("1. DATASET & LEAST SQUARES COMPUTATION")
print("=" * 60)
print(f"Number of data points (n): {n}")
print(f"Study Hours (x): {x}")
print(f"Exam Grades (y): {y}")


# ============================================================
# 2. CALCULATING MEANS, SLOPE, AND INTERCEPT
# ============================================================

x_mean = np.mean(x)
y_mean = np.mean(y)

# حساب المعاملات رياضياً باستخدام طريقة المربعات الصغرى (Least Squares)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

beta_1 = numerator / denominator  # الميل (Slope)
beta_0 = y_mean - (beta_1 * x_mean)  # نقطة التقاطع (Intercept)

print(f"\nMean of x: {x_mean}")
print(f"Mean of y: {y_mean}")
print(f"Calculated Slope (beta_1): {beta_1}")
print(f"Calculated Intercept (beta_0): {beta_0}")
print(f"Fitted Equation: y_hat = {beta_0} + {beta_1} * x")


# ============================================================
# 3. PREDICTIONS AND RESIDUALS
# ============================================================

# حساب القيم المتوقعة (y_hat) والأخطاء (Residuals)
y_pred = beta_0 + beta_1 * x
residuals = y - y_pred
absolute_errors = np.abs(residuals)
squared_errors = residuals ** 2

print("\n" + "=" * 60)
print("2. PREDICTIONS & RESIDUALS")
print("=" * 60)
for i in range(n):
    print(f"x = {x[i]} | Actual y = {y[i]} | Predicted y_hat = {y_pred[i]:.2f} | Residual = {residuals[i]:.2f}")


# ============================================================
# 4. EVALUATION METRICS (MAE, MSE, RMSE, R^2)
# ============================================================

mae = np.mean(absolute_errors)
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)

# حساب التباين الكلي (SST) لحساب معامل التحديد R^2
sst = np.sum((y - y_mean) ** 2)
sse = np.sum(squared_errors)
r2 = 1 - (sse / sst)

print("\n" + "=" * 60)
print("3. EVALUATION METRICS")
print("=" * 60)
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")


# ============================================================
# 5. SCIKIT-LEARN IMPLEMENTATION VERIFICATION
# ============================================================

# تجهيز البيانات لشكل مصفوفة ثنائية الأبعاد لتتوافق مع مكتبة scikit-learn
X_skl = x.reshape(-1, 1)

model = LinearRegression().fit(X_skl, y)
skl_pred = model.predict(X_skl)

print("\n" + "=" * 60)
print("4. SCIKIT-LEARN VERIFICATION")
print("=" * 60)
print(f"Scikit-learn Intercept: {model.intercept_}")
print(f"Scikit-learn Coefficient (Slope): {model.coef_[0]}")
print(f"Prediction for 7 study hours: {model.predict([[7]])[0]:.2f}")
print(f"Scikit-learn MSE: {mean_squared_error(y, skl_pred)}")
print(f"Scikit-learn R2 Score: {r2_score(y, skl_pred):.4f}")
print("=" * 60)