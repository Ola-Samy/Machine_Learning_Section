import numpy as np

# ============================================================
# 1. CLASSIFICATION METRICS
# ============================================================

# القيم من مصفوفة الارتباك المذكورة في المحاضرة (مثال سرطان الثدي)
TP = 90
FN = 10
FP = 5
TN = 895

# حساب إجمالي العينات
total_observations = TP + TN + FP + FN

# حساب مقاييس التصنيف باستخدام المعادلات المذكورة في المحاضرة
accuracy = (TP + TN) / total_observations
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# طباعة نتائج التصنيف بالإنجليزية فقط
print("=" * 60)
print("1. CLASSIFICATION METRICS")
print("=" * 60)
print(f"Total Observations: {total_observations}")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1_score * 100:.2f}%")


# ============================================================
# 2. REGRESSION METRICS
# ============================================================

# مصفوفات تجريبية للقيم الحقيقية والقيم المتوقعة
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# حساب مقاييس الانحدار: MAE, MSE, RMSE
mae = np.mean(np.abs(y_true - y_pred))
mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)

# طباعة نتائج الانحدار بالإنجليزية فقط
print("\n" + "=" * 60)
print("2. REGRESSION METRICS")
print("=" * 60)
print(f"Actual Values (y):    {y_true}")
print(f"Predicted Values (y_hat): {y_pred}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("=" * 60)