
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, mean_squared_error,
                             mean_absolute_error)

print("=" * 60, "\n1) Supervised vs Unsupervised (Iris)\n" + "=" * 60)
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)


clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
print("Supervised (labels used)   -> test accuracy:", round(clf.score(Xte, yte), 3))


km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("Unsupervised (no labels)   -> cluster sizes:", np.bincount(km.labels_))

print("\n" + "=" * 60, "\n2) Classification metrics (cancer example)\n" + "=" * 60)
TP, FN, FP, TN = 90, 10, 5, 895         
print(f"Confusion matrix:\n            Pred+  Pred-\n  Actual+  {TP:5d}  {FN:5d}\n  Actual-  {FP:5d}  {TN:5d}")

accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
print(f"Accuracy  = (TP+TN)/total = {accuracy:.3f}   (lecture: 98.5%)")
print(f"Precision = TP/(TP+FP)    = {precision:.3f}   (lecture: 94.7%)")
print(f"Recall    = TP/(TP+FN)    = {recall:.3f}   (lecture: 90.0%)")
print(f"F1-Score                  = {f1:.3f}   (lecture: 92.2%)")

y_true = np.array([1] * (TP + FN) + [0] * (FP + TN))
y_pred = np.array([1] * TP + [0] * FN + [1] * FP + [0] * TN)
print("\nscikit-learn check:")
print(confusion_matrix(y_true, y_pred, labels=[1, 0]))
print(f"accuracy={accuracy_score(y_true, y_pred):.3f} precision={precision_score(y_true, y_pred):.3f} "
      f"recall={recall_score(y_true, y_pred):.3f} f1={f1_score(y_true, y_pred):.3f}")

print("\n" + "=" * 60, "\n3) Why accuracy can lie (95 / 5 split)\n" + "=" * 60)
y_true = np.array([0] * 95 + [1] * 5)
y_pred = np.zeros(100, dtype=int)        
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Recall on minority class:", recall_score(y_true, y_pred))

print("\n" + "=" * 60, "\n4) Regression metrics\n" + "=" * 60)
y_true = np.array([40, 50, 65, 80, 90], dtype=float)
y_pred = np.array([39, 52, 65, 78, 91], dtype=float)   
errors = y_true - y_pred
mse  = np.mean(errors ** 2)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(errors))
print("errors (y - y_hat):", errors)
print(f"MSE  = mean(error^2)  = {mse:.3f}")
print(f"RMSE = sqrt(MSE)      = {rmse:.3f}   (same unit as y)")
print(f"MAE  = mean(|error|)  = {mae:.3f}   (less sensitive to outliers)")
print("sklearn check:", mean_squared_error(y_true, y_pred),
      round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
      mean_absolute_error(y_true, y_pred))

y_pred_out = y_pred.copy(); y_pred_out[-1] = 60
print("\nWith one big mistake (91 -> 60):")
print(f"MAE  = {mean_absolute_error(y_true, y_pred_out):.2f}   "
      f"MSE = {mean_squared_error(y_true, y_pred_out):.2f}   <- MSE punishes large errors harder")
