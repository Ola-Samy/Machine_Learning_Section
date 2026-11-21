
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

names = [f"P{i}" for i in range(1, 9)]
X = np.array([[1, 2], [2, 1], [2, 3], [3, 2], [6, 5], [7, 7], [8, 8], [7, 5]], dtype=float)  
y = np.array(["Fail"] * 4 + ["Pass"] * 4)


def knn_predict(X, y, q, k=3, weighted=False, verbose=False):
    d = np.sqrt(((X - q) ** 2).sum(axis=1))          
    order = np.argsort(d)[:k]                        
    if verbose:
        for i in order:
            print(f"     {names[i]} {tuple(X[i].tolist())}  d={d[i]:.2f}  {y[i]}")
    if weighted:                                   
        score = {}
        for i in order:
            score[y[i]] = score.get(y[i], 0) + 1 / d[i]
        return str(max(score, key=score.get)), score
    return str(Counter(y[order]).most_common(1)[0][0]), None


Q = np.array([5.0, 4.0])
print("1) Distances from Q = (5, 4)")
for n, p in zip(names, X):
    print(f"   {n} {tuple(p.tolist())} -> {np.sqrt(((p - Q) ** 2).sum()):.2f}")

print("\n2) K = 3 vote")
pred, _ = knn_predict(X, y, Q, k=3, verbose=True)
print("   Decision:", pred, "  (lecture: Pass, 2 of 3)")

print("\n3) What if we change K?  (lecture: 1,3,5 -> Pass; 7 -> Fail)")
for k in [1, 3, 5, 7]:
    print(f"   K={k}: {knn_predict(X, y, Q, k)[0]}")

print("\n4) Weighted KNN (w = 1/d), K = 3")
pred, score = knn_predict(X, y, Q, k=3, weighted=True)
print("   scores:", {str(c): round(float(s), 2) for c, s in score.items()}, "->", pred, "(lecture: Pass 1.16 > Fail 0.35, rounding)")

print("\n5) Min-max scaling: x' = (x - min) / (max - min)")
age = 25
print("   age 25, range 20..60 ->", (age - 20) / (60 - 20), "(lecture: 0.125)")
Xs = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print("   scaled P1..P3:", np.round(Xs[:3], 3).tolist())

print("\n6) Evaluation on 100 test students (TP=40, FN=10, FP=5, TN=45)")
TP, FN, FP, TN = 40, 10, 5, 45
acc = (TP + TN) / 100; prec = TP / (TP + FP); rec = TP / (TP + FN); f1 = 2 * prec * rec / (prec + rec)
print(f"   Accuracy={acc:.2f}  Precision={prec:.3f}  Recall={rec:.2f}  F1={f1:.3f}")

print("\n7) scikit-learn")
knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
print("   Q=(5,4) ->", knn.predict([Q])[0], knn.predict_proba([Q]))
knn_w = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, y)
print("   weighted ->", knn_w.predict([Q])[0])
print("   K=1 on the training data scores", KNeighborsClassifier(1).fit(X, y).score(X, y),
      "<- a trick: each point is its own neighbour, always test on unseen data")

print("\n8) Exercise: Q = (3, 3), K = 3")
Q2 = np.array([3.0, 3.0])
pred, _ = knn_predict(X, y, Q2, k=3, verbose=True)
print("   Decision:", pred)

plt.figure(figsize=(5.5, 5))
for c, col in [("Fail", "tab:red"), ("Pass", "tab:green")]:
    plt.scatter(*X[y == c].T, c=col, label=c, s=70)
for n, p in zip(names, X): plt.annotate(n, p + 0.12)
plt.scatter(*Q, marker="D", c="k", s=90, label="Q (5,4)")
plt.scatter(*Q2, marker="D", c="orange", s=90, label="Q (3,3)")
plt.xlabel("Study hours"); plt.ylabel("Attendance"); plt.legend(); plt.grid(alpha=.3)
plt.title("KNN: which group is the new student closest to?")
plt.tight_layout(); plt.savefig("knn.png", dpi=120)
print("\nSaved knn.png")
