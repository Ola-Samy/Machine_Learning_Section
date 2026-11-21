
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

w, b = np.array([2.0, 1.0]), -5.0
x = np.array([3.0, 2.0])
f = w @ x + b
print(f"1) f(x) = w.x + b = 2*3 + 1*2 - 5 = {f:.0f} -> {'pass (+1)' if f > 0 else 'fail (-1)'}")

X = np.array([[1, 1], [2, 2], [4, 4], [5, 3]], dtype=float)  
y = np.array([-1, -1, 1, 1])
names = list("ABCD")

w, b = np.array([0.5, 0.5]), -3.0     
f = X @ w + b
print("\n2) w = (0.5, 0.5), b = -3")
for n, fi, yi in zip(names, f, y):
    print(f"   {n}: f = {fi:+.1f}   y*f = {yi * fi:.1f}   {'<- on the street edge (support vector)' if abs(yi * fi - 1) < 1e-9 else ''}")
print(f"   ||w|| = {np.linalg.norm(w):.2f}   margin width = 2/||w|| = {2 / np.linalg.norm(w):.2f}   (lecture 2.83)")

svm = SVC(kernel="linear", C=1e6).fit(X, y)
print("\n   sklearn SVC(linear):  w =", np.round(svm.coef_[0], 3), " b =", np.round(svm.intercept_[0], 3))
print("   support vectors found by sklearn:", [names[i] for i in svm.support_], svm.support_vectors_.tolist())
print("   (D also sits exactly on the +1 edge: f(D) = 1. With 3 points on the edges, only 2 are needed to fix the line, so sklearn keeps B and C.)")
print("   predict (3,3):", svm.predict([[3, 3]])[0])

print("\n3) Pick the smallest ||w|| that still obeys y*f >= 1")
for wc in [(1, 1), (0.5, 0.5), (0.25, 0.25)]:
    wc = np.array(wc); bc = -6 * wc[0]           
    yf = y * (X @ wc + bc)
    ok = np.all(yf >= 1 - 1e-9)
    print(f"   w={tuple(wc.tolist())}  1/2||w||^2={0.5 * wc @ wc:.3f}  width={2 / np.linalg.norm(wc):.2f}  valid? {'yes' if ok else 'NO (breaks y*f>=1)'}")

print("\n4) Soft margin: slack xi = max(0, 1 - y*f)")
for yf in [1.4, 0.7, 0.0, -0.2]:
    print(f"   y*f = {yf:+.1f} -> xi = {max(0, 1 - yf):.1f}")
margin_part, sum_xi = 0.5 * 0.5, 0.3
for C in [0.1, 10, 100]:
    print(f"   C = {C:<5} penalty = C*sum(xi) = {C * sum_xi:g}   total cost = {margin_part + C * sum_xi:g}")
print("   (lecture: C=10 -> 0.25 + 3 = 3.25)")

print("\n5) Kernels")
xz, zz = np.array([1, 2]), np.array([2, 1])
print("   Polynomial (x.z + 1)^2 for (1,2),(2,1) =", (xz @ zz + 1) ** 2, "(lecture 25)")
def rbf(x, z, gamma): return np.exp(-gamma * np.sum((np.array(x) - np.array(z)) ** 2))
print("   RBF gamma=0.5, d^2=1 :", round(np.exp(-0.5 * 1), 3), "(0.607)   gamma=2, d^2=1:", round(np.exp(-2 * 1), 3), "(0.135)")
print("   RBF gamma=0.25, x=(1,1), z=(3,1):", round(rbf([1, 1], [3, 1], 0.25), 3), "(0.368)")

print("\n6) Effect of C and gamma (moons dataset, RBF kernel)")
Xm, ym = make_moons(n_samples=300, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=0)
print("   kernel  C      gamma  train  test   n_support")
for kernel, C, g in [("linear", 1, "scale"), ("rbf", 1, 0.5), ("rbf", 0.1, 0.5), ("rbf", 100, 0.5),
                     ("rbf", 1, 0.05), ("rbf", 1, 50)]:
    m = SVC(kernel=kernel, C=C, gamma=g).fit(Xtr, ytr)
    print(f"   {kernel:6s} {C:<6} {str(g):6s} {m.score(Xtr, ytr):.3f}  {m.score(Xte, yte):.3f}  {m.n_support_.sum()}")
print("   small C = wide margin (may underfit); big C / big gamma = tight fit (may overfit)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

a = ax[0]
a.scatter(*X[y == -1].T, c="tab:red", s=80, label="fail"); a.scatter(*X[y == 1].T, c="tab:green", s=80, label="pass")
for n, p in zip(names, X): a.annotate(n, p + 0.1)
xs = np.linspace(0, 6, 10)
for off, ls in [(0, "-"), (1, "--"), (-1, "--")]:
    a.plot(xs, (off - b - w[0] * xs) / w[1], "k", linestyle=ls, lw=1)
a.scatter(*svm.support_vectors_.T, s=250, facecolors="none", edgecolors="b", label="support vectors")
a.set_xlim(0, 6); a.set_ylim(0, 6); a.legend(); a.set_title("Margin and support vectors"); a.grid(alpha=.3)

gx, gy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2.0, 200))
for a, g in zip(ax[1:], [0.5, 50]):
    m = SVC(kernel="rbf", C=1, gamma=g).fit(Xtr, ytr)
    Z = m.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    a.contourf(gx, gy, Z, alpha=.25, cmap="coolwarm"); a.scatter(*Xtr.T, c=ytr, cmap="coolwarm", s=15, edgecolors="k", linewidths=.3)
    a.set_title(f"RBF gamma={g}  (test acc {m.score(Xte, yte):.2f})")
plt.tight_layout(); plt.savefig("svm.png", dpi=110)
print("\nSaved svm.png")
