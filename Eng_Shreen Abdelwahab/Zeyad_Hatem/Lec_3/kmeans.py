
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

print("1) Distances between x=(1,1) and y=(4,5)")
a, b = np.array([1, 1]), np.array([4, 5])
print("   Euclidean:", np.sqrt(np.sum((a - b) ** 2)))
print("   Manhattan:", np.sum(np.abs(a - b)))
print("   Cosine   :", round(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), 4))

X = np.array([[1, 1], [1.5, 2], [3, 3.5], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
names = [f"P{i}" for i in range(1, 8)]
centroids = X[[2, 3]].copy()                    
print("\n2) Lloyd's algorithm, K=2, start c1=P3, c2=P4")

def kmeans_manual(X, centroids, max_iter=20):
    labels = None
    for it in range(1, max_iter + 1):
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  
        new_labels = d.argmin(axis=1)                                      
        print(f"  iter {it}: clusters =",
              {f"C{k+1}": [names[i] for i in np.where(new_labels == k)[0]] for k in range(len(centroids))})
        if labels is not None and np.array_equal(new_labels, labels):
            print("  -> assignments did not change: converged")
            break
        labels = new_labels
        centroids = np.array([X[labels == k].mean(axis=0) for k in range(len(centroids))])  
        print("           centroids =", np.round(centroids, 2).tolist())
    return labels, centroids

labels, centroids = kmeans_manual(X, centroids)
J = sum(np.sum((X[labels == k] - centroids[k]) ** 2) for k in range(2))
print(f"\n  Final centroids = {np.round(centroids, 2).tolist()}   J (WCSS) = {J:.2f}   (lecture: 10.71)")

print("\n3) scikit-learn KMeans on the same points")
km = KMeans(n_clusters=2, init=X[[2, 3]], n_init=1).fit(X)
print("  labels   :", km.labels_)
print("  centroids:", np.round(km.cluster_centers_, 2).tolist())
print("  inertia  :", round(km.inertia_, 2))
print("  silhouette (mean):", round(silhouette_score(X, km.labels_), 3))
print("  silhouette of P1 :", round(silhouette_samples(X, km.labels_)[0], 3))

print("\n4) Choosing K on a synthetic dataset with 3 true groups")
Xb, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=7)
ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(Xb)
    inertias.append(m.inertia_); sils.append(silhouette_score(Xb, m.labels_))
    print(f"  K={k}  inertia={m.inertia_:8.1f}  silhouette={sils[-1]:.3f}")
best_k = list(ks)[int(np.argmax(sils))]
print("  Best K by silhouette:", best_k)

print("\n5) Initialization: random vs k-means++ (inertia over 10 seeds, n_init=1)")
for init in ("random", "k-means++"):
    vals = [KMeans(3, init=init, n_init=1, random_state=s).fit(Xb).inertia_ for s in range(10)]
    print(f"  {init:10s} mean={np.mean(vals):.1f}  worst={np.max(vals):.1f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", s=70)
ax[0].scatter(*centroids.T, marker="X", c="k", s=150, label="centroids")
for p, n_ in zip(X, names): ax[0].annotate(n_, p + 0.08)
ax[0].set_title(f"7 points, K=2 (J={J:.2f})"); ax[0].legend()
ax[1].plot(list(ks), inertias, "o-"); ax[1].set_title("Elbow method"); ax[1].set_xlabel("K"); ax[1].set_ylabel("Inertia")
ax[2].plot(list(ks), sils, "o-g"); ax[2].set_title("Silhouette score"); ax[2].set_xlabel("K")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig("kmeans.png", dpi=120)
print("\nSaved kmeans.png")
