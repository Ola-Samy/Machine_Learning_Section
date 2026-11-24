from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0],
    [5, 8],
    [6, 9],
    [5, 7]
])

# Create KMeans model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model
model.fit(X)

# Get cluster labels
labels = model.labels_

# Get cluster centers
centers = model.cluster_centers_

print("K-Means Results")
print("----------------")
print("Cluster Labels:", labels)
print("Cluster Centers:")
print(centers)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)

plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()