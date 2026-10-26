import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# 1. DATA PREPARATION & SCALING
# ============================================================

# قراءة بيانات تجريبية أو مجموعة بيانات العملاء (Wholesale customers dataset)
# يمكنك استبدال مسار الملف بملف البيانات الخاص بك
print("=" * 60)
print("1. DATA PREPROCESSING & FEATURE SCALING")
print("=" * 60)

# بيانات افتراضية توضيحية في حال عدم توفر الملف الخارجي
np.random.seed(42)
data = np.random.rand(100, 4) * 100 
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4'])

# توحيد مقياس البيانات باستخدام StandardScaler (Z-score scaling) كما شرحت المحاضرة
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print(f"Dataset shape: {X_scaled.shape}")
print("Features successfully scaled using Z-score standardization.")


# ============================================================
# 2. K-MEANS CLUSTERING IMPLEMENTATION
# ============================================================

# تطبيق خوارزمية K-means بـ K = 3 مع استخدام k-means++ للتثبيت
k_optimal = 3
kmeans_model = KMeans(
    n_clusters=k_optimal, 
    init='k-means++', 
    n_init=10, 
    random_state=42
)

# تدريب النموذج وتوقع المجموعات (Labels)
cluster_labels = kmeans_model.fit_predict(X_scaled)

print("\n" + "=" * 60)
print("2. K-MEANS MODEL RESULTS")
print("=" * 60)
print(f"Chosen K: {k_optimal}")
print(f"Inertia (WCSS / J): {kmeans_model.inertia_:.4f}")
print(f"Cluster Centers (Centroids):\n{kmeans_model.cluster_centers_}")


# ============================================================
# 3. CHOOSING K USING SILHOUETTE SCORE
# ============================================================

print("\n" + "=" * 60)
print("3. EVALUATING K USING SILHOUETTE SCORE")
print("=" * 60)

best_silhouette = -1
best_k = 2

# التجربة لمجرد البحث عن أفضل K بين 2 و 6
for k in range(2, 7):
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels_temp = km_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_temp)
    print(f"For K = {k}, The Average Silhouette Score is: {score:.4f}")
    
    if score > best_silhouette:
        best_silhouette = score
        best_k = k

print(f"\nOptimal K based on Silhouette Score: {best_k} (Score: {best_silhouette:.4f})")
print("=" * 60) 