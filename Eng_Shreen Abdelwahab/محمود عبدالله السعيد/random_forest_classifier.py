import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================

print("=" * 60)
print("1. LOADING DATASET & TRAIN-TEST SPLIT")
print("=" * 60)

# تحميل بيانات Iris المتاحة في مكتبة scikit-learn
X, y = load_iris(return_X_y=True)

# تقسيم البيانات إلى بيانات تدريب (70%) وبيانات اختبار (30%) كما ورد في المحاضرة
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# ============================================================
# 2. DECISION TREE CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("2. DECISION TREE MODEL")
print("=" * 60)

# بناء وتدريب نموذج شجرة القرار مع تحديد أقصى عمق (max_depth) لتجنب الـ Overfitting
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

tree_accuracy = tree_model.score(X_test, y_test)
print(f"Decision Tree Accuracy on Test Set: {tree_accuracy * 100:.2f}%")


# ============================================================
# 3. RANDOM FOREST CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("3. RANDOM FOREST MODEL")
print("=" * 60)

# بناء وتدريب نموذج الغابة العشوائية بـ 100 شجرة (n_estimators)
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_features="sqrt", 
    bootstrap=True, 
    oob_score=True, 
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy on Test Set: {rf_accuracy * 100:.2f}%")
print(f"Random Forest OOB Score: {rf_model.oob_score_ * 100:.2f}%")


# ============================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("4. FEATURE IMPORTANCES (RANDOM FOREST)")
print("=" * 60)

feature_importances = rf_model.feature_importances_
for idx, importance in enumerate(feature_importances):
    print(f"Feature {idx + 1}: Importance: {importance * 100:.2f}%")
