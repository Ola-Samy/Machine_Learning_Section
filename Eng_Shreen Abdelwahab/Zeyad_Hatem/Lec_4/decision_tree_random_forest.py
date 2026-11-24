
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def entropy(labels):
    p = pd.Series(labels).value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())

def gini(labels):
    p = pd.Series(labels).value_counts(normalize=True).values
    return float(1 - (p ** 2).sum())

def info_gain(df, feature, target):
    parent = entropy(df[target])
    weighted = sum(len(g) / len(df) * entropy(g[target]) for _, g in df.groupby(feature))
    return parent - weighted


students = pd.DataFrame({
    "Study":      ["High", "High", "High", "Low", "Low", "Low", "Low", "High"],
    "Attendance": ["Good", "Good", "Poor", "Good", "Poor", "Poor", "Good", "Poor"],
    "Homework":   ["Done", "Missed", "Done", "Done", "Missed", "Done", "Missed", "Done"],
    "Result":     ["Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Fail", "Pass"],
})
print("1) 8 students: 5 Pass, 3 Fail")
print(f"   Entropy = {entropy(students.Result):.3f}  (lecture 0.954)")
print(f"   Gini    = {gini(students.Result):.3f}  (lecture 0.469)")
print("   Information gain per feature:")
for f in ["Study", "Attendance", "Homework"]:
    print(f"     Gain({f:10s}) = {info_gain(students, f, 'Result'):.3f}")
print("   -> highest gain is the root (lecture: Study 0.548, tiny difference is rounding)")

Xs = pd.get_dummies(students.drop(columns="Result"))
tree = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(Xs, students.Result)
print("\n   Learned tree:\n" + export_text(tree, feature_names=list(Xs.columns)))

new = pd.DataFrame([
    {"Study": "Low",  "Attendance": "Poor", "Homework": "Done"},   
    {"Study": "Low",  "Attendance": "Good", "Homework": "Done"},   
    {"Study": "High", "Attendance": "Poor", "Homework": "Missed"},  
])
Xnew = pd.get_dummies(new).reindex(columns=Xs.columns, fill_value=0)
for name, pred in zip("ABC", tree.predict(Xnew)):
    print(f"   Instance {name} -> {pred}")

tennis = pd.DataFrame({
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast",
                "Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temp":    ["Hot","Hot","Hot","Mild","Cool","Cool","Cool",
                "Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity":["High","High","High","High","Normal","Normal","Normal",
                "High","Normal","Normal","Normal","High","Normal","High"],
    "Wind":    ["Weak","Strong","Weak","Weak","Weak","Strong","Strong",
                "Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "Play":    ["No","No","Yes","Yes","Yes","No","Yes",
                "No","Yes","Yes","Yes","Yes","Yes","No"],
})
print("\n2) Play Tennis (9 Yes / 5 No)")
print(f"   Parent entropy = {entropy(tennis.Play):.3f}  (lecture 0.940)")
for f in ["Outlook", "Temp", "Humidity", "Wind"]:
    print(f"   Gain({f:8s}) = {info_gain(tennis, f, 'Play'):.3f}")
print("   -> Outlook is the root (lecture 0.247)")

Xt = pd.get_dummies(tennis.drop(columns="Play"))
tt = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(Xt, tennis.Play)
print("\n" + export_text(tt, feature_names=list(Xt.columns)))
print("   Training accuracy:", tt.score(Xt, tennis.Play))

print("\n3) Overfitting vs depth (Iris, 70/30 split)")
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
print("   depth  train   test")
for d in [1, 2, 3, 5, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xtr, ytr)
    print(f"   {str(d):5s}  {t.score(Xtr, ytr):.3f}   {t.score(Xte, yte):.3f}")

print("\n4) Decision tree vs Random Forest (Iris)")
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42).fit(Xtr, ytr)
print(f"   Decision tree test accuracy : {dt.score(Xte, yte):.3f}")
print(f"   Random forest test accuracy : {rf.score(Xte, yte):.3f}")
print(f"   Random forest OOB accuracy  : {rf.oob_score_:.3f}  (free validation, no test set needed)")
names = load_iris().feature_names
for n_, imp in sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1]):
    print(f"   importance {n_:20s} {imp:.3f}")


plt.figure(figsize=(9, 5))
plot_tree(tree, feature_names=list(Xs.columns), class_names=list(tree.classes_), filled=True)
plt.title("Tree learned from the 8 students"); plt.tight_layout(); plt.savefig("tree.png", dpi=120)
print("\nSaved tree.png")
