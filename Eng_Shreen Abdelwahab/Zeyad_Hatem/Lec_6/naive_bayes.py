
import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("1) Bayes: P(Spam | free) = P(free | Spam) P(Spam) / P(free)")
p_spam, p_free_spam = 0.40, 0.60
p_free = 30 / 100
print("   P(Spam | free) =", round(p_free_spam * p_spam / p_free, 2), "  (lecture: 0.80)")

train = pd.DataFrame({
    "free":  ["Yes", "Yes", "Yes", "No", "No", "Yes", "No", "No"],
    "offer": ["Yes", "No", "Yes", "Yes", "No", "No", "No", "Yes"],
    "label": ["Spam"] * 4 + ["Ham"] * 4,
})


def naive_bayes(df, x, alpha=0.0, target="label"):
    classes = df[target].unique()
    scores = {}
    for c in classes:
        sub = df[df[target] == c]
        score = len(sub) / len(df)                                  
        for f, v in x.items():                                        
            V = df[f].nunique()
            score *= ((sub[f] == v).sum() + alpha) / (len(sub) + alpha * V)
        scores[c] = score                                             
    total = sum(scores.values())
    return scores, {c: s / total for c, s in scores.items()}          


print("\n2) Spam filter: new email free=Yes, offer=Yes")
scores, post = naive_bayes(train, {"free": "Yes", "offer": "Yes"})
print("   scores   :", {c: round(float(s), 5) for c, s in scores.items()}, "(lecture 0.28125 / 0.03125)")
print("   posterior:", {c: round(float(p), 2) for c, p in post.items()}, "(lecture: Spam 0.90)")

print("\n3) Laplace smoothing: P = (count + alpha) / (N_C + alpha*V)")
print("   word never seen in 4 spam emails:  without =", 0 / 4, "  with alpha=1 =", round((0 + 1) / (4 + 2), 3), "(lecture 0.167)")
x_new = {"free": "No", "offer": "No"}
print("   no smoothing :", {c: round(float(s), 4) for c, s in naive_bayes(train, x_new)[0].items()})
print("   alpha = 1    :", {c: round(float(s), 4) for c, s in naive_bayes(train, x_new, alpha=1)[0].items()})

print("\n4) Gaussian Naive Bayes: study hours = 6")
pass_h = np.array([5, 6, 7, 8, 9]); fail_h = np.array([0, 2, 3, 4, 6])


def gauss(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


stats = {}
for name, h in [("Pass", pass_h), ("Fail", fail_h)]:
    stats[name] = (h.mean(), h.var())                    
    print(f"   {name}: mean={h.mean():.0f}  variance={h.var():.0f}")
lik = {c: gauss(6, *stats[c]) for c in stats}
sc = {c: 0.5 * lik[c] for c in stats}
tot = sum(sc.values())
print("   likelihoods:", {c: round(float(v), 4) for c, v in lik.items()}, "(lecture 0.2197 / 0.0648)")
print("   posterior  :", {c: round(float(v / tot), 3) for c, v in sc.items()}, "(lecture: Pass 77.2%)")

Xg = np.concatenate([pass_h, fail_h]).reshape(-1, 1); yg = ["Pass"] * 5 + ["Fail"] * 5
gnb = GaussianNB(var_smoothing=1e-12).fit(Xg, yg)
print("   sklearn GaussianNB:", {str(c): round(float(p), 3) for c, p in zip(gnb.classes_, gnb.predict_proba([[6]])[0])})

print("\n5) Flu or cold?  Fever=Yes, Cough=Yes  (Flu: 6 patients, Cold: 4)")
prior = {"Flu": 0.6, "Cold": 0.4}
fever = {"Flu": 5 / 6, "Cold": 1 / 4}
cough = {"Flu": 4 / 6, "Cold": 3 / 4}
s = {c: prior[c] * fever[c] * cough[c] for c in prior}
t = sum(s.values())
print("   scores:", {c: round(v, 5) for c, v in s.items()}, " posterior:", {c: f"{v / t:.1%}" for c, v in s.items()}, "(lecture: Flu 81.6%)")

print("\n6) Evaluate the spam filter on 10 new emails")
test = pd.DataFrame({
    "free":   ["Yes"] * 5 + ["No"] * 5,
    "offer":  ["Yes"] * 5 + ["No"] * 5,
    "actual": ["Spam", "Spam", "Spam", "Ham", "Ham", "Spam", "Ham", "Ham", "Ham", "Ham"],
})
test["pred"] = [max(naive_bayes(train, {"free": r.free, "offer": r.offer})[0].items(), key=lambda kv: kv[1])[0]
                for r in test.itertuples()]
print(test.to_string(index=False))
cm = confusion_matrix(test.actual, test.pred, labels=["Spam", "Ham"])
print("\n   Confusion matrix (rows actual, cols predicted: Spam, Ham)\n", cm)
y_t, y_p = test.actual, test.pred
print(f"   Accuracy={accuracy_score(y_t, y_p):.2f}  Precision={precision_score(y_t, y_p, pos_label='Spam'):.2f}  "
      f"Recall={recall_score(y_t, y_p, pos_label='Spam'):.2f}  F1={f1_score(y_t, y_p, pos_label='Spam'):.2f}"
      "   (lecture 0.70 / 0.60 / 0.75 / 0.67)")

print("\n7) scikit-learn CategoricalNB (alpha=1e-10 ~ no smoothing, matches the hand calculation)")
Xc = (train[["free", "offer"]] == "Yes").astype(int).values
cnb = CategoricalNB(alpha=1e-10).fit(Xc, train.label)
print("   classes:", cnb.classes_, " P(. | free=Yes, offer=Yes):", np.round(cnb.predict_proba([[1, 1]])[0], 2))
print("\n   Which variant?  GaussianNB: continuous numbers | MultinomialNB: word counts | BernoulliNB: yes/no flags")
