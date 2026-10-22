import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

df = pd.read_csv("../spam/data/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
y = (df["label"] == "spam").astype(int)
X = df["text"]
print(df.shape)
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("train:", len(X_train), " test:", len(X_test))

model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english"),   # text  -> features
    MultinomialNB(),                                         # features -> spam/ham
)
model.fit(X_train, y_train)
print("trained!")

proba = model.predict_proba(X_test)[:, 1]
print(proba[:10])
print("shape:", proba.shape)

print(f"Test set: {len(y_test)} messages | {int(y_test.sum())} spam ({y_test.mean():.1%})\n")

# decide: call it spam if P(spam) >= threshold
pred = (proba >= 0.5).astype(int)

# count the 4 outcomes: truth vs prediction
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("TN", tn, " FP", fp, " FN", fn, " TP", tp)

print("precision:", precision_score(y_test, pred))
print("recall   :", recall_score(y_test, pred))
print("f1       :", f1_score(y_test, pred))

for t in np.arange(0.1, 0.95, 0.1):
    pred_t = (proba >= t).astype(int)          # same decision, different threshold
    p = precision_score(y_test, pred_t, zero_division=0)
    r = recall_score(y_test, pred_t)
    f = f1_score(y_test, pred_t)
    print(f"thresh {t:.2f}   precision {p:.3f}   recall {r:.3f}   f1 {f:.3f}")

roc = roc_auc_score(y_test, proba)
pr  = average_precision_score(y_test, proba)
print("ROC-AUC:", roc)
print("PR-AUC :", pr)


frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")

for mp, actual in zip(mean_pred, frac_pos):
    print(f"predicted {mp:.2f}   actually spam {actual:.2f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    model, X, y, cv=cv,
    scoring=["precision", "recall", "f1", "average_precision", "roc_auc"],
)

print("Cross validation:")
for m in ["precision", "recall", "f1", "average_precision", "roc_auc"]:
    s = scores["test_" + m]
    print(f"{m:18s} {s.mean():.3f} ± {s.std():.3f}")


GREEN, AMBER, RED = "🟢 GREEN", "🟡 AMBER", "🔴 RED"

cv_f1    = scores["test_f1"]                 # from step 9
acc      = (tp + tn) / len(y_test)           # accuracy at 0.5, from step 5
baseline = 1 - y_test.mean()                 # "always predict ham"

print("\n===== MODEL REPORT CARD =====")

print(GREEN if acc > baseline + 0.05 else RED,
    "beats baseline:", f"{acc:.3f} vs {baseline:.3f}")

print(GREEN if pr >= 0.98 else AMBER,
    "ranking (PR-AUC):", f"{pr:.3f}")

print(GREEN if cv_f1.std() < 0.03 else AMBER,
    "stability:", f"F1 {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

print(AMBER, "probability honesty: ranks well, but miscalibrated (step 8)")

print("\nVerdict: TRUST its spam/ham DECISIONS "
    f"(PR-AUC {pr:.2f}, stable across folds) — "
    "but DON'T read P(spam) as a literal number.")
