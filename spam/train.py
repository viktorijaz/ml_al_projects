"""
Rung 1 — spam/ · classify SMS as spam/ham, start to finish.
Run:  python train.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

HERE = Path(__file__).parent
DATA = HERE / "data" / "SMSSpamCollection"
MODEL_OUT = HERE / "model.joblib"

# 1. DATA — tab-separated <label>\t<message>; 1 = spam (the class we want to catch).
df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"])
y = (df["label"] == "spam").astype(int)
X = df["text"]

n_spam, n_total = int(y.sum()), len(y)
print(f"Loaded {n_total} messages | spam: {n_spam} ({n_spam/n_total:.1%}) | ham: {n_total-n_spam}")

# TRAIN/TEST SPLIT — never test on training data; stratify keeps the 13% spam ratio in both halves.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# BASELINE — "always predict ham"; the bar to beat (deceptively high on imbalanced data = why accuracy alone lies).
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"\nBaseline (always 'ham') accuracy: {base_acc:.3f}  <- the bar to clear")

# 2/3. FEATURES + MODEL in one Pipeline — TF-IDF is fit inside, so it never sees the test set (no leakage).
model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english"),
    MultinomialNB(),
)
model.fit(X_train, y_train)

# 4. EVALUATION — a panel, not one number.
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)   # when it says spam, how often right?
rec = recall_score(y_test, pred)       # of all real spam, how much caught?
f1 = f1_score(y_test, pred)

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("\nConfusion matrix (test set):")
print("                 predicted ham   predicted spam")
print(f"  actual ham        {tn:5d}           {fp:5d}   <- FP: good mail flagged as spam")
print(f"  actual spam       {fn:5d}           {tp:5d}   <- TP caught / FN missed")

print(f"\n  Accuracy : {acc:.3f}   (baseline was {base_acc:.3f})")
print(f"  Precision: {prec:.3f}   (of flagged-spam, how much really is)")
print(f"  Recall   : {rec:.3f}   (of real spam, how much we caught)")
print(f"  F1       : {f1:.3f}")

print("\nFull report:")
print(classification_report(y_test, pred, target_names=["ham", "spam"]))

# 5. SHIP — save the whole pipeline (vectorizer + model together).
joblib.dump(model, MODEL_OUT)
print(f"Saved model -> {MODEL_OUT.name}   (now run: python score.py \"your message\")")
