"""
Rung 2 — report-card/ · grade a model properly (same data/model as rung 1).
Shows the whole threshold dial, AUC, calibration, cross-validation, and a verdict.
Run:  python report.py   (writes roc.png, pr.png, calibration.png)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")  # no display; just save PNGs
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE.parent / "spam" / "data" / "SMSSpamCollection"   # reuse rung 1's data


# rebuild rung 1 exactly (same split, same pipeline)
df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"])
y = (df["label"] == "spam").astype(int)      # 1 = spam
X = df["text"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english"),
    MultinomialNB(),
)
model.fit(X_train, y_train)

# P(spam) per test message — the raw material every metric below reads.
proba = model.predict_proba(X_test)[:, 1]

print(f"Test set: {len(y_test)} messages | {int(y_test.sum())} spam ({y_test.mean():.1%})\n")


# 1. confusion matrix + P/R/F1 at a given threshold (0.5 = what rung 1 showed).
def report_at(threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return dict(
        t=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
    )

d = report_at(0.5)
print("At the default threshold 0.5 (what rung 1 showed):")
print(f"  precision={d['precision']:.3f}  recall={d['recall']:.3f}  f1={d['f1']:.3f}")
print(f"  (caught {d['tp']} spam, missed {d['fn']}, flagged {d['fp']} real messages)\n")


# 2. threshold sweep — 0.5 is a dial: lower = more recall/less precision, higher = the reverse.
print("Threshold sweep — precision/recall trade as you turn the dial:")
print("  thresh   precision   recall     f1     (FP  FN)")
rows = [report_at(t) for t in np.arange(0.1, 0.96, 0.1)]
for r in rows:
    print(f"   {r['t']:.2f}      {r['precision']:.3f}     {r['recall']:.3f}   {r['f1']:.3f}   ({r['fp']:3d} {r['fn']:3d})")

# pick two operating points: best F1, and (for spam) best recall at precision >= 0.99.
fine = [report_at(t) for t in np.arange(0.05, 0.96, 0.01)]
best_f1 = max(fine, key=lambda r: r["f1"])
print(f"\n  -> best-F1 operating point:  threshold={best_f1['t']:.2f}  "
      f"f1={best_f1['f1']:.3f}  (precision={best_f1['precision']:.3f} recall={best_f1['recall']:.3f})")

safe = [r for r in fine if r["precision"] >= 0.99]
if safe:
    best_safe = max(safe, key=lambda r: r["recall"])
    print(f"  -> precision>=0.99 point:    threshold={best_safe['t']:.2f}  "
          f"recall={best_safe['recall']:.3f}  (catches {best_safe['tp']} spam, "
          f"flags {best_safe['fp']} real messages)")


# 3. AUC — one number for ALL thresholds. PR-AUC is honest on imbalance; ROC-AUC flatters.
roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)
print(f"\nWhole-dial summary:")
print(f"  ROC-AUC = {roc_auc:.3f}   (rank quality: random spam > random ham this often)")
print(f"  PR-AUC  = {pr_auc:.3f}   (the honest one for 13%-spam imbalance)")


# 4. calibration — are the probabilities honest? NB ranks well but its numbers aren't literal.
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
print("\nCalibration — predicted vs. actual (closer to equal = more honest):")
print("   predicted   actually spam")
for mp, fp_ in zip(mean_pred, frac_pos):
    gap = fp_ - mp
    print(f"     {mp:.2f}        {fp_:.2f}   {'(over-confident)' if gap < -0.1 else '(under-confident)' if gap > 0.1 else ''}")


# 5. cross-validation — is the grade a fluke of one split? Pipeline refits per fold = no leakage.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(
    model, X, y, cv=cv,
    scoring=["precision", "recall", "f1", "average_precision", "roc_auc"],
)
print("\n5-fold stratified cross-validation (mean ± std across folds):")
for m in ["precision", "recall", "f1", "average_precision", "roc_auc"]:
    s = cv_scores[f"test_{m}"]
    label = "PR-AUC " if m == "average_precision" else m
    print(f"  {label:9s} {s.mean():.3f} ± {s.std():.3f}")
cv_f1 = cv_scores["test_f1"]
cv_prauc = cv_scores["test_average_precision"]
print(f"  -> F1 spread is {cv_f1.std():.3f}: "
      f"{'tight → the grade is real, not a lucky split' if cv_f1.std() < 0.03 else 'wide → one split would have misled you'}")


# plots: ROC, PR, calibration
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"model (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="coin flip")
plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
plt.title("ROC curve"); plt.legend(); plt.tight_layout()
plt.savefig(HERE / "roc.png", dpi=110); plt.close()

prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
plt.figure(figsize=(5, 5))
plt.plot(rec_c, prec_c, label=f"model (PR-AUC={pr_auc:.3f})")
plt.axhline(y_test.mean(), ls="--", color="gray", label=f"baseline ({y_test.mean():.2f})")
plt.xlabel("recall"); plt.ylabel("precision")
plt.title("Precision–Recall curve"); plt.legend(); plt.tight_layout()
plt.savefig(HERE / "pr.png", dpi=110); plt.close()

plt.figure(figsize=(5, 5))
plt.plot(mean_pred, frac_pos, "o-", label="model")
plt.plot([0, 1], [0, 1], "--", color="gray", label="perfectly honest")
plt.xlabel("predicted P(spam)"); plt.ylabel("actual fraction spam")
plt.title("Calibration"); plt.legend(); plt.tight_layout()
plt.savefig(HERE / "calibration.png", dpi=110); plt.close()

print(f"\nSaved plots -> roc.png, pr.png, calibration.png")


# 6. report card — roll it all into red/amber/green + a one-sentence verdict.
GREEN, AMBER, RED = "🟢 GREEN", "🟡 AMBER", "🔴 RED"

def band(value, green_at, amber_at):
    """Higher is better: >= green_at is green, >= amber_at is amber, else red."""
    return GREEN if value >= green_at else AMBER if value >= amber_at else RED

acc = (d["tp"] + d["tn"]) / len(y_test)        # accuracy at the default 0.5
base_acc = 1 - y_test.mean()                   # "always predict ham" baseline
op = best_safe if safe else best_f1            # the operating point we'd actually ship

card = [
    ("Beats the baseline",
     GREEN if acc > base_acc + 0.05 else AMBER if acc > base_acc else RED,
     f"accuracy {acc:.3f} vs always-ham baseline {base_acc:.3f}"),
    ("Discrimination (ranking)",
     band(pr_auc, 0.90, 0.75),
     f"PR-AUC {pr_auc:.3f}  (ROC-AUC {roc_auc:.3f}; trust PR on imbalance)"),
    ("Stability across folds",
     GREEN if cv_f1.std() < 0.03 else AMBER if cv_f1.std() < 0.06 else RED,
     f"5-fold F1 {cv_f1.mean():.3f} ± {cv_f1.std():.3f}  (not a lucky split)"),
    ("Cost-aware operating point",
     GREEN if op["precision"] >= 0.99 else AMBER,
     f"threshold {op['t']:.2f}: precision {op['precision']:.3f}, recall {op['recall']:.3f}"),
    ("Probability honesty",
     AMBER,
     "ranks well but miscalibrated — treat P(spam) as a score, not a literal %"),
]

print("\n" + "=" * 64)
print("  MODEL REPORT CARD")
print("=" * 64)
for name, verdict, detail in card:
    print(f"  {verdict}  {name}")
    print(f"             {detail}")

print("\n  Verdict — can we trust this model? (one sentence):")
print(f"    TRUST its spam/ham DECISIONS (beats baseline, PR-AUC {pr_auc:.2f},")
print(f"    F1 {cv_f1.mean():.2f}±{cv_f1.std():.2f} across folds), run it at "
      f"threshold ~{op['t']:.2f} not 0.5,")
print(f"    but do NOT read its P(spam) as a literal probability (miscalibrated)")
print(f"    — trust the ranking, not the digit.")
