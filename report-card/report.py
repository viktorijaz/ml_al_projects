"""
Rung 2 — report-card/ · grade a model properly.

Rung 1 printed ONE confusion matrix at the default 0.5 threshold and called it a
day. But 0.5 is an arbitrary dial setting. A real "report card" shows the whole
dial: how precision/recall trade off as you move the threshold, a single number
that summarizes every threshold at once (AUC), and whether the model's stated
probabilities are HONEST (calibration).

Same data, same pipeline as rung 1 — this rung is about *evaluation*, not a new model.

Run:  python report.py
Writes three plots next to this file: roc.png, pr.png, calibration.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
matplotlib.use("Agg")  # no display needed; we only save PNGs
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
# Reuse rung 1's dataset — report-card grades the SAME model, it doesn't retrain a new idea.
DATA = HERE.parent / "spam" / "data" / "SMSSpamCollection"


# --- rebuild rung 1 exactly (same split, same pipeline) ------------------
df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"])
y = (df["label"] == "spam").astype(int)      # 1 = spam (the positive class)
X = df["text"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english"),
    MultinomialNB(),
)
model.fit(X_train, y_train)

# The raw material of this whole rung: P(spam) for every test message, 0.0–1.0.
# Every metric below is just a different way of reading THIS one array of scores.
proba = model.predict_proba(X_test)[:, 1]

print(f"Test set: {len(y_test)} messages | {int(y_test.sum())} spam ({y_test.mean():.1%})\n")


# --- 1. the default: one confusion matrix at threshold 0.5 ---------------
# This is all rung 1 showed. It's one row of the table below.
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


# --- 2. threshold sweep: 0.5 is a DIAL, not a law ------------------------
# Lower the bar -> catch more spam (recall up) but flag more real mail (precision down).
# Raise it -> the opposite. There is no single "right" threshold; there's a right
# threshold FOR A COST. We show the trade and then pick two sensible operating points.
print("Threshold sweep — precision/recall trade as you turn the dial:")
print("  thresh   precision   recall     f1     (FP  FN)")
rows = [report_at(t) for t in np.arange(0.1, 0.96, 0.1)]
for r in rows:
    print(f"   {r['t']:.2f}      {r['precision']:.3f}     {r['recall']:.3f}   {r['f1']:.3f}   ({r['fp']:3d} {r['fn']:3d})")

# Operating point A — best F1 (balance precision and recall).
fine = [report_at(t) for t in np.arange(0.05, 0.96, 0.01)]
best_f1 = max(fine, key=lambda r: r["f1"])
print(f"\n  -> best-F1 operating point:  threshold={best_f1['t']:.2f}  "
      f"f1={best_f1['f1']:.3f}  (precision={best_f1['precision']:.3f} recall={best_f1['recall']:.3f})")

# Operating point B — a spam filter's real cost: flagging a real message (FP) is
# worse than missing a spam (FN). So demand precision >= 0.99, then take the best
# recall we can still get. This is "choosing the threshold for the cost."
safe = [r for r in fine if r["precision"] >= 0.99]
if safe:
    best_safe = max(safe, key=lambda r: r["recall"])
    print(f"  -> precision>=0.99 point:    threshold={best_safe['t']:.2f}  "
          f"recall={best_safe['recall']:.3f}  (catches {best_safe['tp']} spam, "
          f"flags {best_safe['fp']} real messages)")


# --- 3. AUC: one number for the WHOLE dial -------------------------------
# A single threshold is one row. AUC summarizes performance across ALL thresholds
# at once, so you can compare two models without first fixing a threshold.
#   ROC-AUC   : P(a random spam scores higher than a random ham). 0.5 = coin, 1.0 = perfect.
#   PR-AUC    : area under precision-recall. On IMBALANCED data (13% spam) this is the
#               honest one — ROC-AUC looks flatteringly high when negatives dominate.
roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)
print(f"\nWhole-dial summary:")
print(f"  ROC-AUC = {roc_auc:.3f}   (rank quality: random spam > random ham this often)")
print(f"  PR-AUC  = {pr_auc:.3f}   (the honest one for 13%-spam imbalance)")


# --- 4. calibration: are the PROBABILITIES honest? -----------------------
# Rank-quality (AUC) and honesty are different things. A model can rank perfectly
# yet say "0.99" for things that are only 70% spam. Calibration bins the predictions
# and asks: of messages I called ~0.8, were ~80% actually spam?
# (Real-world tidbit: Naive Bayes is famously MIS-calibrated — the "naive" independence
#  assumption multiplies correlated evidence as if independent, so its probabilities
#  are a great *ranker* but a poor *number*. The textbook says it's over-confident
#  (pushed to 0/1); on THIS run it comes out the other way — under-confident in the
#  middle (see the printout: 0.44 predicted, ~1.0 actual). Either way the lesson holds:
#  score.py's "P(spam)=99%" means "very spammy," not a literal 99%. Trust the ranking,
#  not the digit.)
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
print("\nCalibration — predicted vs. actual (closer to equal = more honest):")
print("   predicted   actually spam")
for mp, fp_ in zip(mean_pred, frac_pos):
    gap = fp_ - mp
    print(f"     {mp:.2f}        {fp_:.2f}   {'(over-confident)' if gap < -0.1 else '(under-confident)' if gap > 0.1 else ''}")


# --- plots: the report card as pictures ----------------------------------
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
print("The one-number grade was a lie; this is the report card.")
