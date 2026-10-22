# Rung 2 — `report-card/` · grade a model properly

Rung 1 printed **one** confusion matrix at the default 0.5 threshold and stopped.
This rung shows why that's a lie by omission. Same data, same model — the whole
point is **evaluation**, not a new classifier.

```bash
cd report-card
python report.py        # prints the report card + writes roc.png, pr.png, calibration.png
```

The one idea: **a model isn't one number. It's a dial, and a report card shows the whole dial.**

---

## 1. The threshold is a dial, not a law

The model outputs `P(spam)` — a number from 0 to 1 — for every message. To turn that
into a yes/no decision you pick a **threshold**. Rung 1 used 0.5 because that's the
default, but 0.5 is just one setting of a dial you can turn:

- **Turn it down** → you call more things spam → **catch more spam (recall ↑)** but
  **flag more real messages (precision ↓)**.
- **Turn it up** → the opposite: fewer false alarms, more spam slips through.

There is no universally "right" threshold — only the right threshold **for a cost**.
The sweep from the actual run:

| threshold | precision | recall | F1 | real msgs flagged (FP) | spam missed (FN) |
|---|---|---|---|---|---|
| 0.10 | 0.676 | 0.966 | 0.796 | 69 | 5 |
| 0.20 | 0.964 | 0.893 | 0.927 | 5 | 16 |
| **0.50** (rung 1) | 1.000 | 0.779 | 0.875 | **0** | **33** |
| 0.90 | 1.000 | 0.430 | 0.601 | 0 | 85 |

**The finding:** the default 0.5 was *leaving spam on the table* — it missed 33 of 149
spam to guarantee zero false alarms. Dropping the dial to **~0.19** gets F1 to **0.931**
(catch ~90% of spam, flag only ~5 real messages). Whether that's the right move depends
on the cost — which is the next idea.

### Choosing the threshold *for the cost*

A spam filter's costs aren't symmetric: **flagging a real message (FP) is worse than
letting one spam through (FN)** — nobody wants to miss a real text. So instead of "best
F1", you can say: *demand precision ≥ 0.99, then take the best recall you can still get.*
On this run that lands at **threshold 0.35 → recall 0.81** (catches 121 spam, flags just 1
real message). That's engineering the dial to the business cost, not to a default.

---

## 2. AUC — one number for the *whole* dial

Each threshold is one row of that table. **AUC** collapses *every* threshold into a single
number, so you can compare two models before committing to any threshold.

- **ROC-AUC = 0.989** — the probability that a random spam scores higher than a random
  ham. 0.5 = a coin flip, 1.0 = perfect ranking. (See `roc.png`.)
- **PR-AUC = 0.971** — area under the precision–recall curve. **On imbalanced data this is
  the honest one.** With only 13% spam, ROC-AUC flatters the model (lots of easy
  true-negatives); PR-AUC focuses on the hard positive class. (See `pr.png`.)

> Rule of thumb: imbalanced classes → trust **PR-AUC** over ROC-AUC.

---

## 3. Calibration — are the *probabilities* honest?

Ranking well and reporting honest probabilities are **different skills**. A model can
rank perfectly yet say "0.99" for things that are only 70% spam. Calibration checks this:
bin the predictions, then ask *of the messages I called ~0.8, were ~80% actually spam?*
(See `calibration.png` — a perfectly honest model sits on the diagonal.)

**What this run showed (and why it's a good lesson):** the textbook says Naive Bayes is
*over*-confident — its "naive" independence assumption multiplies correlated evidence as
if independent, slamming scores toward 0 and 1. But our output came out the *other* way —
**under-confident** in the middle (predicted 0.44 for messages that were ~100% spam).

Either way the moral is the same, and it's the honest takeaway of the whole rung:
**Naive Bayes is a great *ranker* but a poor *number*.** When `score.py` prints
"P(spam) = 99%", read it as *"very spammy"*, not a literal 99%. Trust the ordering, not
the digit. (If you needed honest probabilities, you'd wrap it in
`sklearn.calibration.CalibratedClassifierCV` — a rung for another day.)

---

## 4. Cross-validation — is the grade a fluke of one split?

Everything above came from **one** 80/20 split. That's a single draw — the score could be
luck (an easy test set) or bad luck. **Stratified k-fold** cuts the data into k parts, trains
on k−1 and tests on the held-out one, k times, so every message gets tested exactly once. You
report **mean ± spread**:

```
5-fold stratified CV:   F1 = 0.877 ± 0.013     (precision 0.998 ± 0.003, PR-AUC 0.966 ± 0.007)
```

The spread is the point. **±0.013 is tight** → the single-split 0.875 wasn't a lucky draw, the
grade is real. A *wide* spread would mean one split could have badly misled you.

> ⚠️ **The leakage trap inside CV:** you must refit the TfidfVectorizer *inside each fold*, on
> that fold's training data only. We do this by passing the whole **Pipeline** to
> `cross_validate` — so each fold rebuilds its own vocabulary/IDF. Vectorizing all of `X`
> *first* and then running CV would bake every fold's test words into the features → silent
> leakage → inflated scores. (Same leakage lesson as rung 1, one level deeper.)

---

## 5. The report card — red / amber / green

All of the above, rolled into a verdict a human reads in five seconds:

```
🟢 GREEN  Beats the baseline          accuracy 0.970 vs always-ham 0.866
🟢 GREEN  Discrimination (ranking)    PR-AUC 0.971 (ROC-AUC 0.989)
🟢 GREEN  Stability across folds      5-fold F1 0.877 ± 0.013
🟢 GREEN  Cost-aware operating point  threshold 0.35: precision 0.992, recall 0.812
🟡 AMBER  Probability honesty         ranks well but miscalibrated
```

**Verdict (the one sentence the whole rung builds to):**
> **Trust its spam/ham *decisions*** — it beats baseline, ranks excellently (PR-AUC 0.97), and
> holds up across folds — **run it at threshold ~0.35 (not 0.5)**, but **don't read its
> `P(spam)` as a literal probability** (it's miscalibrated). *Trust the ranking, not the digit.*

That sentence is the deliverable. A model isn't "good" or "bad" — it's *trustworthy for X, not
for Y*, and now you can say which, with numbers behind every clause.

---

## What rung 2 adds over rung 1

| | rung 1 (`spam/`) | rung 2 (`report-card/`) |
|---|---|---|
| decision threshold | fixed at 0.5 | swept; chosen for a cost |
| headline metric | accuracy / one F1 | the precision–recall trade, plus AUC |
| imbalance | noted | measured honestly (PR-AUC) |
| probabilities | taken at face value | checked for calibration |
| reliability of the number | one split | **5-fold CV, mean ± spread** |
| output | numbers | numbers + 3 plots **+ a red/amber/green report card & trust verdict** |

**Done when:** confusion matrix → P/R/F1 → ROC-AUC & PR-AUC → a *justified* threshold →
calibration curve → stratified k-fold with variance → a red/amber/green card — **and you can
say in one sentence why this model should or shouldn't be trusted.** ✅ (see §5)

**The one line to keep:** *the single-number grade was a lie; the report card is the truth.*
