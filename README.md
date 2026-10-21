# ml_al_projects — an end-to-end ML/DL ladder

A ladder of **shipping projects**. Each rung is a whole project — `data → model → honest
evaluation → a thing that runs` — built one rung at a time. Every rung ships something you can
show and explains itself from first principles.

## Rungs

| # | Project | What it does | Status |
|---|---|---|---|
| 1 | [`spam/`](spam/) | SMS spam classifier — TF-IDF + Naive Bayes, with a full evaluation report | ✅ shipped |
| 2 | [`report-card/`](report-card/) | grade a model properly — threshold tuning, ROC/PR-AUC, calibration | ✅ shipped |
| 3 | `features/` | why the features decide everything — engineer + compare, measure the lift | next |

Per-rung writeups (the concepts, from first principles):
[`spam/README.md`](spam/README.md) · [`report-card/README.md`](report-card/README.md).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install scikit-learn pandas numpy matplotlib
# rung 1 — train + score
cd spam
python train.py                              # train + print the evaluation report
python score.py "You won a FREE prize!!!"    # score a message
# rung 2 — grade it properly
cd ../report-card
python report.py                             # report card + roc.png, pr.png, calibration.png
```

---

*Built in public, one rung at a time.*
