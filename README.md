# ml_al_projects — an end-to-end ML/DL ladder

A ladder of **shipping projects**. Each rung is a whole project — `data → model → honest
evaluation → a thing that runs` — built one rung at a time. Every rung ships something you can
show and explains itself from first principles.

## Rungs

| # | Project | What it does | Status |
|---|---|---|---|
| 1 | [`spam/`](spam/) | SMS spam classifier — TF-IDF + Naive Bayes, with a full evaluation report | ✅ shipped |
| 2 | `report-card/` | grade a model properly — confusion matrix, threshold tuning, calibration | next |

Rung 1 writeup (the concepts, from first principles): [`spam/README.md`](spam/README.md).

## Run rung 1

```bash
python -m venv .venv && source .venv/bin/activate
pip install scikit-learn pandas numpy
cd spam
python train.py                              # train + print the evaluation report
python score.py "You won a FREE prize!!!"    # score a message
```

---

*Built in public, one rung at a time.*
