# End-to-end ML DL projects

A ladder of **shipping projects**. Each rung is a whole project - `data → model → honest
evaluation → a thing that runs` - built one rung at a time. Every rung ships something you can
show and explains itself from first principles.

## Rungs

| #   | Project                        | What it does                                                              | Status     |
| --- | ------------------------------ | ------------------------------------------------------------------------- | ---------- |
| 1   | [`spam/`](spam/)               | SMS spam classifier - TF-IDF + Naive Bayes, with a full evaluation report | ✅ shipped |
| 2   | [`report-card/`](report-card/) | grade a model properly - threshold tuning, ROC/PR-AUC, calibration        | ✅ shipped |
| 3   | [`features/`](features/)       | why the features decide everything - engineer + compare, measure the lift | ✅ shipped |
| 4   | [`prices/`](prices/)           | regression - predict a number (house prices), not a class; residual plots | ✅ shipped |
| 5   | `forest/`                      | ensembles - decision tree → random forest → gradient boosting (XGBoost)   | next       |
| 6   | `clusters/`                    | unsupervised - find structure with no labels: k-means + PCA               | planned    |

Per-rung writeups (the concepts, from first principles):
[`spam/README.md`](spam/README.md) · [`report-card/README.md`](report-card/README.md) · [`features/README.md`](features/README.md) · [`prices/README.md`](prices/README.md).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install scikit-learn pandas numpy matplotlib
# rung 1 - train + score
cd spam
python train.py                              # train + print the evaluation report
python score.py "You won a FREE prize!!!"    # score a message
# rung 2 - grade it properly
cd ../report-card
python report.py                             # report card + roc.png, pr.png, calibration.png
# rung 3 - feature engineering
cd ../features
python features.py
# rung 4 - regression
cd ../prices
python prices.py                             # three looks before modelling → looks.png
python model.py                              # median guess vs linear vs ridge vs lasso → residuals.png
```

---
