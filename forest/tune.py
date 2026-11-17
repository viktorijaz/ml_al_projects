import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from data import X_test, X_train, dollars, prep, y_test, y_train


def xgb(depth, rate, n=5000):
    return XGBRegressor(
        n_estimators=n,
        learning_rate=rate,
        max_depth=depth,
        subsample=0.8,
        colsample_bytree=0.5,
        early_stopping_rounds=100 if n == 5000 else None,
        eval_metric="rmse",
        random_state=42,
    )


# 5-fold cross-validation on the training houses only, for each setting
folds = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
for depth in [2, 3, 4, 6]:
    for rate in [0.01, 0.03, 0.1]:
        misses, rounds = [], []
        for fit_idx, val_idx in folds.split(X_train):
            p = clone(prep).fit(X_train.iloc[fit_idx])  # prep learned per fold, no leakage
            model = xgb(depth, rate)
            model.fit(
                p.transform(X_train.iloc[fit_idx]), y_train.iloc[fit_idx],
                eval_set=[(p.transform(X_train.iloc[val_idx]), y_train.iloc[val_idx])],
                verbose=False,
            )
            misses.append(model.best_score)
            rounds.append(model.best_iteration + 1)
        results.append((np.mean(misses), depth, rate, int(np.mean(rounds))))
        print(f"depth {depth}  rate {rate:<5} → miss ≈ {np.mean(misses):.1%}  ({int(np.mean(rounds))} trees)")

miss, depth, rate, n = min(results)
print(f"\nbest: depth {depth}, rate {rate}, {n} trees")

# retrain the best setting on all training houses, then score the test set once
final = xgb(depth, rate, n)
p = clone(prep).fit(X_train)
final.fit(p.transform(X_train), y_train)
print(f"test RMSE: ${dollars(y_test, final.predict(p.transform(X_test))):,.0f}   (lasso $19,276)")
