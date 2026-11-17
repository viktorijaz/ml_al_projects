import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from data import X_test, X_train, dollars, prep, y_test, y_train

# carve a validation set out of the training houses, the test set stays untouched
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
prep.fit(X_fit)

boost = XGBRegressor(
    n_estimators=5000,  # an upper limit, early stopping picks the real number
    learning_rate=0.03,  # each new tree fixes only 3% of the remaining error
    max_depth=4,  # small trees
    subsample=0.8,  # each tree sees 80% of the houses
    colsample_bytree=0.5,  # and half of the columns
    early_stopping_rounds=100,  # stop when 100 more trees don't help on validation
    eval_metric="rmse",
    random_state=42,
)
boost.fit(prep.transform(X_fit), y_fit, eval_set=[(prep.transform(X_val), y_val)], verbose=False)

# how the validation error fell as trees were added (log scale ≈ percent)
curve = boost.evals_result()["validation_0"]["rmse"]
for n in [1, 10, 50, 100, 300, 1000]:
    if n <= len(curve):
        print(f"after {n:>4} trees: validation miss ≈ {curve[n - 1]:.1%}")
print(f"stopped at {len(curve)} trees, best was {boost.best_iteration + 1}")

pred = boost.predict(prep.transform(X_test))
print(f"\ntest RMSE: ${dollars(y_test, pred):,.0f}   (lasso $19,276, forest $22,491)")
