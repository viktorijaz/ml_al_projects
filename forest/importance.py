from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from data import X_test, X_train, dollars, prep, y_test, y_train

HERE = Path(__file__).parent

# the best setting from tune.py, trained on all training houses
model = make_pipeline(
    prep,
    XGBRegressor(
        n_estimators=828, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.5, random_state=42,
    ),
)
model.fit(X_train, y_train)
print(f"test RMSE: ${dollars(y_test, model.predict(X_test)):,.0f}")

# shuffle one column at a time on the test houses, see how much worse it gets
perm = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42,
    scoring="neg_root_mean_squared_error",
)
honest = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)

# xgboost's own built-in importance, for comparison
builtin = pd.Series(model[-1].feature_importances_, index=model[0].get_feature_names_out())
builtin = builtin.sort_values(ascending=False)

print("\npermutation (shuffle → extra miss, log scale ≈ %):")
print(honest.head(10).round(3).to_string())
print("\nbuilt-in (share of the model's splits gain):")
print(builtin.head(10).round(3).to_string())

honest.head(15)[::-1].plot.barh(figsize=(7, 5), title="Permutation importance (test houses)")
plt.xlabel("extra miss when shuffled (≈ %)")
plt.tight_layout()
plt.savefig(HERE / "importance.png")
