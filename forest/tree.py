from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree

from data import X_test, X_train, dollars, prep, y_test, y_train

HERE = Path(__file__).parent

# one tree at growing depths: train error vs test error
print(f"{'depth':>6} {'leaves':>7} {'train RMSE':>12} {'test RMSE':>11}")
for depth in [1, 2, 4, 6, 8, 12, None]:
    tree = make_pipeline(prep, DecisionTreeRegressor(max_depth=depth, random_state=42))
    tree.fit(X_train, y_train)
    train = dollars(y_train, tree.predict(X_train))
    test = dollars(y_test, tree.predict(X_test))
    leaves = tree[-1].get_n_leaves()
    print(f"{str(depth):>6} {leaves:>7} {train:>12,.0f} {test:>11,.0f}")

# draw a small tree to see what it learned
small = make_pipeline(prep, DecisionTreeRegressor(max_depth=2, random_state=42))
small.fit(X_train, y_train)
plt.figure(figsize=(14, 6))
plot_tree(small[-1], feature_names=small[0].get_feature_names_out(), filled=True, impurity=False, precision=2)
plt.savefig(HERE / "tree.png")
