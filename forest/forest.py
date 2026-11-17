from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from data import X_test, X_train, dollars, prep, y_test, y_train

# more and more trees, each grown deep on a random sample of houses
print(f"{'trees':>6} {'train RMSE':>12} {'test RMSE':>11}")
for n in [1, 10, 50, 200, 500]:
    forest = make_pipeline(
        prep,
        RandomForestRegressor(
            n_estimators=n,
            max_features=0.33,  # each question may only look at a random third of the columns
            oob_score=n >= 50,  # too few trees leave some houses unscored
            n_jobs=-1,
            random_state=42,
        ),
    )
    forest.fit(X_train, y_train)
    train = dollars(y_train, forest.predict(X_train))
    test = dollars(y_test, forest.predict(X_test))
    print(f"{n:>6} {train:>12,.0f} {test:>11,.0f}")

# out-of-bag: each house scored only by the trees that never saw it
oob = forest[-1].oob_prediction_
print(f"\nout-of-bag RMSE (500 trees, no test set used): {dollars(y_train, oob):,.0f}")
