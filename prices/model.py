from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent

df = fetch_openml(data_id=42165, as_frame=True).frame  # pinned: Ames house prices

# decisions from step 1
partial_giants = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300_000)
df = df[~partial_giants]  # 2 unfinished houses sold cheap, not market prices
df = df.drop(columns="Id")  # a row number, says nothing about the house
df["MSSubClass"] = df["MSSubClass"].astype(str)  # a category code, not a quantity

X = df.drop(columns="SalePrice")
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numbers = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categories = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="None"),  # empty = "no pool/garage/fence"
    OneHotEncoder(handle_unknown="ignore"),
)
prep = ColumnTransformer(
    [
        ("numbers", numbers, make_column_selector(dtype_include="number")),
        ("categories", categories, make_column_selector(dtype_exclude="number")),
    ]
)


def report(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:<20} RMSE ${rmse:>9,.0f}   MAE ${mae:>9,.0f}   R² {r2:6.3f}")
    return pred


# baseline: always guess the median price
report("median guess", DummyRegressor(strategy="median"))

# linear regression on log(price), predictions turned back into dollars
linear = TransformedTargetRegressor(
    regressor=make_pipeline(prep, LinearRegression()),
    func=np.log,
    inverse_func=np.exp,
)
pred = report("linear (log price)", linear)

# step 3: residuals, on the log scale (0.10 ≈ the model guessed 10% too low)
resid = np.log(y_test) - np.log(pred)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].scatter(y_test, pred, s=8)
ax[0].plot([0, 600_000], [0, 600_000], color="red")  # perfect predictions lie on this line
ax[0].set(title="Predicted vs actual", xlabel="Actual price", ylabel="Predicted price")
ax[1].scatter(pred, resid, s=8)
ax[1].axhline(0, color="red")  # random noise around this line = good
ax[1].set(title="Residuals vs predicted", xlabel="Predicted price", ylabel="log(actual) − log(predicted)")
ax[2].hist(resid, bins=40)
ax[2].set(title="Residuals", xlabel="log(actual) − log(predicted)")
plt.tight_layout()
plt.savefig(HERE / "residuals.png")

# the 5 biggest misses
worst = X_test.assign(actual=y_test, predicted=pred.round(), miss=resid.round(2))
cols = ["GrLivArea", "OverallQual", "Neighborhood", "SaleCondition", "actual", "predicted", "miss"]
print(worst.sort_values("miss", key=abs, ascending=False)[cols].head(5).to_string())

# step 4: regularization, the same pipeline with a penalty on big weights
def log_model(regressor):
    return TransformedTargetRegressor(
        regressor=make_pipeline(prep, regressor), func=np.log, inverse_func=np.exp
    )


print()
ridge = log_model(RidgeCV(alphas=np.logspace(-1, 3, 30)))  # picks the penalty by cross-validation
report("ridge", ridge)
lasso = log_model(LassoCV(alphas=np.logspace(-4, -1, 30), max_iter=50_000))
report("lasso", lasso)

print("ridge penalty:", round(ridge.regressor_[-1].alpha_, 2))
print("lasso penalty:", round(lasso.regressor_[-1].alpha_, 5))

# which columns lasso kept, and the strongest ones
names = lasso.regressor_[0].get_feature_names_out()
weights = lasso.regressor_[-1].coef_
print(f"lasso kept {(weights != 0).sum()} of {len(weights)} columns")
top = sorted(zip(weights, names), key=lambda w: abs(w[0]), reverse=True)[:10]
for w, name in top:
    print(f"  {w:+.3f}  {name}")
