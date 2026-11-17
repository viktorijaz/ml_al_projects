from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).parent

df = fetch_openml(data_id=42165, as_frame=True).frame
X = df[["OverallQual"]]
y = df["SalePrice"]

line = LinearRegression().fit(X, y)  # straight line on dollars: the wrong shape
pred = line.predict(X)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(X["OverallQual"], y, s=5, alpha=0.4)
ax[0].plot([1, 10], line.predict(pd.DataFrame({"OverallQual": [1, 10]})), color="red")
ax[0].set(title="Price vs quality, with a straight line", xlabel="OverallQual (1–10)", ylabel="Price")
ax[1].scatter(pred, y - pred, s=5, alpha=0.4)
ax[1].axhline(0, color="red")
means = (y - pred).groupby(X["OverallQual"]).mean()
ax[1].plot(line.predict(means.index.to_frame(name="OverallQual")), means, "o-", color="black", label="average residual")
ax[1].set(title="Residuals vs predicted: the U", xlabel="Predicted price", ylabel="Actual − predicted ($)")
ax[1].legend()
plt.tight_layout()
plt.savefig(HERE / "u_shape.png")
