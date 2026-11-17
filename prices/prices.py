from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

HERE = Path(__file__).parent

df = fetch_openml(name="house_prices", as_frame=True).frame
print(df.shape)
print(df["SalePrice"].describe())

# missing values: top 10 columns by % missing
missing = df.isna().mean().sort_values(ascending=False).head(10)
print((missing * 100).round(1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df["SalePrice"].hist(bins=50, ax=ax[0])  # look 1: how prices are spread
ax[0].set_title("Sale price")
missing.plot.barh(ax=ax[1])  # look 2: which columns are mostly empty
ax[1].set_title("Share missing")
ax[2].scatter(df["GrLivArea"], df["SalePrice"], s=5)  # look 3: size vs price
ax[2].set_xlabel("Living area (sq ft)")
ax[2].set_ylabel("Price")
plt.tight_layout()
plt.savefig(HERE / "looks.png")
