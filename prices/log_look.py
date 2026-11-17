from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

HERE = Path(__file__).parent

price = fetch_openml(data_id=42165, as_frame=True).frame["SalePrice"]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(price, bins=50)
ax[0].set_title(f"Price in dollars (skew {price.skew():.2f})")
ax[1].hist(np.log(price), bins=50)
ax[1].set_title(f"log(price) (skew {np.log(price).skew():.2f})")
for a, values in zip(ax, [price, np.log(price)]):
    a.axvline(values.mean(), color="red", label="mean")  # pulled right by the tail
    a.axvline(values.median(), color="black", linestyle="--", label="median")
    a.legend()
plt.tight_layout()
plt.savefig(HERE / "log.png")
