# Rung 4 - `prices/` · regression

Rungs 1–3 predicted a **class** (spam or ham). This rung predicts a **number**: the sale price of a house.
The data is Ames Housing (1,460 houses, 79 facts about each, from the Ames tax office), loaded with `fetch_openml(data_id=42165)`.

## Step 1: look before you predict (`prices.py` → `looks.png`)

Three charts before any model, each one decided something:

| Look                | What it showed                                                                       | Decision                                         |
| ------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------ |
| price histogram     | lopsided: most houses $100k–250k, a long tail to $755k (skew 1.88)                  | predict **log(price)** (skew drops to 0.12)      |
| missing values      | `PoolQC` 99.5% empty, `FireplaceQu` empty for exactly the 690 houses with no fireplace | empty means **"None"**, not unknown              |
| size vs price       | 2 huge cheap houses: both `SaleCondition = Partial` (sold unfinished)                | drop those 2, not market prices                  |

Also: `Id` is a row number (dropped), and `MSSubClass` is a category code stored as a number (turned into text).
`log_look.py` → `log.png` shows dollars vs log(price) side by side.

## Step 2: first model (`model.py`)

A `Pipeline` so medians, scaling and categories are learned **only from the training houses**:

```
numbers (35 columns)     empty → median, then scale
categories (44 columns)  empty → "None", then one-hot → 278 columns
                         = 313 columns → LinearRegression on log(price) → back to dollars
```

## Results (292 test houses)

| model        | RMSE    | MAE     | R²     |
| ------------ | ------: | ------: | -----: |
| median guess | $76,742 | $57,100 | −0.066 |
| linear       | $22,167 | $15,501 |  0.911 |
| ridge        | $19,919 | $14,344 |  0.928 |
| **lasso**    | **$19,276** | **$13,830** | **0.933** |

- **MAE** = the average miss. **RMSE** = the same, but big misses count extra. **R²** = the share of price differences explained.
- The median guess is the ruler: it never looks at the house.

## Step 3: residuals (`residuals.png`)

Residual = log(actual) − log(predicted), so it reads as roughly a **percent** miss.

- Predicted vs actual hugs the line, but it measures in dollars and hides the problem.
- Residuals vs predicted: a flat band, **no U-shape**, so a straight line on log price is the right shape (`u_shape.py` shows what a U looks like).
- **But cheap houses are the weak spot**: houses predicted under $100k are missed by ~21% on average, the rest by ~8.5%.
- The 5 biggest misses are all small houses, mostly cheap (one `Partial` again).

R² 0.911 sounded excellent and hid all of this. Only the residual plot showed it.

## Step 4: Ridge and Lasso

Plain linear gives every one of 313 columns a weight, including columns only 1–2 houses have:
"no pool: +1.70", "good pool: −0.78". That is **overfitting** (high variance).

Ridge and Lasso add a **penalty on big weights** to the squared error, so a weight must earn its size:

- **Ridge** shrinks all weights.
- **Lasso** sets useless ones to exactly 0: it kept **114 of 313** columns.

Lasso's strongest weights (log scale ≈ percent): commercial zoning −0.19, living area +0.12, Crawfor and StoneBr neighbourhoods +0.10, quality +0.06, newly built +0.06.

Result: sum of |weights| 18.4 → 2.4, RMSE $22k → $19k, cheap-house miss 21% → 16%. Cheap houses are still ~2× harder than the rest.

## Bias and variance

- **Bias**: wrong on average because of the model's shape (the U-shape). More data doesn't fix it.
- **Variance**: the model changes a lot with different training houses (the pool weights). More data or a penalty fixes it.
- Lasso traded a little bias for a lot less variance.

**For rung 5:** the score to beat is **Lasso, RMSE $19,276**. Tune with cross-validation, not the test set (we've already looked at it four times).
