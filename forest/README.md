# Rung 5 - `forest/` · ensembles

Same houses, same split and same log price as rung 4, so the scores compare.
The score to beat: **Lasso, RMSE $19,276**.

## Step 1: one tree (`tree.py` → `tree.png`)

A tree is a chain of yes/no questions. Training picks **which question to ask first** (the column and cut that split the houses into the most similar-price groups), then the next one in each branch. Each leaf predicts the **average price** of the training houses that landed there. No weights.

| depth | leaves | train RMSE | test RMSE |
| ----: | -----: | ---------: | --------: |
|     2 |      4 |    $50,663 |   $47,374 |
|     6 |     63 |    $22,350 |   $34,320 |
|  none |   1132 |         $0 |   $35,980 |

Shallow = too simple (high bias). Deep = memorised the training houses (high variance).

## Step 2: random forest (`forest.py`)

500 deep trees, each on a random sample of the training houses and a random third of the columns, averaged.
Their mistakes cancel out: **$22,491**.
**Out-of-bag**: each house is scored only by the trees that skipped it, a free check without the test set ($27,502).

## Step 3: XGBoost (`boost.py`)

Trees built one after another, each fixing a little of what the previous ones still get wrong.
**Early stopping** on a validation set (20% of the training houses) stopped it at 646 trees: **$21,113**.

## Step 4: tuning (`tune.py`)

12 settings (depth × learning rate), each scored with **5-fold cross-validation on the training houses**, then the best retrained on all of them and tested once.
Small trees won: **depth 3, rate 0.03, 828 trees → $19,509**.

Validation is the practice exam (use it to decide), the test set is the real exam (use it once).

## Step 5: which columns matter (`importance.py` → `importance.png`)

**Permutation importance**: shuffle one column, see how much worse the model gets.
Living area (+7%) and quality (+6.5%) dominate, then basement size, age and condition.

XGBoost's built-in importance put `ExterQual` first: it overlaps with `OverallQual`, and the built-in score credits whichever one the trees happened to split on. Trust permutation.

## Results

| model          | test RMSE |
| -------------- | --------: |
| single tree    |   $34,320 |
| forest         |   $22,491 |
| plain linear   |   $22,167 |
| XGBoost, tuned |   $19,509 |
| **lasso**      | **$19,276** |

One tree is weak, many trees are strong, and on smooth house prices the best trees only **tie** a well-built straight line.
