import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

# same data and split as rung 4, so the scores compare
df = fetch_openml(data_id=42165, as_frame=True).frame
partial_giants = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300_000)
df = df[~partial_giants]  # 2 unfinished houses sold cheap
df = df.drop(columns="Id")
df["MSSubClass"] = df["MSSubClass"].astype(str)

X = df.drop(columns="SalePrice")
y = np.log(df["SalePrice"])  # trees learn log price too
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# trees need no scaling; categories become numbered codes, one column each
prep = ColumnTransformer(
    [
        ("numbers", SimpleImputer(strategy="median"), make_column_selector(dtype_include="number")),
        (
            "categories",
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="None"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
            make_column_selector(dtype_exclude="number"),
        ),
    ],
    verbose_feature_names_out=False,
)


def dollars(y_log, pred_log):
    return root_mean_squared_error(np.exp(y_log), np.exp(pred_log))


