from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).parent
DATA = HERE.parent / "spam" / "data" / "SMSSpamCollection"


def hand_features(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "length": texts.str.len(),  # characters per message
            "n_digits": texts.str.count(r"\d"),  # how many 0-9 characters
            "caps_ratio": texts.str.count(r"[A-Z]")
            / texts.str.len(),  # uppercase letters / length
        }
    )


df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"])
feats = hand_features(df["text"])
feats["label"] = df["label"]
# print(feats.head(10))
print(feats.groupby("label").mean())

# 1. X and y
X = pd.concat([df[["text"]], hand_features(df["text"])], axis=1)
y = (df["label"] == "spam").astype(int)

# 2. SAME split as rung 1 — same test messages, fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. router + same model
prep = ColumnTransformer(
    [
        ("words", TfidfVectorizer(lowercase=True, stop_words="english"), "text"),
        ("shape", MinMaxScaler(), ["length", "n_digits", "caps_ratio"]),
    ]
)
model = make_pipeline(prep, MultinomialNB())

# 4. train + judge
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


# 3–4. ablation: drop one feature at a time, same model, same split
ALL = ["length", "n_digits", "caps_ratio"]

experiments = {"all 3": ALL}
for f in ALL:
    experiments[f"without {f}"] = [c for c in ALL if c != f]

for name, cols in experiments.items():
    prep = ColumnTransformer(
        [
            (
                "words",
                TfidfVectorizer(lowercase=True, stop_words="english"),
                "text",
            ),
            ("shape", MinMaxScaler(), cols),
        ]
    )
    model = make_pipeline(prep, MultinomialNB())
    model.fit(X_train, y_train)
    cm = confusion_matrix(y_test, model.predict(X_test))
    print(f"{name:22} slipped={cm[1, 0]:3}  false_alarms={cm[0, 1]}")

# 5. feature importance — what did the final model rely on?
final = make_pipeline(
    ColumnTransformer(
        [
            ("words", TfidfVectorizer(lowercase=True, stop_words="english"), "text"),
            ("shape", MinMaxScaler(), ALL),
        ]
    ),
    MultinomialNB(),
)
final.fit(X_train, y_train)

names = final[0].get_feature_names_out()  # "words__free", "shape__n_digits", ...
nb = final[-1]
ratio = np.exp(nb.feature_log_prob_[1] - nb.feature_log_prob_[0])  # spam ÷ ham
imp = pd.Series(ratio, index=names).sort_values()

print("\ntop 10 SPAM signals:\n", imp.tail(10).round(1))
print("\ntop 10 HAM signals:\n", imp.head(10).round(3))
print("\nyour hand features:\n", imp[[f"shape__{f}" for f in ALL]].round(2))
print(
    "\ntheir rank (1 = most spammy):\n",
    imp.rank(ascending=False)[[f"shape__{f}" for f in ALL]],
)
