# Rung 1 — `spam/` · SMS spam classifier

Classify SMS messages as **spam** or **ham** (not-spam), start to finish:

## Run it

```bash
cd rungs/spam
source ../../.venv/bin/activate        # activate the venv (has sklearn, pandas)
python train.py                        # trains, prints the report, saves model.joblib
python score.py "You WON a free prize!!!"   # score one message
echo "see you at lunch" | python score.py   # or pipe one in
```

## Files

| File                     | Role                                                     |
| ------------------------ | -------------------------------------------------------- |
| `data/SMSSpamCollection` | the data: 5,572 rows of `label <tab> message`            |
| `train.py`               | **learns** a model from the data, saves `model.joblib`   |
| `score.py`               | **uses** the saved model to score a new message          |
| `model.joblib`           | the trained model (TF-IDF + Naive Bayes), frozen to disk |

The data is imbalanced: **13% spam, 87% ham** — which is why accuracy alone is a lie (see below).

---

## 1. Train/test split — the exam analogy

We hide 20% of the messages in a vault **before** training. The model learns from the other 80%, then we grade it on the 20% it has never seen.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 2. TF-IDF — turning words into numbers

TF-IDF turns each message into a row of numbers, where each number says _"how important is this word to this message?"_

**Why TF-IDF and not just word counts?** Raw counts let common words ("the", "to") drown out rare informative ones. TF-IDF fixes that with two parts:

- **TF (Term Frequency)** — how often the word appears _in this message_.
- **IDF (Inverse Document Frequency)** — how _rare_ the word is _across all messages_.

**TF-IDF = TF × IDF** → a word scores high only when it's **frequent here AND rare overall**:

| Word    | Across all messages                | TF-IDF                    |
| ------- | ---------------------------------- | ------------------------- |
| `the`   | in almost every message → tiny IDF | **low** (ignored)         |
| `prize` | in only a few messages → big IDF   | **high** (strong feature) |

**Important:** high IDF means _rare_, **not** _spammy_. "Jurong" is rare (high IDF) but appears in ham. TF-IDF only makes words **distinctive** (label-blind). _Which_ distinctive words signal spam is learned separately — by Naive Bayes.

---

## 3. Naive Bayes — turning numbers into a spam/ham decision

**The question:** given the words in a message, is it more likely spam or ham?

**Bayes' rule** flips that into things we can count:

```
P(spam | words)  ∝  P(words | spam)  ×  P(spam)
                      ↑                  ↑
            how spammy the words look   base rate of spam (prior = 13%)
```

Compute this for **spam** and for **ham**, pick the bigger.

**The "naive" part:** a message has many words; the exact combination may never have occurred before. So we make a bold (wrong) assumption — pretend words are **independent** given the class — and just **multiply** each word's probability:

```
P(words | spam) ≈ P(w₁|spam) × P(w₂|spam) × P(w₃|spam) × ...
```

Wrong (words do correlate), but it works great — because for classification we only need the spam score to **beat** the ham score, not to be exact.

**Worked example — "free prize":**

```
spam score = P(spam) × P("free"|spam) × P("prize"|spam)   → large
ham score  = P(ham)  × P("free"|ham)  × P("prize"|ham)    → tiny
spam ≫ ham  →  SPAM
```

**How each P(word | spam) is calculated — just counting:**

```
                   count of "free" across ALL spam messages
P("free" | spam) = ─────────────────────────────────────────
                   total word-count across ALL spam messages
```

e.g. "free" appears 500× in 10,000 spam words → 0.05. "free" appears 90× in 90,000 ham words → 0.001. So "free" is **50× more likely in spam** — that ratio _is_ the learned signal. Nobody labeled "free" as spammy; it fell out of the counts.

**Training = counting; predicting = look up and multiply.**

Two numerical fixes that keep it working:

- **Laplace smoothing (`alpha=1`):** add a tiny count to every word so no probability is ever 0 (one zero would veto the whole product). Handles words _in_ the vocabulary but unseen in one class.
- **Log-space:** multiplying hundreds of tiny numbers underflows to 0, so it adds `log`s instead. Same comparison, no underflow.

---

## 4. The metrics — a report card, not one number

All of them come from the **confusion matrix** — the scoreboard crossing _what was true_ against _what the model said_ (numbers from this model's test set):

- **TN** — True negatives
- **FP** — False positives
- **FN** — False negatives
- **TN** — True negatives

```
                   model said HAM    model said SPAM
   actually HAM        966                0        ← TN=966, FP=0
   actually SPAM        33              116        ← FN=33,  TP=116
```

| Term   | Meaning                                           | This model |
| ------ | ------------------------------------------------- | ---------- |
| **TN** | ham correctly called ham ✅                       | 966        |
| **FP** | ham wrongly called spam ❌ (good mail lost)       | 0          |
| **FN** | spam wrongly called ham ❌ (spam slipped through) | 33         |
| **TP** | spam correctly caught ✅                          | 116        |

Diagonal = correct, off-diagonal = mistakes. Every metric is just a way of dividing these four cells:

- **Accuracy** — of **all** messages, how many right? `(TP+TN)/all` = 1082/1115 = **97%**.
  ⚠️ Lies on imbalanced data: "always say ham" scores 86.6% while catching _zero_ spam.

- **Precision** — of the ones it **called spam**, how many really were? `TP/(TP+FP)` = 116/116 = **100%**.
  Guards against **false positives**. _Smoke alarm: when it goes off, is there really a fire?_

- **Recall** — of all the **real spam**, how many did it catch? `TP/(TP+FN)` = 116/149 = **78%**.
  Guards against **false negatives**. _Smoke alarm: of all real fires, how many did it detect?_

- **F1** — one number balancing precision & recall (harmonic mean): **0.875**.
  Punishes imbalance — only high when _both_ are high. The go-to single metric for imbalanced tasks.

**The trade-off:** you usually can't max precision _and_ recall. This model is **cautious** — perfect precision (never loses good mail) but 78% recall (misses some spam). Which to favor depends on **which mistake hurts more**:

- Spam filter → losing a real email is worse than a little junk → favor **precision**.
- Cancer screening → missing a sick patient is catastrophic → favor **recall**.

Tuning that balance (moving the decision threshold) is **rung 2 — `report-card/`**.

---

Done when: beats the baseline (97% > 86.6% ✅), reports more than accuracy ✅, no leakage (TF-IDF fit on train only, inside the Pipeline ✅).\_
