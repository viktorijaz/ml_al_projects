# Rung 3 - `features/` · the data is the model

Rung 1 misclassified **33** spam messages as ham.
This rung shows why we can do better with the same data, same model - just by improving the input to the model.

**Feature engineering** - most real-world gains come from better _features_, not fancier algorithms. The data is the model.
TF-IDF sees only which words appear. But there are also other aspects of the possible spam, which we will call **features**.
Example:

```
A: "win a free prize now"
B: "WIN A FREE PRIZE NOW!!! 🎉 £1000 → 0906170146"
```

To TF-IDF those 2 messages are very similar. But a human sees more: shouting capitals, exclamation marks, a currency sign, a long number.

## Results:

Rung 1 misclassified **33** of 149 spam messages as ham. Rung 3, with the same model plus hand-crafted features, misclassified **27**, with still 0 false alarms.

## But are all features important?

We run an experiment and omit one feature at a time, to see if it changes the results:

| experiment         | spam slipped through | false alarms |
| ------------------ | -------------------: | -----------: |
| all 3              |                   27 |            0 |
| without length     |                   27 |            0 |
| without n_digits   |                   35 |            0 |
| without caps_ratio |                   26 |            0 |

Result: only n_digits feature is relevant to our model

## How important are the features:

| rank | word       | spam ÷ ham | note                                       |
| ---: | ---------- | ---------: | ------------------------------------------ |
|    1 | claim      |       30.9 | real signal                                |
|    2 | prize      |       25.3 | real signal                                |
|    3 | 150p       |       19.5 | artifact (UK, 2010)                        |
|    4 | www        |       18.8 | real signal                                |
|    5 | uk         |       17.1 | artifact (UK)                              |
|    6 | tone       |       16.6 | artifact (ringtones)                       |
|    7 | 18         |       15.8 | adult-content age gate, likely artifact    |
|    8 | guaranteed |       15.5 | real signal                                |
|    9 | 1000       |       14.7 | money amount                               |
|   10 | cs         |       14.7 | artifact (UK "customer service" shortcode) |

The table shows the top spam words. Are they still relevant in 2026 - some of them are not - like tone, 150p.
A model relying on them would generalize poorly to spam from other countries or years
