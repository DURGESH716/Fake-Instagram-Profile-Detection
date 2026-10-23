# Fake Instagram Profile Detection

A feed-forward neural network that classifies Instagram accounts as **fake** or **genuine** using profile-level metadata such as username patterns, bio length, and follower counts.

This repository accompanies my Kaggle notebook: [Fake Instagram Profile Detection Model](https://www.kaggle.com/code/durgeshrao9993/fake-instagram-profile-detection-model).

---

## Motivation

Fake and bot accounts inflate follower counts, distort engagement metrics, and are often used for spam or scams. This project tests whether simple, publicly visible profile features are enough to flag them automatically.

## Dataset

<!-- TODO: add the dataset source link and the exact train/test sizes -->

- **Source:** [Dataset Link](https://www.kaggle.com/code/durgeshrao9993/fake-instagram-profile-detection-model)
- **Split:** `insta_train.csv` (training) and `insta_test.csv` (held-out test)
- **Target:** `fake` (1 = fake, 0 = genuine)
- **Features (11):**

| Feature | Description |
|---|---|
| `profile pic` | Whether the account has a profile picture |
| `nums/length username` | Ratio of digits to username length |
| `fullname words` | Number of words in the full name |
| `nums/length fullname` | Ratio of digits to full-name length |
| `name==username` | Whether the full name matches the username |
| `description length` | Bio length in characters |
| `external URL` | Whether the bio contains a link |
| `private` | Whether the account is private |
| `#posts` | Number of posts |
| `#followers` | Number of followers |
| `#follows` | Number of accounts followed |

## Method

1. **Exploratory analysis:** class balance, missing values, feature distributions, and a correlation heatmap.
2. **Preprocessing:** features standardized with `StandardScaler`, fit on the training set only to avoid leakage. Labels are one-hot encoded.
3. **Model:** a fully connected network trained with Adam and categorical cross-entropy for 50 epochs, with 10% of the training data held out for validation.

```
Input (11) → Dense(50, ReLU) → Dense(150, ReLU) → Dropout(0.3)
           → Dense(150, ReLU) → Dropout(0.3)
           → Dense(25, ReLU)  → Dropout(0.3)
           → Dense(2, Softmax)
```

## Results

<!-- TODO: fill in from a fresh, clean run of the notebook -->

Evaluated on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 88% |
| Precision (fake) | 87% |
| Recall (fake) | 90% |
| F1 (fake) | 89% |

## Getting Started

```bash
git clone https://github.com/DURGESH716/Fake-Instagram-Profile-Detection.git
cd Fake-Instagram-Profile-Detection
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Then open the notebook and run all cells in order.

## Limitations

- The dataset is small, so results from a single train/test split may not generalize to Instagram at large.
- The features are static profile metadata. Sophisticated fake accounts that mimic genuine profiles would likely evade this approach.
- No comparison against simpler baselines (e.g. logistic regression, random forest) is included yet.

## License

MIT
