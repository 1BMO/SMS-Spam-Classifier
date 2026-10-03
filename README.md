# SMS Spam Classifier

A small machine-learning project that classifies SMS messages as spam or not spam, built while studying offensive AI / adversarial machine learning. The goal is to understand how a text classifier works end to end, so I can later test how such models are evaded.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) from the UCI Machine Learning Repository (5,572 labeled messages; 403 exact duplicates were removed, leaving 5,169).

## Pipeline

1. Remove duplicates
2. Lowercase, replace URLs and numbers with placeholder tokens, strip other symbols (`$` and `!` are kept as spam signals)
3. Tokenize (NLTK), remove stop words, stem (Porter)
4. Split 80/20 (stratified, `random_state=42`)
5. `CountVectorizer` with unigrams and bigrams
6. Multinomial Naive Bayes, `alpha` tuned with 5-fold `GridSearchCV` (scoring: F1) on the training part only
7. Evaluate once on the held-out test part
8. Refit the best model on all data and save it with `joblib`

## Results

Evaluated on a held-out test set (20% of the deduplicated data, stratified, 1,034 messages, 131 of them spam), default threshold 0.5. Metrics for the spam class:

| Metric | Value |
|---|---|
| Precision | 0.83 |
| Recall | 0.93 |
| F1 | 0.88 |
| Overall accuracy | 0.97 |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted not spam | Predicted spam |
|---|---|---|
| **Actual not spam** | 878 | 25 |
| **Actual spam** | 9 | 122 |

The model catches most spam (9 missed out of 131) but flags 25 legitimate messages as spam. Class priors are not used (`fit_prior=False`), which pushes recall up at the cost of precision. The decision threshold could be raised to reduce false positives.

The saved model is trained on all the data, so the table above comes from the held-out split, not from the saved model.

## Limitations

- Trained on English SMS from one dataset, so it will not generalize well to email or other languages.
- Rule-based preprocessing and word counts make it easy to fool with obfuscation (misspellings, character substitution, padding with benign words).
- The decision threshold is set manually (`--threshold`) and was not tuned on a validation set.
- Naive Bayes probabilities are extreme (close to 0 or 1), so they are not real confidence.

## Next steps

- Test evasion techniques against my own model and document how they affect the predictions
- Try TF-IDF and a linear model for comparison
- Tune the threshold on a validation set

## Run

```bash
git clone https://github.com/1BMO/SMS-Spam-Classifier.git
cd SMS-Spam-Classifier
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python spam_classifier.py
```

The first run downloads the dataset and the NLTK data (internet required). It trains, evaluates, saves `spam_detection_model.joblib` and classifies a few example messages.

Options:

```bash
python spam_classifier.py --threshold 0.3     # flag a message when p(spam) is above 0.3
python spam_classifier.py --data-dir data/    # where the dataset is stored
python spam_classifier.py --model-path m.joblib
```

## Using the saved model

The saved pipeline contains the vectorizer and the classifier but **not** the text preprocessing, so new messages must go through the same `Preprocessor` first:

```python
import joblib
from spam_classifier import Preprocessor, classify

model = joblib.load("spam_detection_model.joblib")
print(classify(model, Preprocessor(), ["FREE entry! Text WIN to 80085 now"], threshold=0.5))
```

> Note: only load `.joblib` files from sources you trust, since they can execute code when loaded.
