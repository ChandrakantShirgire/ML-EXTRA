---
title: "ML: Naive Bayes vs Logistic Regression"
subtitle: "Fake Job Posting Detector"
---

## Setup

2,000 postings (stratified sample, 97 fake) from the Real or Fake Job Posting Prediction dataset. Title, description and requirements joined, cleaned, and converted with TF-IDF (20,000 features, unigrams and bigrams). Split 80:20: 1,600 train, 400 test (19 fake). Both models use the same features.

## Comparison table (fake class, test set)

| Model | Accuracy | Precision | Recall | F1 |
|:--|:-:|:-:|:-:|:-:|
| Naive Bayes | 0.965 | 0.778 | 0.368 | 0.500 |
| Logistic Regression | 0.980 | 0.867 | 0.684 | 0.765 |

| Confusion matrix | TN | FP | FN | TP |
|:--|:-:|:-:|:-:|:-:|
| Naive Bayes | 379 | 2 | 12 | 7 |
| Logistic Regression | 379 | 2 | 6 | 13 |

## Which is better

**Logistic Regression.** It is higher on all four measures. It catches 13 of 19 fake postings against 7 for Naive Bayes, with the same 2 false alarms. Naive Bayes misses almost two thirds of the scams. Logistic Regression also uses balanced class weights, which makes it pay more attention to the rare fake class.

## Why accuracy alone is not enough

Only about 5% of postings are fake. A model that calls every posting real scores 95.2% accuracy and catches no scam. Naive Bayes already scores 96.5%, close to that useless baseline, yet its recall is only 37%. Precision (how many flagged postings are truly fake), recall (how many fake postings are caught) and F1 (their balance) on the fake class show the real quality. For a scam detector, recall matters most because a missed fake posting harms a job seeker.

*Note: the test set has only 19 fake postings, so scores can shift with a different split.*
