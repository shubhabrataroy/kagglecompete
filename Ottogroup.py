"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle

__author__ : Combined Abhisek Thakur's code plus Kaggle BTB
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# import data
train = pd.read_csv('/home/sroy/Desktop/KaggleDataSet/Otto/train.csv')
test = pd.read_csv('/home/sroy/Desktop/KaggleDataSet/Otto/test.csv')
sample = pd.read_csv('/home/sroy/Desktop/KaggleDataSet/Otto/sampleSubmission.csv')

def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss

    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll

def load_train_data(path=None, train_size=0.8):
    df = pd.read_csv('/home/sroy/Desktop/KaggleDataSet/Otto/train.csv')
    X = df.values.copy()
    np.random.shuffle(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))

## drop ids and get labels for the actual submission
#labels = train.target.values
#train = train.drop('id', axis=1)
#train = train.drop('target', axis=1)
#test = test.drop('id', axis=1)


X_train, X_valid, y_train, y_valid = load_train_data()

labels = y_train
train = X_train
test = X_valid

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train a random forest classifier
clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=400)
clf.fit(train, labels)

# predict on test set
preds = clf.predict_proba(test)

print(" -- Finished training.")

y_true = lbl_enc.fit_transform(y_valid)

score = logloss_mc(y_true, preds)
print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))
# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/home/sroy/Desktop/KaggleDataSet/Otto/benchmark.csv', index_label='id')
# it gives a bit higher error
