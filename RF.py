from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from load_and_process_data import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score

def report_scores(y, y_pred, name=''):
    f1 = f1_score(y, y_pred)
    recall = recall_score(y,y_pred)
    precision = precision_score(y,y_pred)
    acc = accuracy_score(y, y_pred)
    print(f'{name}: accuracy:{acc}')
    print(f'{name}: F1 score:{f1}, recall:{recall}, precision:{precision}')

def report_AUC(y, y_score):
    # y_score = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_score)
    print(f"auc:{auc}")

def train(n_estimator, min_samples_leaf, max_features):
    features, labels = get_samples('./data/train1_icu_data.csv','./data/train1_icu_label.csv')

    model = RandomForestClassifier(n_estimators=n_estimator,min_samples_leaf=min_samples_leaf,max_features=max_features)
    model.fit(features,labels)
    y_pred = model.predict(features)
    print(f"n_estimator:{n_estimator}, min_samples_leaf:{min_samples_leaf}, max_features:{max_features}.")
    report_scores(labels, y_pred, "train")
    # k = 10
    # print(model.feature_importances_.argsort()[-k:])
    report_AUC(labels, model.predict_proba(features)[:, 1])
    cross_val(model, features, labels, 5)
    # test(model)

def cross_val(model,features, labels, cv=5):
    scores = cross_val_score(model, features, labels, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

def test(model):

    features, labels = get_samples('./data/test1_icu_data.csv','./data/test1_icu_label.csv')
    y_pred = model.predict(features)
    # report_scores(labels,y_pred)
    # report_AUC(labels, model.predict_proba(features)[:, 1])

def main():

    train(100,3,'log2')

if __name__ == '__main__':
    main()