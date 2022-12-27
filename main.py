from load_and_process_data import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from utils import *


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

def test(model):
    features, labels = get_samples('./data/test1_icu_data.csv','./data/test1_icu_label.csv')
    y_pred = model.predict(features)
    # report_scores(labels,y_pred)
    # report_AUC(labels, model.predict_proba(features)[:, 1])

def main():
    train(100,3,'log2')

if __name__ == '__main__':
    main()