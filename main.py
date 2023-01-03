from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import *
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

def load_train_data():
    features, labels = get_samples('./data/x_train.csv','./data/y_train.csv')
    print('-----------------------load data finished-----------------------')
    features, labels = under_sample(features,labels)
    print('-----------------------undersample finished-----------------------')
    return features, labels

def load_data_by_cluster(cluster_num):
    features, labels = get_samples_by_cluster('./data/heart/x_train_heart.csv','./data/heart/y_train_heart.csv',
                           './data/heart/cluster_labels_train.csv', cluster_num=cluster_num)
    features, labels = under_sample(features,labels)
    return features, labels

def train(model,features,labels):
    print('-----------------------start training-----------------------')
    model.fit(features,labels)
    y_pred = model.predict(features)
    report_scores(labels, y_pred, "train")
    report_AUC(labels, model.predict_proba(features)[:, 1])
    cross_val(model, features, labels, 5)

    # test(model)

def test(model):
    features = load_features_from_csv('./data/x_test.csv')
    y_pred = model.predict(features)
    path = './result/y_test.csv'
    idx = [ _ for _ in range(1,len(y_pred)+1)]
    data = list(zip(idx,y_pred))
    df = pd.DataFrame(data=data)
    if not os.path.exists(path):
        df.to_csv(path, header=['patient number', 'stroke'], index=False, mode='a')
    else:
        df.to_csv(path, header=False, index=False, mode='a')

def train_replaced_features():
    model = RandomForestClassifier(n_estimators=50, min_samples_leaf=3, max_depth=5)
    features, labels = get_samples('./data/replace/x_train.csv','./data/replace/y_train.csv')
    features, labels = under_sample(features,labels)
    print('--------------------------train with replaced features--------------------------')
    train(model, features,labels)

def train_only_heart_features():
    model = RandomForestClassifier(n_estimators=50, min_samples_leaf=3, max_depth=5)
    features, labels = get_samples('./data/only_heart/x_train.csv','./data/only_heart/y_train.csv')
    features, labels = under_sample(features,labels)
    print('--------------------------train only with heart disease features--------------------------')
    train(model, features,labels)

# RF, KNN, XGBoost, SVC
def main():
    # features, labels = load_train_data()
    # model = RandomForestClassifier(n_estimators=50, min_samples_leaf=3,max_depth=5)
    # model=SVC(C=1, kernel='linear')
    # model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=50)
    # train(model,features,labels)
    # test(model)

    # # train on different clusters
    # cluster_nums = [0,1]
    # for cluster_num in cluster_nums:
    #     features, labels = load_data_by_cluster(cluster_num=cluster_num)
    #     print(f'---------------------train on cluster {cluster_num}---------------------')
    #     train(model, features,labels)

    # train with the features replaced by new heart disease features.
    # train_replaced_features()

    # train only with the heart disease features
    # train_only_heart_features()

    pass
    
if __name__ == '__main__':
    main()