import csv
import numpy as np
from sklearn.feature_selection import SelectKBest

def load_features_from_csv(file):
    features=[]
    with open(file) as f:
        f_scv = csv.reader(f)
        for i, row in enumerate(f_scv):
            if i==0: continue
            row = [float(x) for x in row]
            row.pop(0)
            feature = np.array(row)
            features.append(feature)
    features = np.array(features)
    return features

def load_labels_from_csv(file):
    labels=[]
    with open(file) as f:
        f_scv = csv.reader(f)
        for i, row in enumerate(f_scv):
            if i==0: continue
            label = int(row[1])
            labels.append(label)
    labels = np.array(labels)
    return labels


def get_samples(data_file, label_file):

    features = load_features_from_csv(data_file)
    labels = load_labels_from_csv(label_file)
    print(features.shape)
    print(labels.shape)
    return features, labels


def get_samples_by_cluster(data_file, label_file, cluster_file,cluster_num=0):

    features = load_features_from_csv(data_file)
    labels = load_labels_from_csv(label_file)
    clusters = []
    with open(cluster_file) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            clusters.append(int(row[0]))
    clusters=np.array(clusters)
    features = features[clusters==cluster_num]
    labels = labels[clusters==cluster_num]
    print(features.shape)
    print(labels.shape)
    strokes = labels[labels==1]
    onset_rate = strokes.shape[0] / labels.shape[0]
    print(f'cluster:{cluster_num}, onset rate:{onset_rate:.2f}')
    return features, labels
    


# test
if __name__ == '__main__':
    # get_samples('./data/x_train.csv','./data/y_train.csv')
    get_samples_by_cluster('./data/heart/x_train_heart.csv','./data/heart/y_train_heart.csv',
                           './data/heart/cluster_labels_train.csv', cluster_num=0)
