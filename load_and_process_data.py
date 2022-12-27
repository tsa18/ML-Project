import csv
import numpy as np
from sklearn.feature_selection import SelectKBest

def feature_selecion(X,y,score_func,k):
    selector = SelectKBest(score_func=score_func,k=k)
    X_new = selector.fit_transform(X, y)
    selected_features=selector.get_feature_names_out()
    idxs = [ int(x.replace("x","")) for x in selected_features]

    return X_new, idxs

def mask_test_features(X, idxs):
    return X[:,idxs]

def load_features_from_csv(file):
    features=[]
    with open(file) as f:
        f_scv = csv.reader(f)
        for i, row in enumerate(f_scv):
            if i==0: continue
            row = [float(x) for x in row]
            feature = np.array(row)
            features.append(feature)

    features = np.array(features)

    #feature scaling
    mean = np.mean(features,axis=0)
    std = np.std(features,axis=0)

    scaled_features = (features - mean) / (std+1e-18)
    return scaled_features


def load_labels_from_csv(file):
    labels=[]
    with open(file) as f:
        f_scv = csv.reader(f)
        for i, row in enumerate(f_scv):
            if i==0: continue
            label = int(row[0])
            labels.append(label)
    return labels


def get_samples(data_file, label_file):

    features = load_features_from_csv(data_file)
    labels = load_labels_from_csv(label_file)

    return features, labels

