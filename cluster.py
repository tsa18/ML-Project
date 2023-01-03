from sklearn.cluster import KMeans
from load_data import *
from sklearn.metrics import silhouette_score

def cluster_heart(n_clusters, features):

    model = KMeans(n_clusters= n_clusters)
    y_pred = model.fit_predict(features) 
    silhouette_avg = silhouette_score(features, y_pred)
    print(f"n_clusters={n_clusters}, silhouette_score is {silhouette_avg:.2f}")

    y_pred = y_pred.tolist()
    with open('data/heart/cluster_labels_train.csv','w',newline='') as f:
        csv_writer = csv.writer(f)
        for l in y_pred:
            csv_writer.writerow([int(l)])

if __name__ =='__main__':
    features = load_features_from_csv('./data/heart/x_train_heart.csv')
    # for n_clusters in [2,3,5,8,10]:
    #     cluster_heart(n_clusters=n_clusters, features=features)

    cluster_heart(n_clusters=2,features=features)