import csv
import pandas as pd
if __name__ == '__main__':

    features=[]
    labels= []
    src_file = './data/x_test.csv'
    dst_file = './data/heart/x_test_heart.csv'

    data = pd.read_csv(src_file, index_col=0)
    heart_labels = list(data['HeartDisease_1'])

    with open(src_file) as f:
        csv_reader = csv.reader(f)
        data = list(csv_reader)
    features.append(data[0])
    for i in range(len(heart_labels)):
        if heart_labels[i] == 1:
            features.append(data[i+1])

    with open(dst_file,'w',newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(features)

    # with open('data/y_train.csv') as f:
    #     csv_reader = csv.reader(f)
    #     data = list(csv_reader)

    # labels.append(data[0])
    # for i in range(len(heart_labels)):
    #     if heart_labels[i] == 1:
    #         labels.append(data[i+1])
    
    # with open('data/heart/y_train_heart.csv','w',newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerows(labels)
