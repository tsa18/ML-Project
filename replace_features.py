import csv

# replace features in a file
def replace(data_file, cluster_file):
    with open(data_file) as f:
        csv_reader = csv.reader(f)
        data = list(csv_reader)
    idx_1, idx_2 = data[0].index('HeartDisease_1'), data[0].index('HeartDisease_2')

    new_data=[]
    header = data[0][:idx_1] + ['No_Heart_Disease','Heart_Disease_Type_0','Heart_Disease_Type_1'] +data[0][idx_2+1:]
    new_data.append(header)

    with open(cluster_file) as f:
        csv_reader = csv.reader(f)
        cluster_labels = list(csv_reader)
    
    heart_idx = 0
    for i in range(1,len(data)):
        new_feature = []
        # no heart disease
        if int(data[i][idx_1]) == 0:
            new_feature = [1,0,0]
        else:
            # print(heart_idx)
            new_feature = [0,1,0] if cluster_labels[heart_idx][0]==0 else [0,0,1]
            heart_idx+=1
        new_feature = data[i][:idx_1] + new_feature + data[i][idx_2+1:]
        new_data.append(new_feature)

    save_postfix = data_file.split('/')[-1]
    
    with open('data/replace/'+save_postfix,'w',newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(new_data)

# replace features in a file
def only_heart_features(data_file, cluster_file):
    with open(data_file) as f:
        csv_reader = csv.reader(f)
        data = list(csv_reader)
    idx_1, idx_2 = data[0].index('HeartDisease_1'), data[0].index('HeartDisease_2')

    new_data=[]
    header =  ['','No_Heart_Disease','Heart_Disease_Type_0','Heart_Disease_Type_1']
    new_data.append(header)

    with open(cluster_file) as f:
        csv_reader = csv.reader(f)
        cluster_labels = list(csv_reader)
    
    heart_idx = 0
    for i in range(1,len(data)):
        new_feature = []
        # no heart disease
        if int(data[i][idx_1]) == 0:
            new_feature = [1,0,0]
        else:
            # print(heart_idx)
            new_feature = [0,1,0] if cluster_labels[heart_idx][0]==0 else [0,0,1]
            heart_idx+=1
        new_feature = [ data[i][0] ] + new_feature 
        new_data.append(new_feature)

    save_postfix = data_file.split('/')[-1]
    
    with open('data/only_heart/'+save_postfix,'w',newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(new_data)
    

if __name__ == '__main__':
    replace('data/x_train.csv','data/heart/cluster_labels_train.csv')
    only_heart_features('data/x_train.csv','data/heart/cluster_labels_train.csv')
    
