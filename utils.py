from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate
from imblearn.under_sampling import RandomUnderSampler 
from collections import Counter

def report_scores(y, y_pred, name=''):
    f1 = f1_score(y, y_pred)
    recall = recall_score(y,y_pred)
    precision = precision_score(y,y_pred)
    acc = accuracy_score(y, y_pred)
    print(f'{name}: accuracy:{acc:0.2f}, F1 score:{f1:0.2f}, recall:{recall:0.2f}, precision:{precision:0.2f}')

def report_AUC(y, y_score):
    auc = roc_auc_score(y, y_score)
    print(f"auc:{auc}")

def cross_val(model,features, labels, cv=5):
    print('-----------------------start cross validation-----------------------')
    cv_results = cross_validate(model, features, labels, cv=cv, scoring=['accuracy','f1','recall','precision'])
    acc = cv_results['test_accuracy'].mean()
    f1 = cv_results['test_f1'].mean()
    recall = cv_results['test_recall'].mean()
    precision = cv_results['test_precision'].mean()
    print(f'cross val: accuracy:{acc:0.2f}, F1 score:{f1:0.2f}, recall:{recall:0.2f}, precision:{precision:0.2f}')

def under_sample(X,y):
    rus = RandomUnderSampler(random_state=0) 
    print(Counter(y))
    X_resampled, y_resampled = rus.fit_resample(X,y)
    print(Counter(y_resampled))
    return X_resampled, y_resampled
