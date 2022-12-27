from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
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


def cross_val(model,features, labels, cv=5):
    scores = cross_val_score(model, features, labels, cv=cv)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
