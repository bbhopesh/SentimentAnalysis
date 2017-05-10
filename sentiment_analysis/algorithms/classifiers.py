from sklearn.linear_model import LogisticRegression, perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
import feature_generation as feat_gen
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, classification_report, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.externals import joblib


# #load data
# df = pd.read_csv('feautre_file_test.txt', sep='|', names=['text', 'label'])
# test_x = df['text']
# test_y = df['label']
#
# df_1 = pd.read_csv('feautre_file_train.txt', sep='|', names=['text', 'label'])
# train_x = df_1['text']
# train_y = df_1['label']

#### Perceptron

train_x,train_y,test_x,test_y = feat_gen.select_features_vocabulary()
print('data loaded')
net = perceptron.Perceptron(penalty='l2', n_iter=10)
net.fit(train_x,train_y)
print ("Accuracy   " + str(net.score(test_x, test_y)*100) + "%")
joblib.dump(net, 'perceptron.pkl')


def simple_any(clf, train_features, train_labels, test_features, test_true_labels):
    start_time = time.time()
    stats = dict()
    clf = clf.fit(train_features, train_labels)

    train_predicted_labels = clf.predict(train_features)
    test_predicted_labels = clf.predict(test_features)

    y_score = clf.predict_proba(test_features)

    stats['time'] = time.time() - start_time
    stats['false_pv_arr'], stats['true_pv_arr'], thresholds = roc_curve(test_true_labels, y_score[:, 1])
    stats['precision_arr'], stats['recall_arr'], prthres = precision_recall_curve(test_true_labels, y_score[:, 1])
    stats['auc'] = auc(stats['false_pv_arr'], stats['true_pv_arr'])
    stats['aver_pr'] = average_precision_score(test_true_labels, y_score[:, 1])
    stats['precision_recall_f1'] = classification_report(test_true_labels, test_predicted_labels)
    stats['confusion_matrix'] = confusion_matrix(test_true_labels, test_predicted_labels)
    stats['test_accuracy'] = accuracy_score(test_true_labels, test_predicted_labels)
    stats['train_accuracy'] = accuracy_score(train_labels, train_predicted_labels)
    return stats
#### SVM - S (non-linear)

# tuned_parameters = [{'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}]
# clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
# svm_s = simple_any(clf, train_x,train_y,test_x,test_y)
# print (svm_s['precision_recall_f1'])
# print (svm_s['time'])
# print (svm_s['test_accuracy'])
# print (svm_s['auc'])
# print (clf.best_params_)

#
# #### SVM - Linear
#
# tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
# clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
# svml_s = simple_any(clf, train_x,train_y,test_x,test_y)
# print(svml_s['precision_recall_f1'])
# print(svml_s['time'])
# print(svml_s['test_accuracy'])
# print(svml_s['auc'])
# print(clf.best_params_)

####### Logistic Regression

tuned_parameters = {'C': [0.01, 0.1, 1, 10, 100] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
lr_result = simple_any(clf, train_x,train_y,test_x,test_y)
print (lr_result['precision_recall_f1'])
print (lr_result['time'])
print (lr_result['test_accuracy'])
print (lr_result['auc'])
print (clf.best_params_)

####### Random Forest

tuned_parameters = {'n_estimators': [10, 50, 100, 300] }
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
rf_result = simple_any(clf, train_x,train_y,test_x,test_y)
print (rf_result['precision_recall_f1'])
print (rf_result['time'])
print (rf_result['test_accuracy'])
print (rf_result['auc'])
print (clf.best_params_)


####### KNN

tuned_parameters = {'n_neighbors': [5, 10, 50, 100] }
clf = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree'), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
knn_result = simple_any(clf, train_x,train_y,test_x,test_y)
print (knn_result['precision_recall_f1'])
print (knn_result['time'])
print (knn_result['test_accuracy'])
print (knn_result['auc'])
print (clf.best_params_)

####### adaboost

tuned_parameters = {'n_estimators': [10, 50, 100, 300, 500, 1000], 'learning_rate': [0.1, 0.3, 0.5, 1, 2] }
clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
ab_result = simple_any(clf, train_x,train_y,test_x,test_y)
print (ab_result['precision_recall_f1'])
print (ab_result['time'])
print (ab_result['test_accuracy'])
print (ab_result['auc'])
print (clf.best_params_)

#### DT
dt_stats = simple_any(DecisionTreeClassifier(), train_x,train_y,test_x,test_y)
print (dt_stats['precision_recall_f1'])
print (dt_stats['time'])
print (dt_stats['test_accuracy'])
print (dt_stats['auc'])

# Run MNB
mnb_result = simple_any(MultinomialNB(), train_x,train_y,test_x,test_y)
print (mnb_result['precision_recall_f1'])
print (mnb_result['time'])
print (mnb_result['test_accuracy'])
print (mnb_result['auc'])

def curve_it():
    colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'magenta']
    stats = [(ab_result, "AB"),
    (knn_result, "KNN"),
    (rf_result, "RF"),
    (dt_stats, "DT"),
    (svml_s, "SVML"),
    (lr_result, "LR"),
    (mnb_result, "NB")]

    plt.figure()
    lw = 2
    for i, stat_name in enumerate(stats):
        stat = stat_name[0]
        plt.plot(stat['false_pv_arr'], stat['true_pv_arr'],
             color=colors[i], lw=lw, label='{} (AUC = {})'.format(stat_name[1], round(stat['auc'],2)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

curve_it()