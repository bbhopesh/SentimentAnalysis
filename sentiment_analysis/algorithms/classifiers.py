from sklearn.linear_model import perceptron
from sklearn import svm
import numpy as np
from sentiment_analysis.learning import use_features_example as u_feat_eng
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, classification_report, confusion_matrix, average_precision_score


# train_x = np.load('outfile_x_train.npy')
# train_y = np.load('outfile_y_train.npy')
# test_x = np.load('outfile_x_test.npy')
# test_y = np.load('outfile_y_test.npy')

#### Perceptron

train_x,train_y,test_x,test_y = u_feat_eng.sample_method_for_reference()
print('data loaded')
net = perceptron.Perceptron(penalty='l2', n_iter=10)
net.fit(train_x,train_y)
print ("Accuracy   " + str(net.score(test_x, test_y)*100) + "%")




# def simple_any(clf, train_features, train_labels, test_features, test_true_labels):
#     start_time = time.time()
#     stats = dict()
#     clf = clf.fit(train_features, train_labels)
#
#     train_predicted_labels = clf.predict(train_features)
#     test_predicted_labels = clf.predict(test_features)
#
#     y_score = clf.predict_proba(test_features)
#
#     stats['time'] = time.time() - start_time
#     stats['false_pv_arr'], stats['true_pv_arr'], thresholds = roc_curve(test_true_labels, y_score[:, 1])
#     stats['precision_arr'], stats['recall_arr'], prthres = precision_recall_curve(test_true_labels, y_score[:, 1])
#     stats['auc'] = auc(stats['false_pv_arr'], stats['true_pv_arr'])
#     stats['aver_pr'] = average_precision_score(test_true_labels, y_score[:, 1])
#     stats['precision_recall_f1'] = classification_report(test_true_labels, test_predicted_labels)
#     stats['confusion_matrix'] = confusion_matrix(test_true_labels, test_predicted_labels)
#     stats['test_accuracy'] = accuracy_score(test_true_labels, test_predicted_labels)
#     stats['train_accuracy'] = accuracy_score(train_labels, train_predicted_labels)
#     return stats
# #### SVM - S (non-linear)
#
# tuned_parameters = [{'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}]
# clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
# svm_s = simple_any(clf, train_x, train_y, test_x, test_y)
# print (svm_s['precision_recall_f1'])
# print (svm_s['time'])
# print (svm_s['test_accuracy'])
# print (svm_s['auc'])
# print (clf.best_params_)
#
#
# #### SVM - Linear
#
# tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
# clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='roc_auc', verbose=10)
# svml_s = simple_any(clf, train_x, train_y, test_x, test_y)
# print (svml_s['precision_recall_f1'])
# print (svml_s['time'])
# print (svml_s['test_accuracy'])
# print (svml_s['auc'])
# print (clf.best_params_)


