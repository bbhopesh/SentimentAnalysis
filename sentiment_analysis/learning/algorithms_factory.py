from sklearn.linear_model import perceptron
from sklearn.linear_model import logistic
from sklearn import svm
from sklearn import naive_bayes
from sklearn import tree


def perc_factory(params):
    return perceptron.Perceptron(penalty=params[0], n_iter=params[1], eta0=params[2])


def logistic_regression_factory(params):
    return logistic.LogisticRegression(penalty=params[0], C=params[1])


def svm_factory(params):
    return svm.SVC(kernel=params[0], C=params[1])


def multinomial_naive_bayes_factory(params):
    return naive_bayes.MultinomialNB()


def decision_tree_factory(params):
    return tree.DecisionTreeClassifier(criterion=params[0], max_depth=params[1])
