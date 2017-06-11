import tune_params as tp
import json
from ..features import data_reader as dr
import algorithms_factory as fac
from sklearn.metrics import classification_report
import sys


def load_tuning_results():
    with(open(tp.get_tuning_results_file_path(), "r")) as fp:
        return json.load(fp)


tuning_results = load_tuning_results()


def group_by_feature_size():
    grouped_by_feat_size = {}
    for algorithm in tuning_results:
        best_feat_size = tuning_results[algorithm][tp.features_size_key]
        if best_feat_size in grouped_by_feat_size:
            algorithms_for_feature_size = grouped_by_feat_size[best_feat_size]
        else:
            algorithms_for_feature_size = []
            grouped_by_feat_size[best_feat_size] = algorithms_for_feature_size

        algorithms_for_feature_size.append(algorithm)
    return grouped_by_feat_size


def print_output(line, raw_output=sys.stdout):
    print >> raw_output, line
    raw_output.flush()


def train_and_test():
    algorithms_by_feature_size = group_by_feature_size()
    for feat_size in algorithms_by_feature_size:
        train_X, train_y, test_X, test_y = dr.read_data(feat_size)
        algorithms = algorithms_by_feature_size[feat_size]
        for algorithm in algorithms:
            print_output("Starting {0} with feature size {1}".format(algorithm, feat_size))
            classifier = create_classifier(algorithm)
            # Train
            classifier.fit(train_X, train_y)
            # Test
            accuracy = classifier.score(test_X, test_y)
            y_pred = classifier.predict(test_X)
            print_output("Printing classification report.")
            print_output("Accuracy: {0}".format(accuracy))
            print_output(classification_report(test_y, y_pred))
            print_output("-----------------------------------------------------------------------")


def create_classifier(algorithm):
    if algorithm == "perceptron":
        regularization = tuning_results[algorithm]["regularization"]
        epochs = tuning_results[algorithm]["epochs"]
        learning_rate = tuning_results[algorithm]["learning_rate"]
        return fac.perc_factory((regularization, epochs, learning_rate))
    elif algorithm == "logistic_regression":
        regularization = tuning_results[algorithm]["regularization"]
        C = tuning_results[algorithm]["C"]
        return fac.logistic_regression_factory((regularization, C))
    elif algorithm == "svm":
        kernel = tuning_results[algorithm]["kernel"]
        C = tuning_results[algorithm]["C"]
        return fac.svm_factory((kernel, C))
    elif algorithm == "multinomial_naive_bayes":
        return fac.multinomial_naive_bayes_factory(())
    elif algorithm == "decision_tree":
        criterion = tuning_results[algorithm]["criterion"]
        max_depth = tuning_results[algorithm]["max_depth"]
        return fac.decision_tree_factory((criterion, max_depth))


