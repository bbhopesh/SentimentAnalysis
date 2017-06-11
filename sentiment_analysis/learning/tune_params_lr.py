import itertools
import parameter_tuner_util as pt
import algorithms_factory as fac
from ..features import data_reader as dr
import json
import os


tuning_results = {}
accuracy_key = "accuracy"
features_size_key = "featurs_size"


def perceptron(data_X, data_y, features_size, raw_output):
    print_output("\tPerceptron", raw_output)
    print_output("\t\tStarting 5-fold paramter tuning.", raw_output)
    accuracy_for_params, best_params = tune_perceptron(data_X, data_y)
    print_output("\t\tBest params are found to be {0} with accuracy {1}"
                            .format(best_params, accuracy_for_params[best_params]), raw_output)
    print_output("\t\tAccuracy for params {0}".format(accuracy_for_params), raw_output)
    perceptron_key = "perceptron"
    if perceptron_key in tuning_results:
        params = tuning_results[perceptron_key]
    else:
        params = {}
        params[accuracy_key] = -1
        tuning_results[perceptron_key] = params

    old_acc = params[accuracy_key]
    new_acc = accuracy_for_params[best_params]
    if new_acc > old_acc:
        params[accuracy_key] = new_acc
        params["regularization"] = best_params[0]
        params["epochs"] = best_params[1]
        params["learning_rate"] = best_params[2]
        params[features_size_key] = features_size


def tune_perceptron(data_X, data_y):
    # Various values of diff parameters that we want to choose from.
    regularization_set = (None, 'l2', 'l1', 'elasticnet')
    epochs_set = (5, 10, 15, 20)
    learning_rates_set = (1.5, 1, 0.25, 0.03, 0.005, 0.001)
    # All unique combination of these param values.
    params_set = itertools.product(regularization_set, epochs_set, learning_rates_set)
    # Tune
    accuracy_for_params, best_params = pt.tune_learning_algo(data_X, data_y, params_set, fac.perc_factory)
    return accuracy_for_params, best_params


def logistic_regression(data_X, data_y, features_size, raw_output):
    print_output("\tLogistic Regression", raw_output)
    print_output("\t\tStarting 5-fold paramter tuning.", raw_output)
    accuracy_for_params, best_params = tune_logistic_regression(data_X, data_y)
    print_output("\t\tBest params are found to be {0} with accuracy {1}"
                            .format(best_params, accuracy_for_params[best_params]), raw_output)
    print_output("\t\tAccuracy for params {0}".format(accuracy_for_params), raw_output)
    logistic_regression_key = "logistic_regression"
    if logistic_regression_key in tuning_results:
        params = tuning_results[logistic_regression_key]
    else:
        params = {}
        params[accuracy_key] = -1
        tuning_results[logistic_regression_key] = params

    old_acc = params[accuracy_key]
    new_acc = accuracy_for_params[best_params]
    if new_acc > old_acc:
        params[accuracy_key] = new_acc
        params["regularization"] = best_params[0]
        params["C"] = best_params[1]
        params[features_size_key] = features_size
    return best_params


def tune_logistic_regression(data_X, data_y):
    # Various values of diff parameters that we want to choose from.
    regularization_set = ('l2', 'l1')
    C = (1, 0.1, 0.01, 100, 1000, 10000, 100000)
    # All unique combination of these param values.
    params_set = itertools.product(regularization_set, C)
    # Tune
    accuracy_for_params, best_params = pt.tune_learning_algo(data_X, data_y, params_set, fac.logistic_regression_factory)
    return accuracy_for_params, best_params


def svm(data_X, data_y, features_size, raw_output):
    print_output("\tSVM", raw_output)
    print_output("\t\tStarting 5-fold paramter tuning.", raw_output)
    accuracy_for_params, best_params = tune_svm(data_X, data_y)
    print_output("\t\tBest params are found to be {0} with accuracy {1}"
                            .format(best_params, accuracy_for_params[best_params]), raw_output)
    print_output("\t\tAccuracy for params {0}".format(accuracy_for_params), raw_output)
    svm_key = "svm"
    if svm_key in tuning_results:
        params = tuning_results[svm_key]
    else:
        params = {}
        params[accuracy_key] = -1
        tuning_results[svm_key] = params

    old_acc = params[accuracy_key]
    new_acc = accuracy_for_params[best_params]
    if new_acc > old_acc:
        params[accuracy_key] = new_acc
        params["kernel"] = best_params[0]
        params["C"] = best_params[1]
        params[features_size_key] = features_size
    return best_params


def tune_svm(data_X, data_y):
    # Various values of diff parameters that we want to choose from.
    kernels_set = ('linear', 'rbf')
    C = (1, 0.1, 0.01, 100, 1000, 10000, 100000)
    # All unique combination of these param values.
    params_set = itertools.product(kernels_set, C)
    # Tune
    accuracy_for_params, best_params = pt.tune_learning_algo(data_X, data_y, params_set, fac.svm_factory)
    return accuracy_for_params, best_params


def multinomial_naive_bayes(data_X, data_y, features_size, raw_output):
    print_output("\tMultinomial naive bayes", raw_output)
    print_output("\t\tStarting 5-fold paramter tuning.", raw_output)
    accuracy_for_params, best_params = tune_multinomial_naive_bayes(data_X, data_y)
    print_output("\t\tBest params are found to be {0} with accuracy {1}"
                            .format(best_params, accuracy_for_params[best_params]), raw_output)
    print_output("\t\tAccuracy for params {0}".format(accuracy_for_params), raw_output)
    multinomial_naive_bayes_key = "multinomial_naive_bayes"
    if multinomial_naive_bayes_key in tuning_results:
        params = tuning_results[multinomial_naive_bayes_key]
    else:
        params = {}
        params[accuracy_key] = -1
        tuning_results[multinomial_naive_bayes_key] = params

    old_acc = params[accuracy_key]
    new_acc = accuracy_for_params[best_params]
    if new_acc > old_acc:
        params[accuracy_key] = new_acc
        params[features_size_key] = features_size
    return best_params


def tune_multinomial_naive_bayes(data_X, data_y):
    params_set = itertools.product()
    # Tune
    accuracy_for_params, best_params = pt.tune_learning_algo(data_X, data_y, params_set,
                                                             fac.multinomial_naive_bayes_factory)
    return accuracy_for_params, best_params


def decision_tree(data_X, data_y, features_size, raw_output):
    print_output("\tDecision Tree", raw_output)
    print_output("\t\tStarting 5-fold paramter tuning.", raw_output)
    accuracy_for_params, best_params = tune_decision_tree(data_X, data_y)
    print_output("\t\tBest params are found to be {0} with accuracy {1}"
                            .format(best_params, accuracy_for_params[best_params]), raw_output)
    print_output("\t\tAccuracy for params {0}".format(accuracy_for_params), raw_output)
    decision_tree_key = "decision_tree"
    if decision_tree_key in tuning_results:
        params = tuning_results[decision_tree_key]
    else:
        params = {}
        params[accuracy_key] = -1
        tuning_results[decision_tree_key] = params

    old_acc = params[accuracy_key]
    new_acc = accuracy_for_params[best_params]
    if new_acc > old_acc:
        params[accuracy_key] = new_acc
        params["criterion"] = best_params[0]
        params["max_depth"] = best_params[1]
        params[features_size_key] = features_size
    return best_params


def tune_decision_tree(data_X, data_y):
    criterion_set = ("gini", "entropy")
    max_depth_set = (10000, 100000, None)
    params_set = itertools.product(criterion_set, max_depth_set)
    # Tune
    accuracy_for_params, best_params = pt.tune_learning_algo(data_X, data_y, params_set,
                                                             fac.decision_tree_factory)
    return accuracy_for_params, best_params


def get_raw_output_file_path():
    return os.path.join(get_results_dir_path(), "tuning_raw_output_lr.txt")


def get_tuning_results_file_path():
    return os.path.join(get_results_dir_path(), "tuning_output_lr.json")


def get_results_dir_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def print_output(line, print_to):
    print >> print_to, line
    print_to.flush()


def main():
    # Tune
    with open(get_raw_output_file_path(), "w+") as raw_output:
        relevant_feature_sizes = (100, 500, 1000, 10000, 100000, None)
        #relevant_feature_sizes = (None,)
        #relevant_feature_sizes = (100,)
        for k in relevant_feature_sizes:
            print_output("Running with top {0} uni/bi/trigrams as features.".format(k), raw_output)
            train_X, train_y, test_X, test_y = dr.read_data(k)
            print_output("\tRead and trasformed data.", raw_output)

            logistic_regression(train_X, train_y, k, raw_output)

            '''
            best_params = logistic_regression(train_X, train_y, k, raw_output)
            print_output("\tStarting training.", raw_output)
            perc = fac.logistic_regression_factory(best_params)
            perc.fit(train_X,train_y)
            print_output("\tAccuracy   " + str(perc.score(test_X, test_y)*100) + "%", raw_output)
            print_output("--------------------------------------------------------------------", raw_output)
            '''
    # Save tuning results to file.
    with open(get_tuning_results_file_path(), "w+") as tuning_results_file:
        json.dump(tuning_results, tuning_results_file, sort_keys=True, indent=4)
