from sklearn.linear_model import perceptron
import itertools
import parameter_tuner_util as pt
import sentiment_analysis.features.data_reader as dr


def tune_perceptron(data_X, data_y):
    # Various values of diff parameters that we want to choose from.
    regularization_set = (None, 'l2', 'l1', 'elasticnet')
    epochs_set = (5, 10, 15, 20)
    learning_rates_set = (1.5, 1, 0.25, 0.03, 0.005, 0.001)
    # All unique combination of these param values.
    params_set = itertools.product(regularization_set, epochs_set, learning_rates_set)
    # Tune
    accuracy_for_params, best_params = pt.tune_learning_algo(data_X, data_y, params_set, perc_factory)
    return accuracy_for_params, best_params


def perc_factory(params):
       return perceptron.Perceptron(penalty=params[0], n_iter=params[1], eta0=params[2])


def main():
    relevant_feature_sizes = (100, 500, 1000, 10000, 100000)
    for k in relevant_feature_sizes:
        print "Running with top {0} uni/bi/trigrams as features.".format(k)
        train_X, train_y, test_X, test_y = dr.read_data(k)
        print "\tRead and trasformed data."
        print "\tStarting 5-fold paramter tuning."
        accuracy_for_params, best_params = tune_perceptron(train_X, train_y)
        print "\tBest params are found to be {0} with accuracy {1}".format(best_params, accuracy_for_params[best_params])
        print "\tStarting training."
        perc = perc_factory(best_params)
        perc.fit(train_X,train_y)
        print "\tAccuracy   " + str(perc.score(test_X, test_y)*100) + "%"
        print "--------------------------------------------------------------------"
