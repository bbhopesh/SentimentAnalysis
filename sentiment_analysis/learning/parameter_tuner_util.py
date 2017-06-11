import numpy as np
from sklearn.model_selection import KFold


def tune_learning_algo(data_X, data_y, params_set, algo_factory, folds=5):
    """Tune learning algorithm and find parameters that give best accuracy.

    For each parameters, this function runs k-fold cross validation and finds average accuracy. The function then
    returns the parameter with best accuracy.
    - This is a generic method because responsibility of creating the algo for given set of paramters is left to the
     algo_factory passed to the function. For each params in params_set, we will call algo_factory(params) to instantiate
     algorithm. Hence, as long as algo_factory passed is compatible with each element of params_set, it will work.
     All is required from caller is to make sure that params_set is an iterable and algo_factory knows how to create
     learning algorithm given a single element of params_set
    :param data_X: data that should be used to run k-fold cross validation. shape should be [n_samples, n_features]
    :param data_y: data labels that should be used to run k-fold cross validation. shape should be [n_samples]
    :param params_set: set of parameters from which best parameter is to be tuned. We dont care about ftype of each
    individual element in this set/iterable as long as algo_factory knows how to create learning algorithm given one
    element of this iterable. It's just that each element should hashable as it will be key in returning dict. Read above.
    :param algo_factory: a function that takes parameters as input and creates a learning algorithm for the provided
    parameters. Read above.
    :param folds: num of folds in k-fold cross validation technique.
    :return: a 2-tuple where first element is a dictionary mapping parameters to accuracy and second element is
    parameters which gave best accuracy.
    """
    accuracy_for_params = {}
    best_accuracy = 0.0
    best_params = None

    # For all the available parameters, do a k-fold evaluation to get average accuracy.
    # Report the params with best accuracy.
    for params in params_set:
        # Do a k fold training and evaluation for this params and get average accuracy.
        accuracy = train_and_evaluate_k_fold(data_X, data_y, params, algo_factory, folds)
        # Save
        accuracy_for_params[params] = accuracy
        # Update the best parameters till now.
        if accuracy > best_accuracy:
            best_params = params
            best_accuracy = accuracy
    # Return.
    return accuracy_for_params, best_params


def train_and_evaluate_k_fold(data_X, data_y, params, algo_factory, folds=5):
    """Do a k-fold training and evaluation.

    We consider kth-fold data as test and rest as training data. We train fresh algorithm for each of these different
    training, test dataset combinations and report average accuracy.
    - Each time the learning algorithm has to be instantiated, algo_factory is called with params as first argument.
    :param data_X: data that should be used to run k-fold cross validation. shape should be [n_samples, n_features]
    :param data_y: data labels that should be used to run k-fold cross validation. shape should be [n_samples]
    :param params: parameters object that will be passed to algo_factory function to instantiate learning algorithm.
    :param algo_factory: learning algorithm factory.
    :param folds: num of folds in k-fold cross validation technique.
    :return: average accuracy of k-fold evaluation.
    """
    kf = KFold(n_splits=folds)

    accuracies = []
    for train_index, test_index in kf.split(data_X):
        learning_algo = algo_factory(params)
        train_data_X, train_data_y = data_X[train_index], data_y[train_index]
        test_data_X, test_data_y = data_X[test_index], data_y[test_index]
        # Train
        learning_algo.fit(train_data_X, train_data_y)
        # Evaluate
        accuracy = learning_algo.score(test_data_X, test_data_y)
        # Save
        accuracies.append(accuracy)
    # return average accuracy
    return np.mean(accuracies)
