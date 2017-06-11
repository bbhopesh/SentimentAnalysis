from sentiment_analysis.feature_engineering import feature_eng as feat_eng
import sklearn.feature_extraction.text as fe
from tempfile import TemporaryFile
import numpy as np


def sample_method_for_reference(train_size=None, test_size=None):
    # k is number of features tha we want.
    vocab = feat_eng.select_features_vocabulary(k=10000)
    # Get name of training files with label of each
    train_review_files, train_review_labels = feat_eng.get_train_review_files_with_label(train_size)
    test_review_files, test_review_labels = feat_eng.get_test_review_files_with_label(test_size)

    # Create vectorizer that will convert text to feature vectors.
    feat_vectorizer = fe.TfidfVectorizer(input='filename', ngram_range=(1,3), stop_words='english', vocabulary=vocab)
    # Convert to vectors.
    x_train = feat_vectorizer.transform(train_review_files)
    y_train = train_review_labels
    x_test = feat_vectorizer.transform(test_review_files)
    y_test = test_review_labels


    return x_train,y_train,x_test,y_test

    #save file
    np.save("outfile_x_train", x_train)
    np.save("outfile_y_train", y_train)
    np.save("outfile_x_test", x_test)
    np.save("outfile_y_test", y_test)


    # Tune and Train your algorithms after this step.
    # To tune use 5-fold method, there's a convenient way in scikit-learn to do 5-fold.
    # Evaluate your algorithms on test data.


