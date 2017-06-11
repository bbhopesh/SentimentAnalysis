import sentiment_analysis.features.feature_eng as feat_eng
import sklearn.feature_extraction.text as fe
import os
import numpy as np


def read_data(k):
    # k is number of features tha we want.
    vocab = feat_eng.select_features_vocabulary(k)
    # Get name of training files with label of each
    train_review_files, train_review_labels = get_train_review_files_with_label()
    test_review_files, test_review_labels = get_test_review_files_with_label()
    # Create vectorizer that will convert text to feature vectors.
    feat_vectorizer = fe.TfidfVectorizer(input='filename', ngram_range=(1,3), stop_words='english', vocabulary=vocab)
    # Convert to vectors.
    train_X = feat_vectorizer.fit_transform(train_review_files)
    train_y = train_review_labels
    test_X = feat_vectorizer.fit_transform(test_review_files)
    test_y = test_review_labels
    return train_X, train_y, test_X, test_y


def get_train_review_files_with_label(size=None):
    return _get_review_files_with_label(os.path.join(__get_mydir(), '../dataset/aclImdb/train'), size)


def get_test_review_files_with_label(size=None):
    return _get_review_files_with_label(os.path.join(__get_mydir(), '../dataset/aclImdb/test'), size)


def _get_review_files_with_label(dir, size=None):
    pos_reviews_dir = os.path.join(dir, 'pos')
    neg_reviews_dir = os.path.join(dir, 'neg')
    # list positive and negative review files.
    pos_reviews_files = __prune_to_size(__list_of_files(pos_reviews_dir), size)
    neg_reviews_files = __prune_to_size(__list_of_files(neg_reviews_dir))
    pos_label = 1
    neg_label = -1
    pos_labels = [pos_label]*len(pos_reviews_files)
    neg_labels = [neg_label]*len(neg_reviews_files)
    # Concatenate pos and neg and return
    pos_reviews_files.extend(neg_reviews_files)
    pos_labels.extend(neg_labels)
    return pos_reviews_files, np.array(pos_labels)


def __prune_to_size(l, size=None):
    if size is None or size >= len(l):
        return l
    else:
        return l[0:size]


def __list_of_files(dir):
    x = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
    return x


def __get_mydir():
    return os.path.dirname(os.path.realpath(__file__))
