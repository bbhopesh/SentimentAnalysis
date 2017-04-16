import sklearn.feature_extraction.text as fe
import sklearn.feature_selection as fs
import os


def select_features_vocabulary(k=10000): # k is number of features we want to select
    # Start with all unigrams, bigrams and trigrams in training data as vocabulary.
    # Get data source.
    review_files, review_labels = get_train_review_files_with_label()
    # Create uni,bi and tri grams
    vectorizer = fe.TfidfVectorizer(input='filename', ngram_range=(1,3), stop_words='english')
    X = vectorizer.fit_transform(review_files)
    all_features = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
    return _relevant_features_using_chi2(all_features, X, review_labels, k)


def _relevant_features_using_chi2(all_features, X, y, k):
    feature_selector = fs.SelectKBest(fs.chi2, k=k)
    feature_selector.fit_transform(X, y)
    relevant_features = feature_selector.get_support()
    return [f[0] for f in zip(all_features, relevant_features) if f[1]]


def get_train_review_files_with_label():
    review_files, review_labels = _get_review_files_with_label(os.path.join(__get_mydir(), '../dataset/aclImdb/train'))
    return review_files, review_labels


def get_test_review_files_with_label():
    return _get_review_files_with_label(os.path.join(__get_mydir(), '../dataset/aclImdb/test'))


def _get_review_files_with_label(dir):
    pos_reviews_dir = os.path.join(dir, 'pos')
    neg_reviews_dir = os.path.join(dir, 'neg')
    # list positive and negative review files.
    pos_reviews_files = __list_of_files(pos_reviews_dir)
    neg_reviews_files = __list_of_files(neg_reviews_dir)
    pos_label = 1
    neg_label = -1
    pos_labels = [pos_label]*len(pos_reviews_files)
    neg_labels = [neg_label]*len(neg_reviews_files)
    # Concatenate pos and neg and return
    pos_reviews_files.extend(neg_reviews_files)
    pos_labels.extend(neg_labels)
    return pos_reviews_files, pos_labels


def __list_of_files(dir):
    x = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
    return x


def __get_mydir():
    return os.path.dirname(os.path.realpath(__file__))
