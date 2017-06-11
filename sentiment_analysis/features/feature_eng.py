import sklearn.feature_extraction.text as fe
import sklearn.feature_selection as fs
import data_reader as dr


def select_features_vocabulary(k=10000): # k is number of features we want to select
    # Start with all unigrams, bigrams and trigrams in training data as vocabulary.
    # Get data source.
    review_files, review_labels = dr.get_train_review_files_with_label()
    # Create uni,bi and tri grams
    vectorizer = fe.TfidfVectorizer(input='filename', ngram_range=(1,3), stop_words='english')
    X = vectorizer.fit_transform(review_files)
    all_features = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
    if k is None:
        return all_features
    else:
        return _relevant_features_using_chi2(all_features, X, review_labels, k)


def _relevant_features_using_chi2(all_features, X, y, k):
    feature_selector = fs.SelectKBest(fs.chi2, k=k)
    feature_selector.fit_transform(X, y)
    relevant_features = feature_selector.get_support()
    return [f[0] for f in zip(all_features, relevant_features) if f[1]]
