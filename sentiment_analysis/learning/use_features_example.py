from sentiment_analysis.feature_engineering import feature_eng_poc as feat_eng
import sklearn.feature_extraction.text as fe


def sample_method_for_reference():
    # k is number of features tha we want.
    vocab = feat_eng.select_features_vocabulary(k=10000)
    # Get name of training files with label of each
    train_review_files, train_review_labels = feat_eng.get_train_review_files_with_label()
    # Create vectorizer that will convert text to feature vectors.
    feat_vectorizer = fe.TfidfVectorizer(input='filename', ngram_range=(1,3), stop_words='english', vocabulary=vocab)
    # Convert to vectors.
    X = feat_vectorizer.fit_transform(train_review_files)
    y = train_review_labels
    # Tune and Train your algorithms after this step.
    # To tune use 5-fold method, there's a convenient way in scikit-learn to do 5-fold.
    # Evaluate your algorithms on test data.
    test_review_files, test_review_labels = feat_eng.get_test_review_files_with_label()

