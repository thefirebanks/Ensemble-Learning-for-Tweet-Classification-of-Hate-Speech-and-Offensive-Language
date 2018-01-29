"""Code originally written by Davidson et al. at https://github.com/t-davidson/hate-speech-and-offensive-language
    Modified by: Daniel Firebanks
        - Added lexicon score based on ngram frequency
        - Added random and glove word embeddings
        - Added hard voting classifier (Logisitc Regression, SGDClassifier, Perceptron, XGBClassifier)
"""



import string
import re
import pandas as pd
import numpy as np
import pickle
import multiprocessing
import sys
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import gensim
from time import time
from xgboost import XGBClassifier
from string import punctuation
from collections import defaultdict
from gensim.models.word2vec import Word2Vec


#matplotlib.use('TkAgg')

lexicon = pd.read_csv('ngram_dict.csv')
ngrams_list = lexicon.ngrams
ngrams_probs = lexicon.probs
ngrams_dict = dict(zip(ngrams_list, ngrams_probs))
#print(ngrams_dict)


# Methods for processing data =================================================================================================

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


# Methods for analyzing other features =================================================================================

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))

def get_ngram_score(ngram, prob, ngrams_intweet):
    """Gets an ngram and checks if it is already contained in the tweet's ngrams_list
    If it isn't, it adds the ngram to the ngrams list
    Returns the length of the ngram and its probability"""

    score = 0
    length = 0
    var = False
    for el in ngrams_intweet:
        if ngram in el:
            var = True
            break
    if not var:
        ngrams_intweet.append([ngram])
        score = prob
        length = len(ngram)

    #return score #--> For just the average
    return length, score

def get_lexicon_score(tweet):
    """
    Given a tweet, it calculates the score with this formula

    ngram_score = amount of words in ngrams * average of ngrams probabilities

    """
    ngrams_intweet = []
    ngrams_count = 0
    total_length = 0
    total_prob = 0

    for ngram, prob in ngrams_dict.items():
        if ngram in tweet:
            #score = get_ngram_score(ngram, prob, ngrams_intweet)
            length, score = get_ngram_score(ngram, prob, ngrams_intweet)
            if score != 0:
                ngrams_count += 1
                total_length += length
                total_prob += score

    if ngrams_count > 0:
        prob_avg = total_prob / ngrams_count
    else:
        prob_avg = 0

    #return prob_avg #--> For just the average
    return prob_avg * total_length

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features.

    **Added - presence of word in lexicon
    """

    # Get sentiment analysis score
    sentiment_analyzer = VS()
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    # Get text only
    words = preprocess(tweet)

    # Get lexicon relevance score
    ngrams_relevance = get_lexicon_score(tweet)

    # Get other features from words
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet, ngrams_relevance]
    # features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

def make_word_embeddings(tweets):

    num_features = 300
    min_word_count = 3 #5
    num_workers = multiprocessing.cpu_count()
    context_size = 7 #5
    downsampling = 1e-5
    seed = 1

    w2v_model = gensim.models.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        min_count=min_word_count,
        size=num_features,
        window=context_size,
        sample=downsampling
    )

    print("Building vocab...")
    w2v_model.build_vocab(tweets)
    print("Training..")
    w2v_model.train(tweets, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
    print("Created word embeddings model!")
    w2v_model.save("random_model.txt")


def make_feature_vec(tweet, model, num_features, index2word_set):
    """Returns the word embeddings for one tweet"""

    # Initialize vector and embedding count
    feature_vec = np.zeros((num_features,), dtype="float32")
    num_words = 0

    # Prepare tweet
    #tweet = preprocess(tweet)
    #tweet = tokenize(tweet)

    # Get embedding
    for word in tweet:
        if word in index2word_set:
            num_words += 1
            feature_vec = np.add(feature_vec, model[word])

    feature_vec = np.divide(feature_vec, num_words)
    #print(feature_vec)
    #print(feature_vec.shape)
    return feature_vec


def get_avg_feature_vec(clean_tweets, model, num_features, index2word_set):
    """Gets the set of feature vectors for every tweet as a 2D array"""

    # Initialize count
    count = 0
    tweets_feature_vecs = np.zeros((len(clean_tweets), num_features), dtype="float32")
    #print(tweets_feature_vecs.shape)


    print("Getting embeddings...")
    for tweet in clean_tweets:
        # Print a status message every 1000th tweet
        if count % 1000. == 0.:
            print("Tweet %d of %d" % (count, len(tweets)))

        # Get the embeddings
        tweets_feature_vecs[count] = make_feature_vec(tweet, model, num_features, index2word_set)

        count += 1

    return tweets_feature_vecs


def benchmark(clf, name):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred))
    print()

    print("confusion matrix:")
    confusion_matrix_var = metrics.confusion_matrix(y_test, pred)

    matrix_proportions = np.zeros((3, 3))
    for i in range(0, 3):
        matrix_proportions[i, :] = confusion_matrix_var[i, :] / float(confusion_matrix_var[i, :].sum())

    names = ['Hate', 'Offensive', 'Neither']
    confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
    plt.figure(figsize=(5, 5))
    seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', cbar=False, square=True,
                    fmt='.2f')
    plt.ylabel(r'True categories', fontsize=14)
    plt.xlabel(r'Predicted categories', fontsize=14)
    plt.tick_params(labelsize=12)

    print(confusion_matrix_var)
    print(matrix_proportions)
    plt.savefig("Davidson_random_" + name + "_Classifier2.pdf")
    print()

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


if __name__ == '__main__':

    # Open data ========================================================================================================

    # df = pd.read_csv("../classifier/trump_tweets.csv")
    #df = pd.read_csv("All_Tweets_June2016_Dataset.csv")
    df = pd.read_csv("labeled_data.csv")
    tweets = df.tweet

    # Process data =====================================================================================================

    stopwords = nltk.corpus.stopwords.words("english")
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)

    stemmer = PorterStemmer()

    # Build TF-IDF =====================================================================================================

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords,
        use_idf=True,
        smooth_idf=False,
        binary=True,
        norm=None,
        decode_error='replace',
        max_features=10000,
        min_df=5,
        max_df=0.75
    )

    # Construct tfidf matrix and get relevant scores
    tfidf = vectorizer.fit_transform(tweets).toarray()
    vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    idf_vals = vectorizer.idf_
    idf_dict = {i: idf_vals[i] for i in vocab.values()}  # keys are indices; values are IDF scores

    # Build POS TF-IDF =================================================================================================

    # Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)

    # We can use the TFIDF vectorizer to get a token matrix for the POS tags
    pos_vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.75,
    )

    # Construct POS TF matrix and get vocab dict
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v: i for i, v in enumerate(pos_vectorizer.get_feature_names())}

    # Get word embeddings ==============================================================================================

    # TODO Also try other embeddings/random embeddings
    clean_tweets = []
    for tweet in tweets:
        clean_tweet = preprocess(tweet)
        clean_tweet = tokenize(clean_tweet)
        clean_tweets.append(clean_tweet)

    print("Example of clean tweet:", clean_tweets[0])
    #make_word_embeddings(clean_tweets)
    #glove_model_file = "~/Downloads/glove.twitter.27B/glove.twitter.27B.200d.txt"

    glove_model_file = "random_model_combined_tweets.txt"
    word2vec_model = gensim.models.Word2Vec.load(glove_model_file)
    print("Loading word2vec model...")

    # df1 = pd.read_csv("All_Tweets_June2016_Dataset.csv")
    # tweets1 = df1.tweet
    #
    # clean_tweets1 = []
    # for tweet1 in tweets1:
    #     clean_tweet1 = preprocess(tweet1)
    #     clean_tweet1 = tokenize(clean_tweet1)
    #     clean_tweets1.append(clean_tweet1)
    #
    # word2vec_model.train(clean_tweets1, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
    # word2vec_model.save("random_model_combined_tweets.txt")

    embedding_dim = 300
    word2vec_model = word2vec_model.wv
    # Index2word is a list that contains the names of the words in the model's vocabulary.
    index2word_set = set(word2vec_model.index2word)
    embeddings = get_avg_feature_vec(clean_tweets, word2vec_model, embedding_dim, index2word_set)

    print("Got embeddings!")

    # Get other features ===============================================================================================

    other_features_names = ["FKRA", "FRE", "num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total",
                            "num_terms", "num_words", "num_unique_words", "vader neg", "vader pos", "vader neu",
                            "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet", "ngrams_relevance"]

    feats = get_feature_array(tweets)
    print("Got other features!")

    # Now join them all up ======================================================a=======================================

    M = np.column_stack((tfidf, feats, embeddings)) #emb, embeddings_weights

    # Finally get a list of variable names
    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    pos_variables = [''] * len(pos_vocab)
    for k, v in pos_vocab.items():
        pos_variables[v] = k

    feature_names = variables + pos_variables + other_features_names

    # Running the model ================================================================================================

    # The best model was selected using a GridSearch with 5-fold CV.

    X = pd.DataFrame(M)

    X.fillna(X.mean(), inplace=True)

    y = df['class'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    # Testing classifiers ==============================================================================================
    results = []

    # Classifiers
    clf1 = LogisticRegression(class_weight='balanced', penalty="l2", C=0.01)
    clf2 = SGDClassifier(alpha=.0001, n_iter=50, penalty="l1")
    clf3 = LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr')
    #clf3 = CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr'), method='sigmoid', cv=3)
    #clf3 = SVC(kernel='linear', probability=True, class_weight='balanced', C=0.01)
    clf4 = Perceptron(n_iter=50)
    #clf5 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf5 = XGBClassifier(n_estimators=200, max_depth=4)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('sgd', clf2), ('lsvc', clf3), ('p', clf4), ('xgb', clf5)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf],
                          ['Logistic Regression', 'SGD Classifier', 'Linear SVM', 'Perceptron', 'Gradient Boosting Classifier', 'Ensemble']):
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5,random_state=42).split(X_train, y_train), scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    print('=' * 80)
    print("Perceptron")
    results.append(benchmark(clf4, "Perceptron"))

    print('=' * 80)
    print("Gradient Boosting")
    results.append(benchmark(clf2, "GBClassifier"))

    # for penalty in ["l2", "l1"]:
    #     print('=' * 80)
    #     print("%s penalty" % penalty.upper())
    #     # Train Liblinear model
    #     results.append(benchmark(clf3, "LinearSVM"))
    #
    #     # Train SGD model
    #     results.append(benchmark(clf5, "SGD"))
    #
    #     # Train Logistic regression model
    #     results.append(benchmark(clf1, "Logistic_Regression"))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet"), "SGD_Elastic"))

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                        tol=1e-3))),
        ('classification', LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr'))]), "LinearSVM_L1feature"))

    print('=' * 80)
    print("Logistic Regression with L1-based feature selection")
    results.append(benchmark(Pipeline(
             [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                       penalty="l1", C=0.01))),
             ('model', LogisticRegression(class_weight='balanced', penalty='l2', C=0.01))]), "Logistic_L1feature"))

    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    plt.savefig("Davidson_Classifiers_RandomEmbeddings_2.pdf")
    plt.show()




    # Grid search ======================================================================================================
    # pipe = Pipeline(
    #         [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
    #                                                   penalty="l1", C=0.01))),
    #         ('model', LogisticRegression(class_weight='balanced', penalty='l2', C=0.01))])

    # pipe = Pipeline(
    #         [('select', SelectFromModel(LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr'))),
    #         ('model', LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr'))])

    # Parameter search
    # solver_options = ['newton-cg', 'lbfgs', 'sag']
    # multi_class_options = ['ovr', 'multinomial']
    # class_weight_options = [None, 'balanced']
    #
    # param_grid = [{"model__solver": solver_options, "model__multi_class": multi_class_options, "model__class_weight": class_weight_options}]

    # param_grid = [{}]
    # grid_search = GridSearchCV(pipe,
    #                            param_grid,
    #                            cv=StratifiedKFold(n_splits=5,
    #                                               random_state=42).split(X_train, y_train),
    #                            verbose=2)
    #
    # model = grid_search.fit(X_train, y_train)

    # ========================================================================================================
    # print("Best parameters set found on development set:")
    # print()
    # print(grid_search.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = grid_search.cv_results_['mean_test_score']
    # stds = grid_search.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()

    #y_preds = model.predict(X_test)

    # Print and store classification report ============================================================================

    # report = classification_report(y_test, y_preds)
    # print(report)
    #
    # confusion_matrix = confusion_matrix(y_test, y_preds)
    # matrix_proportions = np.zeros((3, 3))
    # for i in range(0, 3):
    #     matrix_proportions[i, :] = confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
    # names = ['Hate', 'Offensive', 'Neither']
    # confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
    # plt.figure(figsize=(5, 5))
    # seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', cbar=False, square=True,
    #                 fmt='.2f')
    # plt.ylabel(r'True categories', fontsize=14)
    # plt.xlabel(r'Predicted categories', fontsize=14)
    # plt.tick_params(labelsize=12)
    #
    # # Save output
    # plt.savefig('Davidson_experiments.pdf')

    # True distribution
    #y.hist()
    #pd.Series(y_preds).hist()
