"""
Code originally written by Davidson et al. at https://github.com/t-davidson/hate-speech-and-offensive-language
    Taken from Davidson:
        - Preprocessing and tokenizing methods
        - TFIDF Vectorizer, POS Vectorizer, Other features array (Syllables, Character/word count, sentiment analysis)
    Modified by: Daniel Firebanks
        - Added lexicon score based on ngram frequency as a feature
        - Added random and glove word embeddings as features
        - Added hard voting classifier (SGD Classifier, LinearSVM, Perceptron) instead of Logistic Regression Classifier
"""

import string
import re
import pandas as pd
import numpy as np
import pickle
import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import gensim
import nltk
from sklearn.externals import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
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
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from time import time
from string import punctuation
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
from xgboost import XGBClassifier

# Variables for calculating lexicon score ==============================================================================

lexicon = pd.read_csv('ngram_dict.csv')
ngrams_list = lexicon.ngrams
ngrams_probs = lexicon.probs
ngrams_dict = dict(zip(ngrams_list, ngrams_probs))
#print(ngrams_dict)


# Variables for procesing "other features" data ========================================================================

stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()
sentiment_analyzer = VS()

# Methods for processing data ==========================================================================================

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

# Methods for analyzing features =======================================================================================

def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags

def get_clean_tweets(tweets):
    """Applies preprocessing and tokenization to list of tweets"""

    clean_tweets = []

    for tweet in tweets:
        clean_tweet = preprocess(tweet)
        clean_tweet = tokenize(clean_tweet)
        clean_tweets.append(clean_tweet)

    #print("Example of clean tweet:", clean_tweets[0])
    return clean_tweets

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

    **Added - lexicon score
    """

    # Get sentiment analysis score
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
    """Make random word embeddings using 30-40k tweets as reference from:
        - Davidson et al. Dataset of hatespeech, offensive language and none tweets
        - Waseem et al. Dataset of racist, sexist and neither tweets (June 2016)"""

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

    # Add the tweets from racist/sexist/none dataset for more reference
    df1 = pd.read_csv("All_Tweets_June2016_Dataset.csv")
    tweets1 = df1.tweet
    clean_tweets1 = get_clean_tweets(tweets1)
    tweets.extend(clean_tweets1)

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
    # Get embedding
    for word in tweet:
        if word in index2word_set:
            num_words += 1
            feature_vec = np.add(feature_vec, model[word])

    feature_vec = np.divide(feature_vec, num_words)

    #print(feature_vec)
    #print(feature_vec.shape)
    return feature_vec


def get_word_embeddings(clean_tweets, model, num_features, index2word_set):
    """Gets the set of feature vectors for every tweet as a 2D array"""

    # Initialize count
    count = 0
    tweets_feature_vecs = np.zeros((len(clean_tweets), num_features), dtype="float32")
    #print(tweets_feature_vecs.shape)

    print("Getting embeddings...")
    for tweet in clean_tweets:
        # Print a status message every 1000th tweet
        if count % 1000. == 0.:
            print("Tweet %d of %d" % (count, len(clean_tweets)))

        # Get the embeddings
        tweets_feature_vecs[count] = make_feature_vec(tweet, model, num_features, index2word_set)

        count += 1

    return tweets_feature_vecs


def run_classifier(clf, name, X_train, y_train, X_test, y_test):
    """Trains and tests a determined classifier.
    Outputs a confusion matrix, classification report and accuracy score"""

    # Training, testing and saving the model ===========================================================================
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

    joblib.dump(clf, "final_model_full_glove.pkl")
    print("Model saved!")

    # Metrics ==========================================================================================================
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred))
    print()

    # Display and save confusion matrix
    print("confusion matrix:")
    confusion_matrix_var = confusion_matrix(y_test, pred)

    # Turn the matrix quantities into decimals
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

    #print(confusion_matrix_var)
    print(matrix_proportions)
    plt.savefig(name + "_Full_Glove.pdf")
    print()

    # Save and return classifier description
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


if __name__ == '__main__':

    # Open data ========================================================================================================

    # df = pd.read_csv("../classifier/trump_tweets.csv")
    #df = pd.read_csv("All_Tweets_June2016_Dataset.csv")
    df = pd.read_csv("labeled_data.csv")
    tweets = df.tweet

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
    idf_vals = vectorizer.idf_

    print("Got TFIDF!")

    #np.save("TFIDF.npy", tfidf)
    #np.save("idf.npy", vocab)
    #print("TFIDF Saved")
    #tfidf = np.load("TFIDF.npy")
    #vocab = np.load("idf.npy")

    # Build POS TF-IDF =================================================================================================

    # Get POS tags for tweets and save as a string
    tweet_tags = get_pos_tags(tweets)

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
    print("Got POS Tags!")

    #np.save("POS_Tags.npy", pos)
    #print("POS Tags Saved")
    #pos = np.load("POS_Tags.npy")

    # Get word embeddings ==============================================================================================

    clean_tweets = get_clean_tweets(tweets)

    # Uncomment if want to create random word embeddings
    # make_word_embeddings(clean_tweets)
    # random_model_file = "random_model_combined_tweets.txt"
    # word2vec_model = gensim.models.Word2Vec.load(random_model_file)

    # Load model file
    glove_model_file = "glove.twitter.27B.200d.txt"
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(glove_model_file)

    print("Loading word2vec model...")

    embedding_dim = 200
    word2vec_model = word2vec_model.wv

    # Index2word is a list that contains the names of the words in the model's vocabulary.
    index2word_set = set(word2vec_model.index2word) # Convert to set for speed
    embeddings = get_word_embeddings(clean_tweets, word2vec_model, embedding_dim, index2word_set)
    print("Got embeddings!")

    #np.save("Word_embeddings.npy", embeddings)
    #print("Embeddings Saved")
    #embeddings = np.load("Glove_embeddings.npy")

    # Get other features ===============================================================================================

    feats = get_feature_array(tweets)
    print("Got other features!")

    #np.save("Other_features.npy", feats)
    #print("Other features Saved")
    #feats = np.load("Other_features.npy")

    # Now join them all up ======================================================a======================================

    M = np.column_stack((tfidf, pos, feats, embeddings))

    # Running the model ================================================================================================

    # The best model was selected using a GridSearch with 5-fold CV.

    X = pd.DataFrame(M)

    X.fillna(X.mean(), inplace=True)

    y = df['class'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    print("Data split!")

    # Testing classifiers ==============================================================================================

    results = []
    print('=' * 80)
    print("Voting classifier")

    # Main Classifiers (SGD_LOG, SVM, P)

    clf1 = SGDClassifier(class_weight='balanced', penalty='l2', alpha=0.0001, max_iter=50, loss='log')
    clf1_pipe = Pipeline([('select', SelectFromModel(SGDClassifier(class_weight='balanced', penalty='l1', alpha=0.0001, max_iter=50))),
                          ('model', clf1)])

    clf2 = LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr')
    clf2_pipe = Pipeline(
        [('select', SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))),
         ('model', clf2)])

    clf3 = Perceptron(class_weight='balanced', max_iter=50, penalty='l2', tol=1e-3)
    clf3_pipe = Pipeline([('select', SelectFromModel(Perceptron(class_weight='balanced', max_iter=50, penalty='l1', tol=1e-3))),
                          ('model', clf3)])

    eclf = VotingClassifier(estimators=[("sgd_log", clf1_pipe), ("svm", clf2_pipe), ("p", clf3_pipe)], voting='hard')

    # Train, test and save confusion matrix
    run_classifier(eclf, "SGD(Log)_SVM_P", X_train, y_train, X_test, y_test)

    # Extra classifiers (LR, XGB, GBC) *****

    # clf4 = LogisticRegression(class_weight='balanced', penalty='l2', C=0.01)
    # clf4_pipe = Pipeline([('select', SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))),
    #                       ('model', clf4)])

    # clf5 = XGBClassifier(n_estimators=250, max_depth=5)
    # clf5_pipe = Pipeline([('select', SelectFromModel(clf5)),
    #                       ('model', clf5)])

    # clf6 = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=4, random_state=0)
    # clf6_pipe = Pipeline([('select, SelectFromModel(clf6)),
    #                        ('model', clf6)])


    # Grid Search

    # clf_grid = GridSearchCV(clf_pipe, param_grid=[{}],
    #                          cv=StratifiedKFold(n_splits=5, random_state=42).split(X_train, y_train), verbose=1)
    # final_model = clf_grid.fit(X_train, y_train)
    # print("Best model for X Classifier is", final_model.best_estimator_)
    # print("Best parameters for X Classifier are", final_model.best_params_)
    # print("Best scores for X Classifier are", final_model.best_score_)

