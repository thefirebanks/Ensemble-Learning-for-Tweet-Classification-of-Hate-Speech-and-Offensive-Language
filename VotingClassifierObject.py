"""
Code originally written by Davidson et al. at https://github.com/t-davidson/hate-speech-and-offensive-language
    Taken from Davidson:
        - Preprocessing and tokenizing methods
        - TFIDF Vectorizer, POS Vectorizer, Other features array (Syllables, Character/word count, sentiment analysis)
    Modified by: Daniel Firebanks
        - Added lexicon score based on ngram frequency as a feature
        - Added random and glove word embeddings as features
        - Added hard voting classifier (SGD Classifier, LinearSVM, Perceptron) instead of Logistic Regression Classifier

========================================================================================================================
This file contains code to

    (a) Load the pre-trained classifier and
    associated files.

    (b) Transform new input data into the
    correct format for the classifier.

    (c) Run the classifier on the transformed
    data and return results.
"""

import pickle
import numpy as np
import gensim
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem.porter import *
import csv
import string
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *

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
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

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
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

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

    syllables = 1

    try:
        syllables = textstat.syllable_count(words) #count syllables in words
    except Exception:
        pass

    # Get other features from words
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

    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1], ngrams_relevance]
    # features = pandas.DataFrame(features)
    return features

def get_oth_features(tweets):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


def get_clean_tweets(tweets):
    """Applies preprocessing and tokenization to list of tweets"""

    clean_tweets = []

    for tweet in tweets:
        clean_tweet = preprocess(tweet)
        clean_tweet = tokenize(clean_tweet)
        clean_tweets.append(clean_tweet)

    return clean_tweets

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


def get_embeddings(clean_tweets, model, num_features, index2word_set):
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

class VotingModel:

    def __init__(self):

        # Load model
        self.model = joblib.load('final_model.pkl')
        self.tf_vectorizer= joblib.load('final_tfidf.pkl')
        self.idf_vector = joblib.load('final_idf.pkl')
        self.pos_vectorizer = joblib.load('final_pos.pkl')

        # Load embeddings
        glove_model_file = "glove.twitter.27B.200d.txt"

        # Uncomment this if using random embeddings
        # random_model_file = "random_model_combined_tweets.txt"
        # word2vec_model = gensim.models.Word2Vec.load(glove_model_file)

        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(glove_model_file) #Comment this out if using random embeddings
        self.emb = word2vec_model.wv

        # Variable to store confusion matrix
        self.confusion_matrix = None

        #TODO Add conditional for random embedings in the future (model and emb_dimension)

        # Variables for data
        self.X = None
        self.y = None
        self.data = None


    def read_input(self, dataset):

        # Load data
        print("Loading data to classify...")
        tweets = [t.getFullTweet() for t in dataset]
        #labels = [l.getLabel() for l in dataset]
        print(len(tweets), "tweets detected")

        # Format data
        tf_array = self.tf_vectorizer.fit_transform(tweets).toarray()
        tfidf_array = tf_array * self.idf_vector
        print("Built TF-IDF array")

        pos_tags = get_pos_tags(tweets)
        pos_array = self.pos_vectorizer.fit_transform(pos_tags).toarray()
        print("Built POS array")

        oth_array = get_oth_features(tweets)
        print("Built Sentiment/Lexicon score array")

        clean_tweets = get_clean_tweets(tweets)
        embedding_dim = 200

        # Index2word is a list that contains the names of the words in the model's vocabulary.
        index2word_set = set(self.emb.index2word)  # Convert to set for speed

        emb_array = get_embeddings(clean_tweets, self.emb, embedding_dim, index2word_set)
        print("Built Word embeddings array")

        # Combine everything
        X = np.column_stack((tfidf_array, pos_array, oth_array, emb_array))

        return X

    def train(self, dataset):

        self.data = dataset
        self.X = self.read_input(dataset)
        self.y = [t.getLabel() for t in dataset]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42, test_size=0.1)

        clf1 = SGDClassifier(class_weight='balanced', penalty='l2', alpha=0.0001, max_iter=50, loss='log')
        clf1_pipe = Pipeline([('select', SelectFromModel(
            SGDClassifier(class_weight='balanced', penalty='l1', alpha=0.0001, max_iter=50))),
                              ('model', clf1)])

        clf2 = LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr')
        clf2_pipe = Pipeline(
            [('select', SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l1", C=0.01))),
             ('model', clf2)])

        clf3 = Perceptron(class_weight='balanced', max_iter=50, penalty='l2', tol=1e-3)
        clf3_pipe = Pipeline(
            [('select', SelectFromModel(Perceptron(class_weight='balanced', max_iter=50, penalty='l1', tol=1e-3))),
             ('model', clf3)])

        eclf = VotingClassifier(estimators=[("sgd_log", clf1_pipe), ("svm", clf2_pipe), ("p", clf3_pipe)],
                                voting='hard')

        # Train, test and save model
        eclf.fit(X_train, y_train)

        # scores = cross_val_score(eclf, X_train, y_train,
        #                          cv=StratifiedKFold(n_splits=5, random_state=42).split(X_train, y_train),
        #                          scoring='accuracy', n_jobs=-1, verbose=1)
        #
        # print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), "Voting Classifier"))

        pred = eclf.predict(X_test)

        print("classification report:")
        print(classification_report(y_test, pred))

        #Save and store model
        self.model = eclf
        joblib.dump(eclf, "newly_trained_model.pkl")
        print("Model saved!")


    def predict(self, dataset):

        self.data = dataset
        self.X = self.read_input(dataset)
        self.y = self.model.predict(self.X)

        return self.y

    def getConfusionMatrix(self):

        y_labels = [t.getLabel() for t in self.data]

        self.confusion_matrix = confusion_matrix(self.y, y_labels)

        conf_dict = {}

        for i in range(len(self.confusion_matrix)):
            #total_row = sum(conf[i])
            for j in range(len(self.confusion_matrix[i])):
                key = (i, j)
                value = self.confusion_matrix[i][j]
                conf_dict[key] = value
                #print("Pairs are", key, value)

        return conf_dict

if __name__ == '__main__':

    print("Import and initialize VotingModel object to begin.")
    # global exception_count
    # exception_count = 0
    #
    # print("Loading data to classify...")
    #
    # #Tweets obtained here: https://github.com/sashaperigo/Trump-Tweets
    # df = pd.read_csv('trump_tweets.csv')
    # tweets = df.Text
    # tweets = [x for x in tweets if type(x) == str]
    # print(len(tweets), "tweets detected")
    #
    # print("Loading trained classifier... ")
    # model = joblib.load('final_model.pkl')
    #
    # print("Loading other information...")
    # tf_vectorizer = joblib.load('final_tfidf.pkl')
    # idf_vector = joblib.load('final_idf.pkl')
    # pos_vectorizer = joblib.load('final_pos.pkl')
    # #Load ngram dict
    # #Load pos dictionary
    # #Load function to transform data
    #
    # print("Transforming inputs...")
    # X = transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer)
    #
    # print("Running classification model...")
    # y = predictions(X, model)
    #
    # print("Number of exceptions is " + str(exception_count))
    #
    # with open("labeled_trump_tweets.csv", "w") as outfile:
    #     writer = csv.writer(outfile)
    #
    #     print("Saving predicted values: ")
    #
    #     for i,t in enumerate(tweets):
    #         row = [t, class_to_name(y[i])]
    #         writer.writerow(row)
    #         #print t
    #         #print class_to_name(y[i])
