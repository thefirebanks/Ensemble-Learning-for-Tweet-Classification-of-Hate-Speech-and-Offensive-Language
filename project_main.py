import nltk
from nltk.stem.porter import *
import re
import pickle
import pandas as pd
import random
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from instance import Instance
from weighting import voting
#from bayes import BayesModel
from proximity import ProximityModel
from attempt import LSTM
from voting_classifier import VotingModel
from confusion_matrix import ConfusionMatrix

# TODO Run tests on multiple voting systems at once
# TODO First run project_main.py, then tests.py

def main(tperc, seed, fpaths, weighting_type):
    """Parses files, trains the models, tests the models,
    creates the weights, makes predictions and evaluates results"""

    files = openFiles(fpaths)
    instances = parseFiles(files)
    train_set, test_set1, test_set2 = splitSets(tperc, seed, instances)

    # Initialize all models
    p = ProximityModel()
    v = VotingModel()
    # b = BayesModel()
    r = LSTM()

    print("Initialized all models!")

    # Train all models

    p.train(train_set)
    v.train(train_set)
    # b.train(train_set)
    # r.train(train_set)

    print("Trained all models!")

    # Run models and store first set of results

    p_pred = p.batchTest(test_set1)
    v_pred = v.batchTest(test_set1)
    # b_pred = b.batchTest(test_set1)
    r_pred = r.batchTest(test_set1)

    print("Predictions made for first test set!")

    # Store first set of predictions

    preds1 = [p_pred, v_pred, r_pred]#, b_pred, r_pred]
    test_set1_labels = [i.getLabel() for i in test_set1]
    store_preds(preds1, test_set1_labels, 1)

    print("Stored predictions for first test set!")

    # Get confusion matrices for first set of results

    p_cm = ConfusionMatrix(test_set1_labels, p_pred, "Proximity")
    v_cm = ConfusionMatrix(test_set1_labels, v_pred, "Voting")
    # b_cm = ConfusionMatrix(test_set1_labels, b_pred, "Bayes")
    r_cm = ConfusionMatrix(test_set1_labels, r_pred, "LSTM")

    confusionMatrices = [p_cm, v_cm, r_cm]
    # confusionMatices = [p_cm, v_cm, b_cm, r_cm]

    # Save individual confusion matrices to files

    for cm in confusionMatrices:
        cm.store_cm()

    print("Individual confusion matrices created and stored!")

    # Second set of predictions

    p_pred2 = p.batchTest(test_set2)
    v_pred2 = v.batchTest(test_set2)
    #b_pred2 = b.batchTest(test_set2)
    r_pred2 = r.batchTest(test_set2)

    print("Predictions made for second test set!")

    # Store second set of predictions

    preds2 = [p_pred2, v_pred2, r_pred2]  # , b_pred2, r_pred2]
    test_set2_labels = [i.getLabel() for i in test_set2]
    store_preds(preds2, test_set2_labels, 2)

    print("Stored predictions for second test set!")

    # Weight second set of results, using confusion matrices from first set

    weightingInput = [
        [confusionMatrices[0], p_pred2],
        [confusionMatrices[1], v_pred2]
        # [confusionMatrices[2] ,b_pred2],
        [confusionMatrices[2], r_pred2],
    ]

    # Get the weighting results

    guesses = voting(weightingInput, weighting_type)
    print("Voting done!")
    # print(guesses)

    # Create confusion matrix for final model and store it in a file

    final_cm = ConfusionMatrix(test_set2_labels, guesses, "Final_Model_" + weighting_type)
    final_cm.store_cm()
    print("Stored confusion matrix!")

    # Store second set of tweets and guesses

    test_set2_tweets = [t.getFullTweet() for t in test_set2]
    store_new_labels(test_set2_tweets, guesses, test_set2_labels)
    print("Stored new predictions!")


def alternative_main(tperc, seed, weighting_type, fpaths):
    """COPY OF MAIN FUNCTION - EXCEPT ONLY USED TO STORE PREDICTIONS FOR BOTH TEST SETS
        Parses files, trains the models, tests the models,
        creates the weights, makes predictions"""

    files = openFiles(fpaths)
    instances = parseFiles(files)
    train_set, test_set1, test_set2 = splitSets(tperc, seed, instances)

    # Initialize all models

    p = ProximityModel()
    v = VotingModel()
    # b = BayesModel()
    r = LSTM()

    print("Initialized all models!")

    # Train all models

    p.train(train_set)
    v.train(train_set)
    # b.train(train_set)
    # r.train(train_set)

    print("Trained all models!")

    # Run models and store first set of results

    p_pred = p.batchTest(test_set1)
    v_pred = v.batchTest(test_set1)
    # b_pred = b.batchTest(test_set1)
    r_pred = r.batchTest(test_set1)

    print("Predictions made for first test set!")

    # Store first set of predictions

    preds1 = [p_pred, v_pred, r_pred]#, b_pred, r_pred]
    test_set1_labels = [i.getLabel() for i in test_set1]
    store_preds(preds1, test_set1_labels, 1)

    print("Stored predictions for first test set!")

    # Run models and store second set of results

    p_pred2 = p.batchTest(test_set2)
    v_pred2 = v.batchTest(test_set2)
    # b_pred2 = b.batchTest(test_set2)
    r_pred2 = r.batchTest(test_set2)

    print("Predictions made for second test set!")

    # Store first set of predictions

    preds2 = [p_pred2, v_pred2, r_pred2]  # , b_pred2, r_pred2]
    test_set2_labels = [i.getLabel() for i in test_set2]
    store_preds(preds2, test_set2_labels, 2)

    print("Stored predictions for second test set!")


def run_multiple_voting():
    """Run tests on multiple weighting systems given stored predictions of the classifiers in main()
        To be used in conjunction with alternative_main to determine which weighting method performs better
    """

    # Load predictions from all classifiers and actual labels for test_set_1
    preds1, actual1 = load_preds(1)

    # Load predictions from all classifiers and actual labels for test_set_2
    preds2, actual2 = load_preds(2)

    # Create confusion matrices for each classifier
    p_cm = ConfusionMatrix(actual1, preds1[0], "Proximity")
    v_cm = ConfusionMatrix(actual1, preds1[1], "Voting")
    # b_cm = ConfusionMatrix(actual1, preds1[2], "Bayes")
    # r_cm = ConfusionMatrix(actual1, preds1[2], "LSTM")

    confusionMatrices = [p_cm, v_cm]
    # confusionMatices = [p_cm, v_cm, b_cm, r_cm]

    # Save individual confusion matrices to files
    for cm in confusionMatrices:
        cm.store_cm()

    print("Individual confusion matrices created and stored!")

    # Weight second set of results, using confusion matrices from first set
    weightingInput = [
        [confusionMatrices[0], preds2[0]],
        [confusionMatrices[1], preds2[1]]
        # [confusionMatrices[2] ,b.batchTest(test_set2)],
        # [confusionMatrices[3], r.batchTest(test_set2)],
    ]

    # Get the weighted voting results
    votes_p = voting(weightingInput, "Precision")
    votes_CEN_p = voting(weightingInput, "CEN_Precision")
    votes_CEN = voting(weightingInput, "CEN")
    votes_eq = voting(weightingInput, "Equal_Vote")

    # Check metrics
    print(classification_report(actual2, votes_p))
    print(classification_report(actual2, votes_CEN_p))
    print(classification_report(actual2, votes_CEN))
    print(classification_report(actual2, votes_eq))

    # Create final confusion matrices depending on votes
    p_cm = ConfusionMatrix(actual2, votes_p, "Precision")
    p_CEN_cm = ConfusionMatrix(actual2, votes_CEN_p, "CEN_Precision")
    CEN_cm = ConfusionMatrix(actual2, votes_CEN, "CEN")
    eq_cm = ConfusionMatrix(actual2, votes_eq, "Equal")

    # Store confusion matrices
    p_cm.store_cm()
    p_CEN_cm.store_cm()
    CEN_cm.store_cm()
    eq_cm.store_cm()

    return votes_p, votes_CEN_p, votes_CEN, votes_eq

def store_new_labels(t2tweets, guesses, labels):
    """Creates a csv document that stores the tweets tested and their predicted labels"""

    with open("FinalModel_Predictions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["tweet", "class", "label"])
        for tweet, guess, label in zip(t2tweets, guesses, labels):
            writer.writerow([tweet, guess, label])

def store_preds(preds, actual, num_test):
    """Stores the list of predictions into a text file to be loaded later
        num_test: number of test_set
    """

    num_test = str(num_test)

    # Proximity, Voting, Bayes, LSTM
    f1 = open("proximity_preds_" + num_test + ".txt", "w+b")
    f2 = open("voting_preds_" + num_test + ".txt", "w+b")
    #f3 = open("bayes_preds_" + num_test + ".txt", "w+b")
    f4 = open("lstm_preds_" + num_test + ".txt", "w+b")
    f5 = open("actual_labels_" + num_test + ".txt", "w+b")

    files = [f1, f2, f4, f5]#, f3, f4]

    for i in range(len(preds)):
        pickle.dump(preds[i], files[i])

    pickle.dump(actual, f5)

def load_preds(num_test):
    """Loads the predictions file and returns a list of prediction lists
        num_test: number of test_set
    """

    num_test = str(num_test)

    # Proximity, Voting, Bayes, LSTM
    f1 = open("proximity_preds_" + num_test + ".txt", "rb")
    f2 = open("voting_preds_" + num_test + ".txt", "rb")
    #f3 = open("bayes_preds_" + num_test + ".txt", "rb")
    f4 = open("lstm_preds_" + num_test + ".txt", "rb")
    f5 = open("actual_labels_" + num_test + ".txt", "rb")

    files = [f1, f2, f4, f5]#, f3, f4]
    preds = []

    for i in range(len(files)-1):
        l = pickle.load(files[i])
        preds.append(l)

    actual = pickle.load(f5)

    return preds, actual

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
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    stemmer = PorterStemmer()
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    # tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


def main_parser(f):
    """"
        @input file
        @output list of instance objects

        Reads files in the format as labeled_data.csv as a pandas dataframe
        This means that it contains a top row with the words tweets | class,
        so they can be referenced easily.

        Creates instance objects with the full text, the tokenized text and the label
    """

    # Read inputs using pandas
    df = pd.read_csv(f)
    raw_tweets = df.tweet
    labels = df['class'].astype(int)
    instances = []

    # Process tweets and create instances
    for tweet, label in zip(raw_tweets, labels):

        # Raw tweet and label
        i = Instance()
        i.label = label
        i.fulltweet = tweet

        # Get just text
        clean_tweet = preprocess(tweet)
        i.clean_tweet = clean_tweet

        # Tokenize tweet
        tokenized_tweet = basic_tokenize(clean_tweet)
        # stemmed_tweet = tokenize(clean_tweet)
        i.wordlist = tokenized_tweet

        instances.append(i)

    return instances


def splitSets(tperc, seed, instances):
    random.seed(seed)
    random.shuffle(instances)
    split = int(tperc * len(instances))
    second_split = int(split + float(len(instances) - split) / 2)

    return instances[:split], instances[split:second_split], instances[second_split:]


def parseFiles(files):
    instances = []
    for f in files:
        instances += main_parser(f)
    return instances


def openFiles(filepaths):  # takes in file paths and attempts to open, fails if zero valid files
    files = []
    for fp in filepaths:
        try:
            f = open(fp, 'r')
            files.append(f)
        except FileNotFoundError:
            print("Readin: NonFatal: file " + str(fp) + " not found.")

    if len(files) == 0:
        raise FileNotFoundError("Readin: Fatal: No Files Found")
    else:
        return files


if __name__ == "__main__":
    if len(sys.argv) < 5:
        # not enough args to make tperc or seed
        raise IndexError("Readin: Fatal: Not enough given arguments.")
    else:
        tperc = float(sys.argv[1])
        seed = int(sys.argv[2])
        weighting_type = sys.argv[3]
        fpaths = sys.argv[4:]

        # Check that weighting type is correct
        if weighting_type not in ["Precision", "CEN_Precision", "CEN", "Equal_Vote"]:
            print("Please give a correct weighting option. Choose from the following: "
                  "\nPrecision, CEN_Precision, CEN, Equal_Vote")
        else:
            #main(tperc, seed, fpaths, weighting_type)
            alternative_main(tperc, seed, weighting_type, fpaths)
            #run_multiple_voting()
