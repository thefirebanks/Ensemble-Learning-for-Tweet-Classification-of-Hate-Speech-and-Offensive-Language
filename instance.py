#!/usr/bin/env python

""" instance.py: Instance class for WT2018 project. Contains non-binary label and ordered
        list of words for each tweet. """

__author__ = "Quinn Barker-Plummer"
__email__  = "qbarkerp@oberlin.edu"
__status__ = "Development"

class Instance:
    def __init__(self):
        self.label = None
        self.fulltweet = "" # Original tweet
        self.wordlist = []  # Tokenized tweet
        self.clean_tweet = "" # Processed tweet

    def __str__(self):
        return str([self.label, self.fulltweet])

    def getCleanTweet(self):
        return self.clean_tweet

    def getFullTweet(self):
        return self.fulltweet

    def getWordList(self):
        return self.wordlist

    def getLabel(self):
        return self.label
