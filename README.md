# Ensemble Learning for Tweet Classification of Hate Speech and Offensive Language - Winter/Spring Project 2018

These programs are part of a project that will use an ensemble learning model to detect offensive language and hate speech in tweets. It is composed of:

  - A Voting classifier
  - An LSTM network
  - A Bayesian model
  - A Proximity model
 
Link to full project: https://github.com/quinnbp/WT2018 

------------------------------------------------------------------------------------------------------------------------------
This repository contains: 

## Voting classifier for hate-speech and offensive language detection in tweets:

  - Uses a Voting classifier that evaluates the outputs of:
    - An SGD Classifier with log loss
    - A LinearSVM Classifier with L1 feature selection and L2 classification
    - A Perceptron
  
  - Features:
    - TFIDF matrix
    - POS-Tags matrix
    - Sentiment analysis
    - Prescence-of-lexicon-terms score 
    - Word embeddings (random and GloVe)

TODO:
  - Try improving word embeddings using a neural network based on [2]
  
## Weighting system for ensemble learning

  - Has 3 different options for applying weighted voting:
    - Precision score of the classifiers' confusion matrices
    - CEN score 
    - Precision + CEN score
    - Equal voting

## Confusion matrix class

  - Creates a confusion matrix given the output predictions of a classifer and the set of true labels
  - Contains operations like getting precision score, storing it as a pdf, getting number of false positives, getting the CEN score of the matrix, etc.

----------------------------------------
*All written by Daniel Firebanks*

Inspired by the research of: 
  - [1]Davidson et al. (https://github.com/t-davidson/hate-speech-and-offensive-language) 
  - [2]Badjatiya et al. (https://github.com/pinkeshbadjatiya/twitter-hatespeech) 



