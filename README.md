# Offensive-language-and-hate-speech-tweet-classification-

Voting classifier for hate-speech and offensive language detection in tweets:

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
  - Try XGB Classifier on a better computer and evaluate its performance
  - Join it with the rest of the models that form part of the VotingClassifier. 

Inspired by the research of Davidson et al. (https://github.com/t-davidson/hate-speech-and-offensive-language) and Badjatiya et al. (https://github.com/pinkeshbadjatiya/twitter-hatespeech) 

------------------------------------------------------------------------------------------------------------------------------

This program is part of a project that will use a councilor model to detect offensive language and hate speech in tweets. It includes:

  - A Voting classifier
  - An LSTM network
  - A Bayesian model
  - A Proximity model
 
 All of the outputs will then be averaged and each classifier's output will be assigned a weight depending on the class in which it has gotten the most accuracy. Finally, the weighted average will be calculated as the model's output. 
 
Link to full project: https://github.com/quinnbp/WT2018 
 
Written by Daniel Firebanks, Sage Vouse and Quinn Barker-Plummer
