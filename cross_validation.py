#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################
#   Twitter Spam Classifier                                        #
#   cross_validation.py                                            #               
#   Aline Castendiek, 20.3.16                                      #
#                                                                  #
#   Prints the evaluation of all specified models to the console   #
#                                                                  #          
####################################################################

# Example call: python3 cross_validation.py data/small_spam.json data/small_ham.json

from __future__ import print_function, unicode_literals, division
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from PrepareData import Preparation
import random
from pprint import pprint
import pickle
import sys

def cross_validation():
  from sklearn import cross_validation
  from sklearn.naive_bayes import BernoulliNB
  from sklearn.linear_model import LogisticRegression
  from sklearn.linear_model import Perceptron
  from sklearn.svm import LinearSVC
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.ensemble import RandomForestClassifier

  # Create list of all classifiers
  classifiers_to_test = [
    BernoulliNB(),
    LogisticRegression(),
    Perceptron(n_iter=100),
    LinearSVC(),
    KNeighborsClassifier(warn_on_equidistant=False, weights="distance"),
    RandomForestClassifier()
  ]

  for model in classifiers_to_test:
    # Splits the data, fits the model and computes the scores 10 consecutive times (cv = 10; with different splits each time)
    # The tenth subset is used as a training set. Uses F1 Score as metric.
    scores = cross_validation.cross_val_score(model, X, y, cv=10, scoring="f1")
    print("\n", model)
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

"""
  Command line arguments: 
  [0]: cross_validation.py
  [1]: labeled spam data in JSON format
  [2]: labeled ham data in JSON format
"""
if len(sys.argv) == 3:        

  training_data = Preparation.read_in(sys.argv[1], sys.argv[2])

  random.shuffle(training_data)

  # Unzip labels and texts into separate lists:
  labels, feature_vectors = zip(*training_data)

  vectorizer = DictVectorizer()

  X = vectorizer.fit_transform(feature_vectors)
  y = labels

  # Train a classifier
  print("Starting the cross-validation...")
  cross_validation()

  print("Done!")


####################################################################
#     Print Instructions                                           #
####################################################################

else:

  print("\nUSAGE:\n")
    
  print("python3 cross_validation.py spam_data.json ham_data.json \n")
  print("spam_data.json: labeled spam data in JSON format.") 
  print("ham_data.json: labeled ham data in JSON format.\n\n")

