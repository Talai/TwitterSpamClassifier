#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################
#   Twitter Spam Classifier                                        #
#   train.py                                                       #               
#   Aline Castendiek, 20.3.16                                      #
#                                                                  #
#   This scipt performs tranining on a data set. It saves the      #   
#   resulting model via pickle.                                    #
####################################################################

# Example call: python3 train.py data/small_spam.json data/small_ham.json model.pickle

from __future__ import print_function, unicode_literals, division
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from PrepareData import Preparation
import random
from pprint import pprint
import pickle
import sys

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

####################################################################
#       Main                                                       #
####################################################################

"""
  Train a desired model.

  Command line arguments: 
  [0]: train.py
  [1]: labeled spam data in JSON format
  [2]: labeled ham data in JSON format
  [3]: name of file in which the trained model will be saved
"""
if len(sys.argv) == 4:        

  training_data = Preparation.read_in(sys.argv[1], sys.argv[2])

  random.shuffle(training_data)

  # Unzip labels and texts into separate lists:
  labels, feature_vectors = zip(*training_data)

  vectorizer = DictVectorizer()

  X = vectorizer.fit_transform(feature_vectors)
  y = labels

  # Train a classifier
  print("Starting the training...")

  #clf = classifiers_to_test = [
  #clf = BernoulliNB(),
  #clf = LogisticRegression(),
  #clf = Perceptron(n_iter=100),
  #clf = LinearSVC(),
  #clf = KNeighborsClassifier(warn_on_equidistant=False, weights="distance"),
  #clf = RandomForestClassifier()
  clf = LogisticRegression()
  clf.fit(X,  y)

  # Save trained model in separate file 
  print("Training done. Saving to file now...")
  pickle.dump((clf, vectorizer), open(sys.argv[3], "wb"))

  print("Done!")


####################################################################
#     Print Instructions                                           #
####################################################################

else:

  print("\nUSAGE:\n")
    
  print("python3 train.py spam_data.json ham_data.json model.pickle \n")
  print("spam_data.json: labeled spam data in JSON format.") 
  print("ham_data.json: labeled ham data in JSON format.")
  print("model.pickle: Desired file in which the trained model will be saved via pickle.\n\n")
