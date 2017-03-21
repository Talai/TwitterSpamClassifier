#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################
#   Twitter Spam Classifier                                        #
#   Preparation.py                                                 #               
#   Aline Castendiek, 20.3.16                                      #
#                                                                  #
#   This file contains some functions that will be used            #
#   by the training and testing scripts.                           #
####################################################################

import json
import re
from pprint import pprint

# Extract a hashtag and ignore trailing symbols like "," or ":"
CLEAN_HASHTAG = re.compile(r'^#(\w|_|-|\d|ü|Ü|ä|Ä|ö|Ö|ß)+')

"""
  Create a dense vector (dict) that represents the features of a tweet and their corresponding weights/counts.
  The weights can be tuned by hand to get the best results. 
"""
def create_weighted_vector(tokens, hashtags, at_users, userId, HASHTAG_WEIGHT=10, USERID_WEIGHT=100, ATUSER_WEIGHT=1):
  # There is a possibility that MongoDB has returned userId as a dict of the form {'$numberLong' : 1234}. Catch it.
  if type(userId) is dict:
    userId = userId['$numberLong']

  # Count the occurrences of the tokens:
  vector = {token : tokens.count(token) for token in tokens}

  # For each hashtag set the defined weight. After that, add the dictionary to the existing one:
  vector.update({hashtag : HASHTAG_WEIGHT for hashtag in hashtags})

  # Use the number of addressed users as features:
  vector["**AT_USER_LENGTH**"] = ATUSER_WEIGHT * len(at_users)

  vector["**USERID:" + str(userId) + "**"] = USERID_WEIGHT
  return vector


"""
  Reads in the spam and ham files and returns a list of feature tuples. 
  Each tuple corresponds to one tweet and is a pair consisting of the label ('spam' or 'ham') and a
  dictionary that represents all the features of this tweet.
  e.g.:  ('ham', {'#hach': 10, '#tierkinder': 10, '**AT_USER_LENGTH**': 0, '**USERID:18830429**': 100} )
"""
def read_in(path_to_spam, path_to_ham):
  documents = []

  # Read in spam candidates:
  with open(path_to_spam) as json_file:
    for line in json_file:
      # Following metrics require binary 0-1 labels: 
      # If used with these metrics,....
      #documents.append(("spam", json.loads(line)))
      documents.append((1, json.loads(line)))

  # Read in ham candidates:    
  with open(path_to_ham) as json_file:
    for line in json_file:
      documents.append((0, json.loads(line)))
  
  # Now we want to create a list of tuples from the data:
  data_list = []  

  for category, document in documents:
    # First collect hashtags. If there are no hashtags, we will append an empty list.
    # Also, clean some hashtags that are badly tokenized.
    hashtags = []
    for tag in document.get("hashtags",[]):
      clean = CLEAN_HASHTAG.match(tag)
      # If there was a match, add it to the list of hashtags:
      if clean: 
        hashtags.append(clean.group(0))

    # Create a tuple of the class of this tweet and its feature vector:
    # The weights of the specific features can be tuned here.
    data = (category, 
      create_weighted_vector(
        document["tokensClean"], # tokensLow XXX
        hashtags, 
        document.get("atUsers",[]), 
        document["userId"],
        HASHTAG_WEIGHT=10,
        USERID_WEIGHT=100,
        ATUSER_WEIGHT=10
        )
      )
    data_list.append(data)

  return data_list
  