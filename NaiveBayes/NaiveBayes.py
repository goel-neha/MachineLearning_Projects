import pandas as pd
import re
import nltk
import numpy as np
nltk.download('stopwords')


import csv
with open("train.csv", 'r') as file:
    reviews = list(csv.reader(file))
# reviews = pd.read_csv("train.csv")
# print(reviews[1])
# print(len(reviews))

from bs4 import BeautifulSoup

from nltk.corpus import stopwords # Import the stop word list

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


# Computing the prior (H=positive reviews) according to the Naive Bayes' equation

def get_H_count(score):
    # Compute the count of each classification occurring in the data
    return len([r for r in reviews if r[1] == str(score)])

# We'll use these counts for smoothing when computing the prediction
positive_review_count = get_H_count(1)
negative_review_count = get_H_count(0)

# These are the prior probabilities (we saw them in the formula as P(H))
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)
print("P(H) or the prior is:", prob_positive)

# Python class that lets us count how many times items occur in a list
from collections import Counter
import re

def get_text(reviews, score):
    # Join together the text in the reviews for a particular tone
    # Lowercase the text so that the algorithm doesn't see "Not" and "not" as different words, for example
    return " ".join([r[0].lower() for r in reviews if r[1] == str(score)])

def count_text(text):
    # Split text into words based on whitespace -- simple but effective
    words = re.split("\s+", text)
    # Count up the occurrence of each word
    return Counter(words)

negative_text = get_text(reviews, 0)
positive_text = get_text(reviews, 1)

# Generate word counts(WC) dictionary for negative tone
negative_WC_dict = count_text(negative_text)

# Generate word counts(WC) dictionary for positive tone
positive_WC_dict = count_text(positive_text)
# print (positive_WC_dict)
# H = positive review or negative review
def make_class_prediction(text, H_WC_dict, H_prob, H_count):
    prediction = 1
    text_WC_dict = count_text(text)

    for word in text_WC_dict:
        prediction *=  text_WC_dict.get(word,0) * ((H_WC_dict.get(word, 0) + 1) / (sum(H_WC_dict.values()) + H_count))

        # Now we multiply by the probability of the class existing in the documents
    return prediction * H_prob

# Now we can generate probabilities for the classes our reviews belong to
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater
def make_decision(text):

    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_WC_dict, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_WC_dict, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater
    if negative_prediction > positive_prediction:
        return 0
    return 1

predictions = [make_decision(r[0]) for r in reviews]
# print (predictions)

actual = [int(r[1]) for r in reviews]

from sklearn import metrics

# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions of train Set: {0}".format(metrics.auc(fpr, tpr)))


with open("dev.csv", 'r') as file:
    dev = list(csv.reader(file))

predictions = [make_decision(r[0]) for r in dev]
# print (predictions)

actual = [int(r[1]) for r in dev]

from sklearn import metrics

# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions of Dev Set: {0}".format(metrics.auc(fpr, tpr)))


with open("test.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[0]) for r in test]
# print (predictions)

actual = [int(r[1]) for r in test]

from sklearn import metrics

# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions of Test Set: {0}".format(metrics.auc(fpr, tpr)))
