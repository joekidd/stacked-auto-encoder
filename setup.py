#!/usr/bin/env python
"""
    Implementation of the preprocessing for the horse racing data.

    The scripts gets rid of unneeded features and converts textual features
    into numerical features, so that the neural network can operate on it.

    Please notice that I have absolutely no clue about horse racing and all
    this operations are based more on intuition and logical thinking rather
    than on horse racing knowledge.
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

training_set = pd.read_csv("data/training.csv")
testing_set = pd.read_csv("data/testing.csv")

# Concat two sets two apply uniform set of operations on both of them
concat_set = pd.concat([training_set, testing_set])

# Store classes
classes = concat_set['rank']
classes = [ c - 1 for c in classes]

# It seems that those params are not necessary
concat_set = concat_set.drop(
    ['behindn', 'rank', 'blinkerclasses', 'raceref_id'], 1
)

# The year will be always different between training and testing set, then
# I have decided to get rid of it. However the performance of the horse
# may vary from month to month/day of month.
concat_set['date'] = concat_set['date'].apply(lambda x: x[5:])

# Extract textual features
text_features_labels = [
    'course', 'date', 'going', 'horse', 'jockey', 'track_name', 'trainer'
]
text_features = concat_set[text_features_labels]

# Encode the featuers into numerical form
for label in text_features_labels:
    encoder = preprocessing.LabelEncoder()
    concat_set[label] = encoder.fit_transform(text_features[label])

# Scale features
scaler = preprocessing.MinMaxScaler()
scaled_features = scaler.fit_transform(concat_set)
#scaled_labels = preprocessing.scale(classes)

# Save preprocessed data
x, y = training_set.shape
np.savetxt('data/train.csv', scaled_features[:x], delimiter=',')
np.savetxt('data/train.labels.csv', classes[:x], delimiter=',')

np.savetxt('data/test.csv', scaled_features[x:], delimiter=',')
np.savetxt('data/test.labels.csv', classes[x:], delimiter=',')

print "Preprocessing completed (data/train.* data/test.* files created)"
