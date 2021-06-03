import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def encoder(df):
    features = list(df)
    for feature in features:
        df[feature] = preprocessing.LabelEncoder().fit_transform(df[feature])


warnings.filterwarnings(action='ignore')

# Read dataset
df = pd.read_csv('decision_tree_data.csv', encoding='utf-8')
# Encoding data before using model
encoder(df)
# Independent feature
X = df.drop(['interview'], axis=1)
# Target feature
y = df['interview']

# Split the dataset into 90% for training and 10% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)

# Use a decision tree
classifier = DecisionTreeClassifier()
# Learn using training set
classifier.fit(X_train, y_train)
# Predict using test set
classifier.predict(X_test)

# Evaluate the model
accuracy = classifier.score(X_test, y_test)
print("Accuracy: %0.3f" % accuracy)