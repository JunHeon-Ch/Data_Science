import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings(action='ignore')


# Encoding target columns
def encoder(y):
    y = preprocessing.LabelEncoder().fit_transform(y)

    return y


# Scaling independent columns
def scaling(X):
    X = MinMaxScaler().fit_transform(X)

    return X


# Read bagging dataset
samples = []
for i in range(1, 11):
    title = 'Iris_bagging_dataset (' + str(i) + ').csv'
    sample = pd.read_csv(title, encoding='utf-8')
    samples.append(sample)

# Read testing dataset
test = pd.read_csv('Iris.csv', encoding='utf-8')
# split independent columns and target column in testing set
testX = test.drop('Species', axis=1)
testy = test['Species']


# Plot confusion matrix using heatmap
def confusionMatrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)

    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.title('Confusion Matrix', fontsize=20)
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], fontsize=12, rotation=45)
    ax.set_yticklabels(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], fontsize=12, rotation=45)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.show()

    return conf_matrix


classifiers = []
dtc = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=1)
# Run 10 bagging rounds
for i in range(len(samples)):
    # Read data
    sample = samples[i]
    # Split independent columns and target column in training set
    X = sample.drop('Species', axis=1)
    y = sample['Species']
    X = scaling(X)
    y = encoder(y)

    # Generate a decision stump classifier model and train with training set
    fit_dtc = dtc.fit(X, y)

    classifiers.append(fit_dtc)

testX = scaling(testX)
testy = encoder(testy)
vc = VotingClassifier(estimators=[('0', classifiers[0]), ('1', classifiers[1]), ('2', classifiers[2]),
                                  ('3', classifiers[3]), ('4', classifiers[4]), ('5', classifiers[5]),
                                  ('6', classifiers[6]), ('7', classifiers[7]), ('8', classifiers[8]),
                                  ('9', classifiers[9])], voting='hard').fit(testX, testy)

# predict the labels using voting
pred = vc.predict(testX)

# calculate the accuracy using confusion matrix and classification report
print(confusionMatrix(testy, pred), '\n')
print(classification_report(testy, pred))