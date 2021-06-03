import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings(action='ignore')

# Read bagging dataset
samples = []
for i in range(1, 11):
    title = 'Iris_bagging_dataset (' + str(i) + ').csv'
    sample = pd.read_csv(title, encoding='utf-8')
    samples.append(sample)

# Read testing dataset
test = pd.read_csv('Iris.csv', encoding='utf-8')
# split independent columns and target column in testing set
test_X = test.drop('Species', axis=1)
test_y = test['Species']

# Data frame for storing predicted results using decision stump classifier model
results = pd.DataFrame(index=np.arange(10), columns=np.arange(len(test)))


def generateDecisionTree(i):
    # Read data
    sample = samples[i]
    # Split independent columns and target column in training set
    X = sample.drop('Species', axis=1)
    y = sample['Species']

    # Generate a decision stump classifier model and train with training set
    dtc = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    # Predict the labels with testing set
    pred = dtc.predict(test_X)
    # Store result into DataFrame
    for j in range(len(test)):
        results.iat[i, j] = pred[j]


# Find the most frequent element
def majorityCount(column):
    elements, count = np.unique(column, return_counts=True)
    return elements[count.argmax()]


# The majority vote determines the final forecast
def majorityVoting():
    # List for storing final forecast
    final_result = []
    for i in range(len(test)):
        final_result.append(majorityCount(results.iloc[:, i]))

    return final_result


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


# Run 10 bagging rounds
for i in range(len(samples)):
    generateDecisionTree(i)

final_result = majorityVoting()

# calculate the accuracy using confusion matrix and classification report
print(confusionMatrix(test_y, final_result), '\n')
print(classification_report(test_y, final_result))
