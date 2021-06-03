import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler


# initialize dataset
def initDataset(dataset, newH, newW):
    # add new column 'Distance' and 'Rank'
    dataset['Distance'] = 0
    dataset['Rank'] = np.NaN

    # insert target data to dataset
    index = len(dataset)
    dataset.loc[index, 'Height'] = newH
    dataset.loc[index, 'Weight'] = newW

    # Because the units and sizes of height and weight do not fit,
    # the height and weight column are normalized, respectively.
    normalize(dataset, dataset['Height'], dataset['Weight'])


# Standard Scaling
def normalize(dataset, height, weight):
    dataset['Height'] = StandardScaler().fit_transform(height[:, np.newaxis]).reshape(-1)
    dataset['Weight'] = StandardScaler().fit_transform(weight[:, np.newaxis]).reshape(-1)


# Calculate the distance of data from the new data height and weight.
def calcDistance(dataset):
    index = len(dataset) - 1
    targetH = dataset.loc[index, 'Height']
    targetW = dataset.loc[index, 'Weight']

    for i in range(index):
        height = dataset.loc[i, 'Height']
        weight = dataset.loc[i, 'Weight']

        dataset.loc[i, 'Distance'] = getDistance(height, weight, targetH, targetW)


# Return distance of entered height and weight between each row's height and weight
def getDistance(h, w, targetH, targetW):
    return math.sqrt((targetH - h) ** 2 + (targetW - w) ** 2)


# Ranked based on distance from target data.
def rankByDist(dataset):
    # Just pull out the 'Distance' column and sort it.
    rank = dataset['Distance'].sort_values()

    # Enter the ranking value in the 'Rank' column.
    i = 1
    for idx in rank.index:
        if i <= k:
            dataset.loc[idx, 'Rank'] = i
            i += 1


# Determine the 'size' of the target data through the 'Rank' column values.
def evaluate(dataset, taget_name):
    # Remove unnecessary row.
    evaluate_dataset = dataset.dropna(axis=0, how='any')

    # Select the most target value among records with the 'Rank' value.
    # If 'M' is more than 'L', 'Size' of the target data is 'M'.
    # Otherwise, the target data's size is 'L'.
    return np.unique(evaluate_dataset[taget_name])[
        np.argmax(np.unique(evaluate_dataset[taget_name], return_counts=True)[1])]


# Read dataset
dataset = pd.read_excel('PHW4_2_dataset.xlsx', 'dataset')
dataset.rename(columns={'Height(cm)': 'Height', 'Weight(kg)': 'Weight'}, inplace=True)

# Get new customer's height and weight and K
newH = int(input('Enter the height: '))
newW = int(input('Enter the weight: '))
k = int(input('Enter K(Only odd number) : '))

new_dataset = dataset.copy()
initDataset(new_dataset, newH, newW)
calcDistance(new_dataset)
rankByDist(new_dataset)
target = evaluate(new_dataset, 'Size')

print('========== Original Dataset ==========')
print(dataset)
print('========== Result Dataset ==========')
print(new_dataset)
print('========== Target Result ==========')
print([newH, newW, target])
