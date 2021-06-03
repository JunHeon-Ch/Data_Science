import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

# New data is entered by the user
new_height = int(input("Enter new data's Height(cm) : "))
new_weight = int(input("Enter new data's Weight(kg) : "))
k = int(input("This is K-Nearest Neighbors Algorithm, Enter the k (ex)3,4,5..) : "))

# training data
data = pd.read_csv("C://Users/tldus/Desktop/training dataset.csv")
print(data)
print()

# new data
newdata = pd.DataFrame({"Height(cm)": [new_height],
                        "Weight(kg)": [new_weight]},
                       columns=["Height(cm)", "Weight(kg)", "T shirt size"])

# visual representation
sns.scatterplot(x='Weight(kg)',
                y='Height(cm)',
                hue='T shirt size',
                style='T shirt size',
                s=100,
                data=data)
plt.scatter(newdata.loc[0]["Weight(kg)"], newdata.loc[0]["Height(cm)"], c="red")
plt.show()


# method for calculating distance
def cal_distance(seq, index):
    if seq == 1:
        distance = ((data.loc[index]["Height(cm)"] - newdata.loc[0]["Height(cm)"]) ** 2 + (
                data.loc[index]["Weight(kg)"] - newdata.loc[0]["Weight(kg)"]) ** 2)
    if seq == 2:
        distance = ((data2.loc[index]["Height(cm)"] - newdata2.loc[0]["Height(cm)"]) ** 2 + (
                data2.loc[index]["Weight(kg)"] - newdata2.loc[0]["Weight(kg)"]) ** 2)
    return sqrt(distance)


# Method of finding a close neighbor
def get_neighbors(seq, num_neighbors):
    distances = list()
    distances2 = list()
    for i in range(18):
        dist = cal_distance(seq, i)
        distances.append((i, dist))
        distances2.append(dist)
    # Adding a distance to a dataset
    if seq == 1:
        data["Distance"] = distances2
    if seq == 2:
        distances2.append("NaN")
        data2["Distance"] = distances2
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Check the neighbors T shirt size to predict the T shirt size of new data.
def prediction(seq, neighbors):
    num_m = 0
    num_l = 0
    if seq == 1:
        for neighbor in neighbors:
            if (data.loc[neighbor]["T shirt size"] == "M"):
                num_m = num_m + 1
            if (data.loc[neighbor]["T shirt size"] == "L"):
                num_l = num_l + 1
    if seq == 2:
        for neighbor in neighbors:
            if (data2.loc[neighbor]["T shirt size"] == "M"):
                num_m = num_m + 1
            if (data2.loc[neighbor]["T shirt size"] == "L"):
                num_l = num_l + 1
    if (num_m > num_l):
        print("T shirt size M:", num_m, "T shirt size L:", num_l)
        print("New data's T shirt size is predicted M")
        print()
    if (num_m < num_l):
        print("T shirt size M:", num_m, "T shirt size L:", num_l)
        print("New data's T shirt size is predicted L")
        print()


neighbors = get_neighbors(1, k)
prediction(1, neighbors)

print("Nearest neighbors's index")
for neighbor in neighbors:
    print(neighbor)

print(data)
print()

# Normalized Training Dataset

data_copy = pd.DataFrame(
    {'Height(cm)': data['Height(cm)'], 'Weight(kg)': data['Weight(kg)'], 'T shirt size': data['T shirt size']})
newdata2 = {'Height(cm)': new_height, 'Weight(kg)': new_weight}
data_copy = data_copy.append(newdata2, ignore_index=True)

dbp = {'M': 1, 'L': 0}
data_copy['T shirt size'] = data_copy['T shirt size'].replace(dbp)

scaler = preprocessing.StandardScaler()  # StandardScaler
data2 = scaler.fit_transform(data_copy)
data2 = pd.DataFrame(data2, columns=["Height(cm)", "Weight(kg)", "T shirt size"])

data2['T shirt size'] = data['T shirt size']
print(data2)
print()

newdata2 = pd.DataFrame({'Height(cm)': [data2.loc[18]['Height(cm)']], 'Weight(kg)': [data2.loc[18]['Weight(kg)']]},
                        columns=['Height(cm)', 'Weight(kg)'])

neighbors = get_neighbors(2, k)
prediction(2, neighbors)

print("Nearest neighbors's index")
for neighbor in neighbors:
    print(neighbor)

print(data2)
print()
