import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.decomposition import PCA
import time
from sklearn.datasets import fetch_openml

# mnist = fetch_openml('fashion-mnist')
# train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target,
#                                                             test_size=1 / 7.0, random_state=0)
# print(train_img.head())
# print(train_lbl.head())

# Read the CSV dataset file
mnist = pd.read_csv('fashion-mnist.csv')
# In the mnist dataset, pixel features excluding labels are extracted.
x = mnist.drop(['label'], axis=1)

'''a'''
print("Standard Scaling")
# Normalizing using the stardard scaler
standScaler = preprocessing.StandardScaler()
df_std = standScaler.fit_transform(x)
df_std = pd.DataFrame(df_std)
print(df_std.head(30))
# Normalizing using the minmax scaler
print("MinMax Scaling")
minmaxScaler = preprocessing.MinMaxScaler()
df_mms = minmaxScaler.fit_transform(x)
df_mms = pd.DataFrame(df_mms)
print(df_mms.head(30))
# Normalizing using the MaxAbs scaler
print("MaxAbs Scaling")
MaxAbsScaler = preprocessing.MaxAbsScaler()
df_Mas = MaxAbsScaler.fit_transform(x)
df_Mas = pd.DataFrame(df_mms)
print(df_Mas.head(30))
# Normalizing using the robust scaler
print("Robust Scaling")
RobustScaler = preprocessing.RobustScaler()
df_Rob = RobustScaler.fit_transform(x)
df_Rob = pd.DataFrame(df_Rob)
print(df_Rob.head(30))

'''b, c, d'''
# Split the whole dataset into train and test datasets
train, test = train_test_split(mnist, test_size=1 / 7.0, shuffle=False, random_state=0)

# Separate label features and pixel features from train and test datasets.
train_images = train.drop(["label"], axis=1)
train_label = train["label"]

test_images = test.drop(["label"], axis=1)
test_label = test["label"]

# When PCA is 0.8, apply train image to pca1 model.
start1 = time.time()
pca1 = PCA(n_components=.80)
pca1.fit(train_images)
train_images1 = pca1.transform(train_images)
test_images1 = pca1.transform(test_images)

# Apply the train image value to which the model of pca1 is applied using logisticRegression.
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(train_images1, train_label)

# In the trained model, the label value is predicted using test_image.
pred1 = logisticRegr.predict(test_images1)
score1 = logisticRegr.score(test_images1, test_label)
end1 = time.time()
print(pred1)

# When PCA is 0.85, apply train image to pca2 model.
start2 = time.time()
pca2 = PCA(n_components=0.85)
pca2.fit(train_images)
train_images2 = pca2.transform(train_images)
test_images2 = pca2.transform(test_images)

# Apply the train image value to which the model of pca2 is applied using logisticRegression.
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(train_images2, train_label)
# In the trained model, the label value is predicted using test_image.
pred2 = logisticRegr.predict(test_images2)
score2 = logisticRegr.score(test_images2, test_label)
end2 = time.time()
print(pred2)

# When PCA is 0.90, apply train image to pca3 model.
start3 = time.time()
pca3 = PCA(n_components=0.90)
pca3.fit(train_images)
train_images3 = pca3.transform(train_images)
test_images3 = pca3.transform(test_images)

# Apply the train image value to which the model of pca3 is applied using logisticRegression.
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(train_images3, train_label)
# In the trained model, the label value is predicted using test_image.
pred3 = logisticRegr.predict(test_images3)
score3 = logisticRegr.score(test_images3, test_label)
end3 = time.time()
print(pred3)

# When PCA is 0.95, apply train image to pca4 model.
start4 = time.time()
pca4 = PCA(n_components=0.95)
pca4.fit(train_images)
train_images4 = pca4.transform(train_images)
test_images4 = pca4.transform(test_images)

# Apply the train image value to which the model of pca4 is applied using logisticRegression.
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(train_images4, train_label)
# In the trained model, the label value is predicted using test_image.
pred4 = logisticRegr.predict(test_images4)
score4 = logisticRegr.score(test_images4, test_label)
end4 = time.time()
print(pred4)

# When PCA is 0.99, apply train image to pca5 model.
start5 = time.time()
pca5 = PCA(n_components=0.99)
pca5.fit(train_images)
train_images5 = pca5.transform(train_images)
test_images5 = pca5.transform(test_images)

# Apply the train image value to which the model of pca5 is applied using logisticRegression.
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(train_images5, train_label)
# In the trained model, the label value is predicted using test_image.
pred5 = logisticRegr.predict(test_images5)
score5 = logisticRegr.score(test_images5, test_label)
end5 = time.time()
print(pred5)

# When PCA is 1.00, apply train image to pca5 model.
start6 = time.time()
pca6 = PCA(n_components=784)
pca6.fit(train_images)
train_images6 = pca6.transform(train_images)
test_images6 = pca6.transform(test_images)
# Apply the train image value to which the model of pca6 is applied using logisticRegression.
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(train_images6, train_label)
# In the trained model, the label value is predicted using test_image.
pred6 = logisticRegr.predict(test_images6)
score6 = logisticRegr.score(test_images6, test_label)
end6 = time.time()
print(pred6)

# Create a table with the number of components, time, and accuracy according to each Pac value
x = pd.DataFrame({'Variance Retained': [0.80, 0.85, 0.90, 0.95, 0.99, 1.00],
                  'Number of Components': [pca1.n_components_, pca2.n_components_, pca3.n_components_,
                                           pca4.n_components_, pca5.n_components_, pca6.n_components_],
                  'Time (seconds)': [end1 - start1, end2 - start2, end3 - start3, end4 - start4, end5 - start5,
                                     end6 - start6],
                  'Accuracy': [score1, score2, score3, score4, score5, score6]})
print(x)

# Standard Scaling accuracy - I will only upload the code of the changed parts
# In the mnist dataset, pixel features excluding labels are extracted.
x = mnist.iloc[:, 1:]
y = mnist.iloc[:, 0]
print("Standard Scaling")
# Normalizing using the stardard scaler
standScaler = preprocessing.StandardScaler()
df_std = standScaler.fit_transform(x)
df_std = pd.DataFrame(df_std)
print(df_std.head(30))

# Split the whole dataset into train and test datasets with standard scaled dataset
train, test = train_test_split(df_std, test_size=1 / 7.0, shuffle=False, random_state=0)
label_train, label_test = train_test_split(y, test_size=1 / 7.0, shuffle=False, random_state=0)
# Separate label features and pixel features from train and test datasets.
train_images = train
train_label = label_train
test_images = test
test_label = label_test

# MinMax Scaling
# In the mnist dataset, pixel features excluding labels are extracted.
x = mnist.iloc[:, 1:]
y = mnist.iloc[:, 0]

# Normalizing using the minmax scaler
print("MinMax Scaling")
minmaxScaler = preprocessing.MinMaxScaler()
df_mms = minmaxScaler.fit_transform(x)
df_mms = pd.DataFrame(df_mms)
print(df_mms.head(30))

# Split the whole dataset into train and test datasets with minmax scaled dataset
train, test = train_test_split(df_mms, test_size=1 / 7.0, shuffle=False, random_state=0)

label_train, label_test = train_test_split(y, test_size=1 / 7.0, shuffle=False, random_state=0)
# Separate label features and pixel features from train and test datasets.
train_images = train
train_label = label_train

test_images = test
test_label = label_test

# MaxAbs Scaling
# In the mnist dataset, pixel features excluding labels are extracted.
x = mnist.iloc[:, 1:]
y = mnist.iloc[:, 0]

# Normalizing using the MaxAbs scaler
print("MaxAbs Scaling")
MaxAbsScaler = preprocessing.MaxAbsScaler()
df_Mas = MaxAbsScaler.fit_transform(x)
df_Mas = pd.DataFrame(df_Mas)

# Split the whole dataset into train and test datasets with maxabs scaled dataset
train, test = train_test_split(df_Mas, test_size=1 / 7.0, shuffle=False, random_state=0)

label_train, label_test = train_test_split(y, test_size=1 / 7.0, shuffle=False, random_state=0)

# Separate label features and pixel features from train and test datasets.
train_images = train
train_label = label_train
test_images = test
test_label = label_test

# Robust Scaling
# In the mnist dataset, pixel features excluding labels are extracted.
x = mnist.iloc[:, 1:]
y = mnist.iloc[:, 0]

# Normalizing using the robust scaler
print("Robust Scaling")
RobustScaler = preprocessing.RobustScaler()
df_Rob = RobustScaler.fit_transform(x)
df_Rob = pd.DataFrame(df_Rob)

# Split the whole dataset into train and test datasets with robust scaled dataset
train, test = train_test_split(df_Rob, test_size=1 / 7.0, shuffle=False, random_state=0)

label_train, label_test = train_test_split(y, test_size=1 / 7.0, shuffle=False, random_state=0)

# Separate label features and pixel features from train and test datasets.
train_images = train
train_label = label_train

test_images = test
test_label = label_test
