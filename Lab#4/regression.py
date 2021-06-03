import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scaling(df):
    sc = StandardScaler()
    ndf = pd.DataFrame(sc.fit_transform(df))
    ndf.columns = df.columns.values
    ndf.index = df.index.values

    return ndf


warnings.filterwarnings(action='ignore')

# Read dataset
df = pd.read_csv('linear_regression_data.csv', encoding='utf-8')
# Scaling data before using model
df = scaling(df)
# Independent feature
X = df['Distance'].to_frame()
# Target feature
y = df['Delivery Time'].to_frame()

# Split the dataset into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Use a linear regression
model = LinearRegression()
# Learn using training set
model.fit(X=X_train, y=y_train)
# Predict using test set
model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy: %0.3f" % accuracy)
