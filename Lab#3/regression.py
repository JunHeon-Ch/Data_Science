import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing


# Identify all dirty records with likely-wrong or missing height or weight values
# Replace all dirty records to NaN
def findWrongData(df):
    for i in range(len(df)):
        # If the height is less than 10 inches or more than 90 inches, it is considered a dirty record.
        if df['Height'][i] < 10 or df['Height'][i] > 90:
            df.replace({df['Height'][i]: np.NaN}, inplace=True)
        # If the weight is less than 10 pounds or more than 400 pounds, it is considered a dirty record.
        if df['Weight'][i] < 10 or df['Weight'][i] > 400:
            df.replace({df['Weight'][i]: np.NaN}, inplace=True)


# Predict missing values using linear regression
def predictMissingValue(df, w, b):
    # List for inserting predicted missing values
    x_mv = []
    y_mv = []

    # Get 'Height' and 'Weight' columns
    x = np.array(df['Height'])
    y = np.array(df['Weight'])

    # Store the predicted values through linear registration equation.
    for i in range(len(df)):
        if math.isnan(x[i]):
            x_mv.append((y[i] - b) / w)
            y_mv.append(y[i])
        elif math.isnan(y[i]):
            x_mv.append(x[i])
            y_mv.append(x[i] * w + b)

    return x_mv, y_mv


# Compute linear regression equation and find slope and y-intercept
def makeRegression(df):
    # Drop all the rows that have Nan values in the data frame.
    tempdf =df.dropna(axis=0, how='any')

    # Get 'Height' and 'Weight' columns
    height = tempdf['Height']
    weight = tempdf['Weight']

    # Compute linear regression equation through 'Height', 'Weight'
    reg = linear_model.LinearRegression()
    reg.fit(height[:, np.newaxis], weight)

    w = reg.coef_       # slope
    b = reg.intercept_  # y-intercept

    # Predicting missing values through the linear registration calculation computed.
    height_mv, weight_mv = predictMissingValue(df, w, b)

    # Plots data without NaN values into the scatter
    plt.scatter(height, weight)
    # Plots data that predicted NaN values into the scatter
    plt.scatter(height_mv, weight_mv, color='green')
    # Plots linear regression equation
    plt.plot(height, w * height + b, 'r', label='Y = wX + b')

# read csv file
df = pd.read_csv('C:/Users/MSI/Desktop/JH/O/Programming_PYTHON/Data_Science/Lab#3/bmi_data_lab3.csv')
df.rename(columns={'Height (Inches)': 'Height', 'Weight (Pounds)': 'Weight'}, inplace=True)

# Identify all dirty records and Replace them to NaN
findWrongData(df)

# Compute linear regression equation in total dataset
makeRegression(df)

# Total Dataset
plt.title('Input Dataset')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()

# Transform 'Sex' categorical feature
enc = preprocessing.OrdinalEncoder()
enc.fit(df['Sex'][:, np.newaxis])
df['Sex'] = enc.transform(df['Sex'][:, np.newaxis]).reshape(-1)


mdf = df[df['Sex'] == 1.0]  # 'Sex' = 'Male'
# Compute linear regression equation in men dataset
makeRegression(mdf)

# Men Dataset
plt.title('Men Dataset')
plt.xlabel('Men Height (Inches)')
plt.ylabel('Men Weight (Pounds)')
plt.show()

fdf = df[df['Sex'] == 0.0]  # 'Sex' = 'Female'
# Compute linear regression equation in women dataset
makeRegression(fdf)

# Women Dataset
plt.title('Women Dataset')
plt.xlabel('Women Height (Inches)')
plt.ylabel('Women Weight (Pounds)')
plt.show()

# Drop the row with NaN value in BMI column
bmi = np.array(df['BMI'])
for i in range(len(df)):
    if math.isnan(bmi[i]):
        df.drop([i], inplace=True)

# Transform 'BMI' categorical feature
enc = preprocessing.OrdinalEncoder()
enc.fit(df['BMI'][:, np.newaxis])
df['BMI'] = enc.transform(df['BMI'][:, np.newaxis]).reshape(-1)

df_bmi = df[df['BMI'] == 0.0]  # BMI = 1
# Compute linear regression equation in 'BMI=1' dataset
makeRegression(df_bmi)

# 'BMI=1' Dataset
plt.title("'BMI = 1' Dataset")
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()

df_bmi = df[df['BMI'] == 1.0]  # BMI = 2
# Compute linear regression equation in 'BMI=2' dataset
makeRegression(df_bmi)

# 'BMI=2' Dataset
plt.title("'BMI = 2' Dataset")
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()

df_bmi = df[df['BMI'] == 2.0]  # BMI = 3
# Compute linear regression equation in 'BMI=3' dataset
makeRegression(df_bmi)

# 'BMI=3' Dataset
plt.title("'BMI = 3' Dataset")
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()