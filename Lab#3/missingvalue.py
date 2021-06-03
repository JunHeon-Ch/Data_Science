import numpy as np
import pandas as pd

# Identify all dirty records with likely-wrong or missing height or weight values
# Replace all likely-wrong values to NaN
def findWrongData(df):
    for i in range(len(df)):
        # If the height is less than 10 inches or more than 90 inches, it is considered a dirty record.
        if df['Height'][i] < 10 or df['Height'][i] > 90:
            df.replace({df['Height'][i]: np.NaN}, inplace=True)
        # If the weight is less than 10 pounds or more than 400 pounds, it is considered a dirty record.
        if df['Weight'][i] < 10 or df['Weight'][i] > 400:
            df.replace({df['Weight'][i]: np.NaN}, inplace=True)


# read csv file
df = pd.read_csv('C:/Users/MSI/Desktop/JH/O/Programming_PYTHON/Data_Science/Lab#3/bmi_data_lab3.csv')
df.rename(columns={'Height (Inches)': 'Height', 'Weight (Pounds)': 'Weight'}, inplace=True)

findWrongData(df)


# Print number of rows with NAN
cnt = 0
s = df.isnull().sum(axis=1)
for i in range(len(s)):
    if s[i] != 0:
        cnt += 1
print('Number of rows with NAN')
print(cnt)

# number of NAN for each column
print('\n\nNumber of NAN for each column')
print(df.isna().sum())

# Extract all rows without NAN
print('\n\nExtract all rows without NAN')
print(df.dropna(axis=0, how='any'))

# Fill NAN with mean
print('\n\nFill NAN with mean')
print(df.fillna(df.mean()))

# Fill NAN with median
print('\n\nFill NAN with median')
print(df.fillna(df.median()))

# Fill NaN with the front value.
print('\n\nFill NAN with ffill function')
print(df.fillna(axis=0, method='ffill'))

# Fill NaN with the back value.
print('\n\nFill NAN with bfill function')
print(df.fillna(axis=0, method='bfill'))