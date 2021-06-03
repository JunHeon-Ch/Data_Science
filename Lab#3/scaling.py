import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Read csv file
df = pd.read_csv('C:/Users/MSI/Desktop/JH/O/Programming_PYTHON/Data_Science/Lab#3/bmi_data_lab3.csv')
df.rename(columns={'Height (Inches)':'Height', 'Weight (Pounds)':'Weight'}, inplace=True)

# Print dataset statistical data
print('Statistical Data')
print(df.describe())
print('\n\n')

# Print feature names & data types
print('feature names & data types')
print(df.info())


# Plot height histogram for each BMI value
grid = sns.FacetGrid(df, col='BMI')
grid.map(plt.hist, "Height", bins=10)
grid.set_axis_labels('Height (Inches)', 'Number of people')
plt.show()

# Plot weight histogram for each BMI value
grid = sns.FacetGrid(df, col='BMI')
grid.map(plt.hist, "Weight", bins=10)
grid.set_axis_labels('Weight (Pounds)', 'Number of people')
plt.show()


# Plot scaling results for height, weight
# Get 'Height' and 'Weight' columns
height = df['Height']
weight = df['Weight']

# Standard Scaling
heightStandardScale = StandardScaler().fit_transform(height[:, np.newaxis]).reshape(-1)
weightStandardScale = StandardScaler().fit_transform(weight[:, np.newaxis]).reshape(-1)

# MinMax Scaling
heightMinMaxScale = MinMaxScaler().fit_transform(height[:, np.newaxis]).reshape(-1)
weightMinMaxScale = MinMaxScaler().fit_transform(weight[:, np.newaxis]).reshape(-1)

# Robust Scaling
heightRobustScale = RobustScaler().fit_transform(height[:, np.newaxis]).reshape(-1)
weightRobustScale = RobustScaler().fit_transform(weight[:, np.newaxis]).reshape(-1)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(6, 5))

# Plot Standard Scaling result
ax1.set_title('Standard Scaling')
sns.kdeplot(heightStandardScale, ax=ax1, color='red', label='h')
sns.kdeplot(weightStandardScale, ax=ax1, color='blue', label='w')

# Plot MinMax Scaling result
ax2.set_title('MinMax Scaling')
sns.kdeplot(heightMinMaxScale, ax=ax2, color='red', label='h')
sns.kdeplot(weightMinMaxScale, ax=ax2, color='blue', label='w')

# Plot Robust Scaling result
ax3.set_title('Robust Scaling')
sns.kdeplot(heightRobustScale, ax=ax3, color='red', label='h')
sns.kdeplot(weightRobustScale, ax=ax3, color='blue', label='w')
plt.legend()

plt.show()
