import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Compute e = w - w’ where w’ is obtained for height using linear regression equation.
# Normalize the e values.
def normalize(df):
    # Get 'Height' and 'Weight' columns
    height = df['Height']
    weight = df['Weight']

    # Compute linear regression equation E through 'Height', 'Weight'
    reg = linear_model.LinearRegression()
    reg.fit(height[:, np.newaxis], weight)

    a = reg.coef_  # slope
    b = reg.intercept_  # y-intercept

    # Compute e = w - w’ where w’ is obtained for h using E
    e = []
    for i in range(len(df)):
        w = a * height[i] + b
        e.append(weight[i] - w)

    # Normalize the e values, i.e., compute Z(e) = [e-μ(e)]/σ(e)
    z = np.zeros(len(e))
    mean = np.mean(e)  # μ(e)
    std = np.std(e)  # σ(e)
    for i in range(len(e)):
        z[i] = (e[i] - mean) / std

    return z


# Decide a value "(alpha >= 0)" from histogram
# If z-score of e > alpha, BMI is 4, else if z-score of e < -alpha, BMI is 0
def bmiEstimate(df, z):
    a = 1.5  # Decide alpha is 1.5
    
    for i in range(len(z)):
        if z[i] < -a:
            df.iloc[i, 4] = 0
        elif z[i] > a:
            df.iloc[i, 4] = 4

    return df


# read excel file
df = pd.read_excel('C:/Users/MSI/Desktop/JH/O/Programming_PYTHON/Data_Science/HW#3/bmi_data_phw3.xlsx', 'dataset')
df.rename(columns={'Height (Inches)': 'Height', 'Weight (Pounds)': 'Weight'}, inplace=True)

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

# Normalize e values for total data
z = normalize(df)

# Plot a histogram showing the distribution of Z(e) for total input data
plt.hist(z, bins=10, rwidth=0.8, color='blue')
plt.title('Distribution of Ze')
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

# Transform 'Sex' categorical feature
enc = preprocessing.OrdinalEncoder()
enc.fit(df['Sex'][:, np.newaxis])
df['Sex'] = enc.transform(df['Sex'][:, np.newaxis]).reshape(-1)

# Normalize e values for men dataset
mdf = df[df['Sex'] == 1.0]  # 'Sex' = 'Male'
mdf = mdf.reset_index(drop=True)
zm = normalize(mdf)

# Plot a histogram showing the distribution of Z(e) for men input data
# 'Sex' = 'Male' Dataset
plt.hist(zm, bins=10, rwidth=0.8, color='red')
plt.title('Distribution of Ze for Men')
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

# Normalize e values for women dataset
fdf = df[df['Sex'] == 0.0]  # 'Sex' = 'Female'
fdf = fdf.reset_index(drop=True)
zf = normalize(fdf)

# Plot a histogram showing the distribution of Z(e) for women input data
# 'Sex' = 'Female' Dataset
plt.hist(zf, bins=10, rwidth=0.8, color='green')
plt.title('Distribution of Ze for Women')
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()

# for total dataset
print('===== Origin Total Dataset =====')
print(df)
df = bmiEstimate(df, z)
print('\n===== BMI Estimated Total Dataset ======\n')
print(df)

# for 'Sex' = 'Male' dataset
print("\n===== Origin 'Sex' = 'Male' Dataset =====\n")
print(mdf)
mdf = bmiEstimate(mdf, zm)
print("\n===== BMI Estimated 'Sex' = 'Male' Dataset =====\n")
print(mdf)

# for 'Sex' = 'Female' dataset
print("\n===== Origin 'Sex' = 'Female' Dataset =====\n")
print(fdf)
fdf = bmiEstimate(fdf, zf)
print("\n===== BMI Estimated 'Sex' = 'Female' Dataset =====\n")
print(fdf)