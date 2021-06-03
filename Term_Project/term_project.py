import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


# Import dataset
def readDataset():
    df = pd.read_csv('dataset_modified_v2.csv')

    return df


# Print 5 rows in dataset
def datasetInfo(df):
    print(df.head())
    print('\n\n')


# Numerical data statistics information
def statisticalInfo(df):
    print('Numerical Data Statistics Information')
    print(df.describe())
    print('\n\n')


# Feature information and data types
def featureInfo(df):
    print('feature names & data types')
    print(df.info())
    print('\n\n')


def verifyMissingValue(df):
    # Verifying null values
    print('전체 dataset 중 missing value')
    print(df.isna().any())
    print('\n\n')

    print('Total missing value by column')
    print(df.isna().sum())
    print('\n\n')
    df.isna().sum().plot.bar(title='Total missing value by column', rot=45)
    plt.show()

    # find percentage of missing values for each column
    missing_values = df.isnull().mean() * 100
    print('컬럼별 missing value 확률')
    print(missing_values)
    print('\n\n')

    # how many total missing values do we have?
    total_cells = np.product(df.shape)
    total_missing = df.isna().sum().sum()

    # percent of data that is missing
    total_missing_values = (total_missing / total_cells) * 100
    print('percent of data that is missing')
    print(total_missing_values)
    print('\n\n')


# In the categorical column, change all characters to lowercase because data is not available if they are mixed with
# upper and lowercase letters.
def replaceUnusableData(df):
    replaced_df = df.copy()

    features = df.columns.values
    for feature in features:
        if replaced_df[feature].dtype == 'object':
            replaced_df[feature] = replaced_df[feature].str.lower()

    return replaced_df


# Replace wrong data with a nan value.
def replaceWrongData(df):
    replaced_df = df.copy()
    job = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
           'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    marital = ['divorced', 'married', 'single']
    education = ['primary', 'secondary', 'tertiary', 'unknown']
    default = ['no', 'yes']
    housing = ['no', 'yes']
    loan = ['no', 'yes']
    contact = ['cellular', 'telephone', 'unknown']
    month = ['apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep']
    poutcome = ["unknown", "other", "failure", "success"]
    count = np.zeros(16)

    for i in range(len(replaced_df)):
        # age column
        # If the age is less than 0 or greater than 100, replace it with nan.
        if replaced_df.iloc[i, 0] < 0 or replaced_df.iloc[i, 0] > 100:
            replaced_df.iloc[i, 0] = np.nan
            count[0] += 1
        # job column
        if replaced_df.iloc[i, 1] not in job:
            replaced_df.iloc[i, 1] = np.NaN
            count[1] += 1
        # marital column
        if replaced_df.iloc[i, 2] not in marital:
            replaced_df.iloc[i, 2] = np.NaN
            count[2] += 1
        # education column
        if replaced_df.iloc[i, 3] not in education:
            replaced_df.iloc[i, 3] = np.NaN
            count[3] += 1
        # default column
        if replaced_df.iloc[i, 4] not in default:
            replaced_df.iloc[i, 4] = np.NaN
            count[4] += 1
        # housing column
        if replaced_df.iloc[i, 6] not in housing:
            replaced_df.iloc[i, 6] = np.NaN
            count[6] += 1
        # loan column
        if replaced_df.iloc[i, 7] not in loan:
            replaced_df.iloc[i, 7] = np.NaN
            count[7] += 1
        # contact column
        if replaced_df.iloc[i, 8] not in contact:
            replaced_df.iloc[i, 8] = np.NaN
            count[8] += 1
        # day column
        if replaced_df.iloc[i, 9] < 1 or replaced_df.iloc[i, 9] > 31:
            replaced_df.iloc[i, 9] = np.NaN
            count[9] += 1
        # month column
        if replaced_df.iloc[i, 10] not in month:
            replaced_df.iloc[i, 10] = np.NaN
            count[10] += 1
        # duration column
        # If the duration is less than 0, replace it with nan.
        if replaced_df.iloc[i, 11] < 0:
            replaced_df.iloc[i, 11] = np.NaN
            count[11] += 1
        # campaign column
        # If the campaign is less than 0, replace it with nan.
        if replaced_df.iloc[i, 12] < 0:
            replaced_df.iloc[i, 12] = np.NaN
            count[12] += 1
        # pdays column
        # If the pdays is less than -1, replace it with nan.
        if replaced_df.iloc[i, 13] < -1:
            replaced_df.iloc[i, 13] = np.NaN
            count[13] += 1
        # previous column
        # If the previous is less than 0, replace it with nan.
        if replaced_df.iloc[i, 14] < 0:
            replaced_df.iloc[i, 14] = np.NaN
            count[14] += 1
        # poutcome column
        if replaced_df.iloc[i, 15] not in poutcome:
            replaced_df.iloc[i, 15] = np.NaN
            count[15] += 1

    print('wrong value count')
    print(count)
    print('\n\n')

    return replaced_df


def cleanMissingData1(df):
    cleaned_df = df.copy()

    # Visualizing numerical data
    # job and deposit
    color_list = ['salmon', 'darkslateblue']

    f, ax = plt.subplots(1, 3, figsize=(18, 8))

    j_df = pd.Series(index=['yes', 'no'], dtype='float64')
    j_df['yes'] = df[df['deposit'] == 'yes']['age'].median()
    j_df['no'] = df[df['deposit'] == 'no']['age'].median()
    j_df.plot.bar(title='Age median', ax=ax[0], color=color_list, rot=45)

    j_df = pd.Series(index=['yes', 'no'], dtype='float64')
    j_df['yes'] = df[df['deposit'] == 'yes']['balance'].median()
    j_df['no'] = df[df['deposit'] == 'no']['balance'].median()
    j_df.plot.bar(title='Balance median', ax=ax[1], color=color_list, rot=45)

    j_df = pd.Series(index=['yes', 'no'], dtype='float64')
    j_df['yes'] = df[df['deposit'] == 'yes']['duration'].median()
    j_df['no'] = df[df['deposit'] == 'no']['duration'].median()
    j_df.plot.bar(title='Duration median', ax=ax[2], color=color_list, rot=45)

    plt.show()

    # Cleaning numerical data
    cleaned_df['age'].fillna(cleaned_df.groupby('deposit')['age'].transform('median'), inplace=True)
    cleaned_df['balance'].fillna(cleaned_df.groupby('deposit')['balance'].transform('median'), inplace=True)
    cleaned_df['duration'].fillna(cleaned_df.groupby('deposit')['duration'].transform('median'), inplace=True)

    # Visualizing categorical data
    # marital and deposit
    fig, axs = plt.subplots(1, 3, figsize=(18, 8))

    j_df = pd.DataFrame()
    j_df['yes'] = df[df['deposit'] == 'yes']['marital'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['marital'].value_counts()
    j_df.plot.bar(title='Marital and deposit', ax=axs[0], color=color_list, rot=45)

    # education and deposit
    j_df = pd.DataFrame()
    j_df['yes'] = df[df['deposit'] == 'yes']['education'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['education'].value_counts()
    j_df.plot.bar(title='Education and deposit', ax=axs[1], color=color_list, rot=45)

    # contact and deposit
    j_df = pd.DataFrame()
    j_df['yes'] = df[df['deposit'] == 'yes']['contact'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['contact'].value_counts()
    j_df.plot.bar(title='Contact and deposit', ax=axs[2], color=color_list, rot=45)

    plt.show()

    # Cleaning categorical data
    depositYes = cleaned_df[cleaned_df['deposit'] == 'yes'].copy()
    depositNo = cleaned_df[cleaned_df['deposit'] == 'no'].copy()

    elements, counts = np.unique(depositYes['marital'].dropna(), return_counts=True)
    depositYes['marital'].fillna(elements[counts.argmax()], inplace=True)
    elements, counts = np.unique(depositNo['marital'].dropna(), return_counts=True)
    depositNo['marital'].fillna(elements[counts.argmax()], inplace=True)

    elements, counts = np.unique(depositYes['education'].dropna(), return_counts=True)
    depositYes['education'].fillna(elements[counts.argmax()], inplace=True)
    elements, counts = np.unique(depositNo['education'].dropna(), return_counts=True)
    depositNo['education'].fillna(elements[counts.argmax()], inplace=True)

    elements, counts = np.unique(depositYes['contact'].dropna(), return_counts=True)
    depositYes['contact'].fillna(elements[counts.argmax()], inplace=True)
    elements, counts = np.unique(depositNo['contact'].dropna(), return_counts=True)
    depositNo['contact'].fillna(elements[counts.argmax()], inplace=True)

    temp = pd.concat([depositYes, depositNo], axis=0)
    cleaned_df = temp

    return cleaned_df


def cleanMissingData2(df):
    cleaned_df = df.copy()

    # correlation heatmap
    corr_df = cleaned_df.dropna()
    corr_df = encoder(corr_df)
    correlationWithTarget(corr_df)

    '''
    Numerical columns
    age -> marital / balance -> age / duration -> deposit
    Categorical columns
    marital -> age / education -> job / contact -> month
    '''

    color_list = ['salmon', 'darkslateblue']
    '''
    Age Columns
    Age is deeply related to marital status, so the group is divided into marital value and filled with the median 
    value of the group. 
    '''
    cleaned_df.groupby('marital')['age'].median().plot.bar(title='Age median', color=color_list, rot=45)
    plt.show()

    g = sns.FacetGrid(cleaned_df, hue="marital", height=4, aspect=2)
    g.map(sns.kdeplot, "age")
    plt.legend()
    plt.show()

    cleaned_df['age'].fillna(cleaned_df.groupby('marital')['age'].transform('median'), inplace=True)

    '''Balance Columns'''
    # Linear regression
    reg_df = cleaned_df.dropna()
    lm = LinearRegression()
    x = reg_df['age'][:, np.newaxis]
    y = reg_df['balance'][:, np.newaxis]
    lm.fit(x, y)

    # Visualize age - balance graph
    plt.figure(figsize=(20, 20))
    plt.plot(reg_df['age'], reg_df['balance'], 'bo')
    plt.plot(x, lm.coef_[0] * x + lm.intercept_, 'r')
    plt.grid(True)
    plt.ylim(-10000, 10000)
    plt.show()

    # replace missing value(balance column)
    for i in cleaned_df.index:
        if np.isnan(cleaned_df.loc[i, 'balance']):
            pred = lm.coef_[0] * cleaned_df.loc[i, 'age'] + lm.intercept_
            cleaned_df.loc[i, 'balance'] = pred

    '''Duration Columns'''
    # replace missing value(duration column)
    cleaned_df.groupby('deposit')['duration'].median().plot.bar(title='Duration median', color=color_list, rot=45)
    plt.show()

    g = sns.FacetGrid(cleaned_df, hue="deposit", height=4, aspect=2)
    g.map(sns.kdeplot, "duration")
    plt.legend()
    plt.show()

    cleaned_df['duration'].fillna(cleaned_df.groupby('deposit')['duration'].transform('median'), inplace=True)

    '''Marital Columns'''
    # Add columns by age group
    # Grouped by age group and filled with the mode value
    # Graph - frequency of marriage by age group
    cleaned_df['age range'] = cleaned_df['age'] // 10
    cleaned_df['By age'] = cleaned_df['age range'] * 10
    cleaned_df['By age'] = cleaned_df['By age'].astype(int)
    sns.catplot(x='By age', kind='count', data=cleaned_df, hue='marital', legend=True, height=6, aspect=2)
    plt.show()

    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 15))
    counter = 0

    for name, group in cleaned_df.groupby(['age range']):
        elements, counts = np.unique(group['marital'].dropna(), return_counts=True)
        temp = cleaned_df[cleaned_df['age range'] == name]['marital'].fillna(elements[counts.argmax()])
        for i in temp.index:
            cleaned_df.loc[i, 'marital'] = temp.loc[i]

        trace_x = counter // 3
        trace_y = counter % 3
        x_pos = np.arange(0, len(elements))

        axs[trace_x, trace_y].bar(x_pos, counts, tick_label=elements)

        title = 'age range = ' + str(int(name * 10))
        axs[trace_x, trace_y].set_title(title)

        for tick in axs[trace_x, trace_y].get_xticklabels():
            tick.set_rotation(45)

        counter += 1

    plt.show()

    cleaned_df = cleaned_df.drop(columns=['age range', 'By age'])

    '''Education Columns'''
    # Grouped by occupation and filled with the lowest possible value
    # Graphs -> Frequency of Education by Job
    fig, axs = plt.subplots(4, 3, sharex=False, sharey=False, figsize=(20, 25))
    counter = 0

    for name, group in cleaned_df.groupby(['job']):
        elements, counts = np.unique(group['education'].dropna(), return_counts=True)
        temp = cleaned_df[cleaned_df['job'] == name]['education'].fillna(elements[counts.argmax()])
        for i in temp.index:
            cleaned_df.loc[i, 'education'] = temp.loc[i]

        trace_x = counter // 3
        trace_y = counter % 3
        x_pos = np.arange(0, len(elements))

        axs[trace_x, trace_y].bar(x_pos, counts, tick_label=elements)
        axs[trace_x, trace_y].set_title(name)

        for tick in axs[trace_x, trace_y].get_xticklabels():
            tick.set_rotation(45)

        counter += 1

    plt.show()

    '''Contact Columns'''
    # Grouped by poutcome and filled with the lowest value
    # Graphs -> contact frequency by poutcome
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(15, 15))
    counter = 0

    for name, group in cleaned_df.groupby(['poutcome']):
        elements, counts = np.unique(group['contact'].dropna(), return_counts=True)
        temp = cleaned_df[cleaned_df['poutcome'] == name]['contact'].fillna(elements[counts.argmax()])
        for i in temp.index:
            cleaned_df.loc[i, 'contact'] = temp.loc[i]

        trace_x = counter // 2
        trace_y = counter % 2
        x_pos = np.arange(0, len(elements))

        axs[trace_x, trace_y].bar(x_pos, counts, tick_label=elements)
        axs[trace_x, trace_y].set_title(name)

        for tick in axs[trace_x, trace_y].get_xticklabels():
            tick.set_rotation(45)

        counter += 1

    plt.show()

    return cleaned_df


def cleaningData(df):
    # If there are more than five missing values in a row, it is judged that it cannot be used and dropped.
    cleaned_df = df.dropna(axis=0, thresh=13)

    '''
    Two cleaning missing data
    1. Numerical column filled with median value based on target column
    The categorical column is filled with the mode value based on the target column.
    2. View the correction heatmap and fill it with the highest value by column.
    Numerical column is filled with median value or predicted value using linear regression.
    Categorical column is filled with the mode value.
    '''
    # cleaned_df = cleanMissingData1(cleaned_df)
    cleaned_df = cleanMissingData2(cleaned_df)

    cleaned_df = detectOutlier(cleaned_df)

    return cleaned_df


def visualizeCatCol(df):
    # Categorical columns exploration
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 15))

    counter = 0
    for cat_column in cat_columns:
        value_counts = df[cat_column].value_counts()

        trace_x = counter // 3
        trace_y = counter % 3
        x_pos = np.arange(0, len(value_counts))

        axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label=value_counts.index, alpha=0.8)

        axs[trace_x, trace_y].set_title(cat_column)

        for tick in axs[trace_x, trace_y].get_xticklabels():
            tick.set_rotation(45)

        counter += 1

    plt.show()


def visualizeNumCol(df):
    # Numerical columns exploration
    num_columns = ['balance', 'day', 'duration', 'campaign', 'previous', 'pdays']

    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 15))

    counter = 0

    for num_column in num_columns:
        trace_x = counter // 3
        trace_y = counter % 3
        data = df[num_column]
        sns.distplot(data, ax=axs[trace_x, trace_y])
        axs[trace_x, trace_y].set_title(num_column, fontsize=15)

        counter += 1

    plt.show()


def scaling(X_train):
    sc = MinMaxScaler()
    X_train2 = pd.DataFrame(sc.fit_transform(X_train))
    X_train2.columns = X_train.columns.values
    X_train2.index = X_train.index.values
    X_train = X_train2

    return X_train


def findImportantFeat(X, y):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()


# It is very important to look at the response column, which holds the information, which we are going to predict. In
# our case we should look at 'deposit' column and compare its values to other columns. First of all we should look at
# the number of 'yes' and 'no' values in the response column 'deposit'.
def targetValueInfo(df):
    print('target value count')
    print(df['deposit'].value_counts())
    print('\n\n')

    countNo = len(df[df.deposit == 'no'])
    countYes = len(df[df.deposit == 'yes'])
    print('target value percentage')
    print('Percentage of "No": {:.3f}%'.format((countNo / (len(df.deposit)) * 100)))
    print('Percentage of "Yes": {:.3f}%'.format((countYes / (len(df.deposit)) * 100)))
    print('\n\n')

    color_list = ['salmon', 'darkslateblue']
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    df['deposit'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True, colors=color_list)
    ax[0].set_title('Deposit value percentage')
    ax[0].set_ylabel('')
    value_counts = df['deposit'].value_counts()
    value_counts.plot.bar(title='Deposit value counts', ax=ax[1], color=color_list)

    plt.show()


# Let's see how 'deposit' column value varies depending on other categorical columns' values:
def visualizeCatColAndTarget(df):
    color_list = ['salmon', 'darkslateblue']

    # job and deposit
    j_df = pd.DataFrame()

    # '블루칼라'와 '서비스' 직업을 가진 고객은 정기예금을 가입할 가능성이 적다.
    j_df['yes'] = df[df['deposit'] == 'yes']['job'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['job'].value_counts()

    j_df.plot.bar(title='Job and deposit', color=color_list)

    plt.show()

    # marital status and deposit
    j_df = pd.DataFrame()

    # 기혼 고객은 정기예금을 가입할 가능성이 적다.
    j_df['yes'] = df[df['deposit'] == 'yes']['marital'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['marital'].value_counts()

    j_df.plot.bar(title='Marital status and deposit', color=color_list)

    plt.show()

    # education and deposit
    j_df = pd.DataFrame()

    j_df['yes'] = df[df['deposit'] == 'yes']['education'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['education'].value_counts()

    j_df.plot.bar(title='Education and deposit', color=color_list)

    plt.show()

    # default and deposit
    j_df = pd.DataFrame()

    j_df['yes'] = df[df['deposit'] == 'yes']['default'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['default'].value_counts()

    j_df.plot.bar(title='Default and deposit', color=color_list)

    plt.show()

    # housing and deposit
    j_df = pd.DataFrame()

    j_df['yes'] = df[df['deposit'] == 'yes']['housing'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['housing'].value_counts()

    j_df.plot.bar(title='Housing and deposit', color=color_list)

    plt.show()

    # loan and deposit
    j_df = pd.DataFrame()

    j_df['yes'] = df[df['deposit'] == 'yes']['loan'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['loan'].value_counts()

    j_df.plot.bar(title='Loan and deposit', color=color_list)

    plt.show()

    # contact and deposit
    j_df = pd.DataFrame()

    j_df['yes'] = df[df['deposit'] == 'yes']['contact'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['contact'].value_counts()

    j_df.plot.bar(title='Type of contact and deposit', color=color_list)

    plt.show()

    # month and deposit
    j_df = pd.DataFrame()

    j_df['yes'] = df[df['deposit'] == 'yes']['month'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['month'].value_counts()

    j_df.plot.bar(title='Month and deposit', color=color_list)

    plt.show()

    # poutcome and deposit
    j_df = pd.DataFrame()

    j_df['yes'] = df[df['deposit'] == 'yes']['poutcome'].value_counts()
    j_df['no'] = df[df['deposit'] == 'no']['poutcome'].value_counts()

    j_df.plot.bar(title='Previous outcome and deposit', color=color_list)

    plt.show()


def get_correct_values(row, column_name, threshold, df):
    ''' Returns mean value if value in column_name is above threshold'''
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].median()
        return mean


def detectOutlier(df):
    """
    We can see that numerical columns have outliers (especially 'pdays', 'campaign' and 'previous' columns).
    Possibly there are incorrect values (noisy data),
    so we should look closer at the data and decide how do we manage the noise.
    Let's look closer at the values of 'campaign', 'pdays' and 'previous' columns:'
    """
    print(df[['pdays', 'campaign', 'previous']].describe())
    print('\n\n')

    '''
    -1 possibly means that the client wasn't contacted before or stands for missing data. 
    Since we are not sure exactly what -1 means I suggest to drop this column, 
    because -1 makes more than 70 % of the values of the column.
    '''
    sns.boxplot(df.pdays)
    plt.show()
    dataWOP = df[df.pdays != -1]
    sns.boxplot(dataWOP.pdays)
    plt.show()
    print("Percentage of -1 of 'pdays' column: ", len(df[df['pdays'] == -1]) / len(df) * 100)
    print('\n')
    print("Percentage of 0 of 'previous' column: ", len(df[df['previous'] == 0]) / len(df) * 100)
    print('\n')

    print("Percentage of above 40 of 'campaign' column: ", len(df[df['campaign'] > 40]) / len(df) * 100)
    print('\n')
    print("Percentage of above 35 of 'previous' column: ", len(df[df['previous'] > 35]) / len(df) * 100)
    print('\n')

    sns.boxplot(df.campaign)
    plt.show()
    '''
    'campaign' holds the number of contacts performed during this campaign and for this client(numeric, includes 
    last contact) Numbers for 'campaign' above 35 are clearly noise, 
    so I suggest to impute them with average campaign values while data cleaning
    '''

    sns.boxplot(df.previous)
    plt.show()
    '''
    previous ' holds the number of contacts performed before this campaign and for this client (numeric) Numbers for 
    'previous' above 40 are also really strange, 
    so I suggest to impute them with average campaign values while data cleaning.
    '''

    # # drop irrelevant columns
    cleaned_df = df.drop(columns=['pdays'])

    # impute incorrect values and drop original columns
    cleaned_df['campaign'] = df.apply(lambda row: get_correct_values(row, 'campaign', 40, cleaned_df), axis=1)
    cleaned_df['previous'] = df.apply(lambda row: get_correct_values(row, 'previous', 35, cleaned_df), axis=1)

    return cleaned_df


def get_dummy_from_bool(row, column_name):
    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''
    return 1 if row[column_name] == 'yes' else 0


def encoder(df):
    cleaned_df = df.copy()
    features = ['job', 'marital', 'education', 'contact', 'poutcome', 'housing', 'loan', 'default', 'month', 'deposit']
    for feature in features:
        cleaned_df[feature] = preprocessing.LabelEncoder().fit_transform(cleaned_df[feature])
    # convert columns containing 'yes' and 'no' values to boolean variables and drop original columns

    return cleaned_df


def correlationWithTarget(df):
    plt.figure(figsize=(20, 10))
    sns.heatmap(data=df.corr(), annot=True, cmap='viridis')
    plt.show()


def predictResult(y_test, y_pred, title, results):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_results = pd.DataFrame([[title, acc, prec, rec, f1]],
                                 columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    results = results.append(model_results, ignore_index=True)

    return results


def predictCVResult(X, y, classifier, title, cvResults):
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    accuracies = cross_val_score(classifier, X, y, scoring="accuracy", cv=cv)

    model_results = pd.DataFrame([[title, accuracies.mean()]], columns=['Model', 'Mean accuracy'])

    cvResults = cvResults.append(model_results, ignore_index=True)

    return cvResults


def predictBaggingResult(X_train, X_test, y_train, y_test, classifier, title, baggingResults):
    bc = BaggingClassifier(base_estimator=classifier).fit(X_train, y_train)
    y_pred = bc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_results = pd.DataFrame([[title, acc, prec, rec, f1]],
                                 columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    baggingResults = baggingResults.append(model_results, ignore_index=True)

    title = title + ' (Bagging method) Confusion Matrix'
    confusionMatrix(y_test, y_pred, title)

    return baggingResults


def confusionMatrix(y_test, y_pred, title):
    conf_matrix = confusion_matrix(y_test, y_pred)
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.title(title, fontsize=20)
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels("")
    ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)
    plt.show()


def trainDecisionTree(X, y, results, cvResults, baggingResults):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''Default parameter'''
    classifier1 = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier1.fit(X_train, y_train)
    # Predicting the best set result
    y_pred = classifier1.predict(X_test)

    compare = pd.DataFrame()
    compare = predictResult(y_test, y_pred, 'Decision Tree (default = None)', compare)

    '''Tuning hyper-parameter using GridSearchCV'''
    classifier2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
    param_grid = {'max_depth': np.arange(1, 30)}

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    dt_gscv = GridSearchCV(classifier2, param_grid, cv=cv)
    dt_gscv.fit(X, y)
    bestParams = dt_gscv.best_params_
    bestEstimator = dt_gscv.best_estimator_

    # Decision tree classifier tuned parameter
    classifier2 = DecisionTreeClassifier(max_depth=bestParams['max_depth'])
    classifier2.fit(X_train, y_train)

    # Predicting the best set result
    y_pred = classifier2.predict(X_test)

    # Compare default parameter to tuned parameter
    compare = predictResult(y_test, y_pred, 'Decision Tree (Best max depth)', compare)

    # tuned parameter
    results = predictResult(y_test, y_pred, 'Decision Tree', results)
    confusionMatrix(y_test, y_pred, "Decision Tree Confusion Matrix")

    cvResults = predictCVResult(X, y, classifier2, 'Decision Tree', cvResults)
    baggingResults = predictBaggingResult(X_train, X_test, y_train, y_test, bestEstimator, 'Decision Tree',
                                          baggingResults)

    print('========== Holdout method(Decision Tree) ==========')
    print('Best Parameter: ', bestParams)
    print(compare)
    print('\n')

    return results, cvResults, baggingResults, bestEstimator


def trainKNN(X, y, results, cvResults, baggingResults):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''Default parameter'''
    classifier1 = KNeighborsClassifier()
    classifier1.fit(X_train, y_train)
    # Predicting the best set result
    y_pred = classifier1.predict(X_test)

    compare = pd.DataFrame()
    compare = predictResult(y_test, y_pred, 'K-Nearest Neighbors (default = 5)', compare)

    '''Tuning hyper-parameter using GridSearchCV'''
    classifier2 = KNeighborsClassifier(n_jobs=-1)
    param_grid = {'n_neighbors': np.arange(1, 20, 2)}

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    knn_gscv = GridSearchCV(classifier2, param_grid, cv=cv)
    knn_gscv.fit(X, y)
    bestParams = knn_gscv.best_params_
    bestEstimator = knn_gscv.best_estimator_

    # KNN classifier tuned parameter
    classifier2 = KNeighborsClassifier(n_neighbors=bestParams['n_neighbors'])
    classifier2.fit(X_train, y_train)

    # Predicting the best set result
    y_pred = classifier2.predict(X_test)

    # Compare default parameter to tuned parameter
    compare = predictResult(y_test, y_pred, 'K-Nearest Neighbors (Best k)', compare)

    # tuned parameter
    results = predictResult(y_test, y_pred, 'K-Nearest Neighbors', results)
    confusionMatrix(y_test, y_pred, "K-Nearest Neighbors Confusion Matrix")

    cvResults = predictCVResult(X, y, classifier2, 'K-Nearest Neighbors', cvResults)
    # baggingResults = predictBaggingResult(X_train, X_test, y_train, y_test, bestEstimator, 'K-Nearest Neighbors',
    #                                       baggingResults)

    print('========== Holdout method(K-Nearest Neighbors) ==========')
    print('Best Parameter: ', bestParams)
    print(compare)
    print('\n')

    return results, cvResults, baggingResults, bestEstimator


def trainRandomForest(X, y, results, cvResults, baggingResults):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''Default parameter'''
    classifier1 = RandomForestClassifier(random_state=0, criterion='entropy', bootstrap=True,
                                         n_jobs=-1)
    classifier1.fit(X_train, y_train)
    # Predicting the best set result
    y_pred = classifier1.predict(X_test)

    compare = pd.DataFrame()
    compare = predictResult(y_test, y_pred, 'Random Forest (default = 100)', compare)

    '''Tuning hyper-parameter using GridSearchCV'''
    classifier2 = RandomForestClassifier(random_state=0, criterion='entropy', bootstrap=True,
                                         n_jobs=-1)
    param_grid = {'n_estimators': [100, 200],
                  'max_depth': [6, 8, 10]}

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    rf_gscv = GridSearchCV(classifier2, param_grid, cv=cv)
    rf_gscv.fit(X, y)
    bestParams = rf_gscv.best_params_
    bestEstimator = rf_gscv.best_estimator_

    # RandomForest classifier tuned parameter
    classifier2 = RandomForestClassifier(n_estimators=bestParams['n_estimators'], max_depth=bestParams['max_depth'])

    classifier2.fit(X_train, y_train)

    # Predicting the best set result
    y_pred = classifier2.predict(X_test)

    # Compare default parameter to tuned parameter
    compare = predictResult(y_test, y_pred, 'Random Forest (best parameters)', compare)

    # tuned parameter
    results = predictResult(y_test, y_pred, 'Random Forest', results)
    confusionMatrix(y_test, y_pred, "Random Forest Confusion Matrix")

    cvResults = predictCVResult(X, y, classifier2, 'Random Forest', cvResults)
    baggingResults = predictBaggingResult(X_train, X_test, y_train, y_test, bestEstimator, 'Random Forest',
                                          baggingResults)

    print('==========  Holdout method(Random Forest) ==========')
    print('Best Parameter: ', bestParams)
    print(compare)
    print('\n')

    return results, cvResults, baggingResults, bestEstimator


def trainXGBoost(X, y, results, cvResults, baggingResults):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''Default parameter'''
    classifier1 = xgb.XGBClassifier()
    classifier1.fit(X_train, y_train.squeeze().values)
    # Predicting the best set result
    y_pred = classifier1.predict(X_test)

    compare = pd.DataFrame()
    compare = predictResult(y_test, y_pred, 'XGBoost (default)', compare)

    '''Tuning hyper-parameter using GridSearchCV'''
    classifier2 = xgb.XGBClassifier()
    param_grid = {'min_child_weight': [1],
                  'n_estimators': [100, 200, 300],
                  'max_depth': [6, 8, 10]}

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    xgb_gscv = GridSearchCV(classifier2, param_grid, cv=cv)
    xgb_gscv.fit(X, y)
    bestParams = xgb_gscv.best_params_
    bestEstimator = xgb_gscv.best_estimator_

    # XGBoost classifier tuned parameter
    classifier2 = xgb.XGBClassifier(min_child_weight=bestParams['min_child_weight'],
                                    max_depth=bestParams['max_depth'],
                                    n_estimators=bestParams['n_estimators'])
    classifier2.fit(X_train, y_train)

    # Predicting the best set result
    y_pred = classifier2.predict(X_test)

    # Compare default parameter to tuned parameter
    compare = predictResult(y_test, y_pred, 'XGBoost (best parameters)', compare)

    # tuned parameter
    results = predictResult(y_test, y_pred, 'XGBoost', results)
    confusionMatrix(y_test, y_pred, "XGBoost Confusion Matrix")

    cvResults = predictCVResult(X, y, classifier2, 'XGBoost', cvResults)

    print('==========  Holdout method(XGBoost) ==========')
    print('Best Parameter: ', bestParams)
    print(compare)
    print('\n')

    # Find important features
    ftr_importances_values = classifier2.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
    ftr_top = ftr_importances.sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    plt.title('Feature Importances (XGBoost)')
    sns.barplot(x=ftr_top, y=ftr_top.index)
    plt.show()

    return results, cvResults, baggingResults, bestEstimator


def majorityVoting(X, y, bestEstimator):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vc = VotingClassifier(estimators=[('knn', bestEstimator[0]),
                                      ('dt', bestEstimator[1]),
                                      ('xgb', bestEstimator[2])], voting='hard').fit(X_train, y_train)

    y_pred = vc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    voting_results = pd.DataFrame([[acc, prec, rec, f1]],
                                  columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    return voting_results


def outcome(df):
    '''
    Customer's account balance,
    Customer's age,
    Number of contacts performed during this campaign and contact duration,
    Number of contacts performed before this campaign.
    '''
    df_new = df.copy()

    '''poutcome'''
    # group by 'poutcome' and find average campaign outcome per past outcome result
    mean_deposit = df_new.groupby(['poutcome'])['deposit'].mean()
    index = ['failure', 'other', 'success', 'unknown']

    # plot
    plt.bar(index, mean_deposit.values)
    plt.title('Mean % subscription depending on past outcome result')
    plt.xlabel('past outcome')
    plt.ylabel('% subscription')
    plt.show()

    '''previous'''
    # introduce new column 'previous_buckets' to  ''
    df_new['previous_buckets'] = pd.qcut(df_new['previous'], 20, labels=False, duplicates='drop')

    # group by 'previous_buckets' and find average campaign outcome per number of previous contacts bucket
    mean_deposit = df_new.groupby(['previous_buckets'])['deposit'].mean()

    # plot
    plt.plot(mean_deposit.index, mean_deposit.values)
    plt.title('Mean % subscription depending on number of previous contacts')
    plt.xlabel('previous bucket')
    plt.ylabel('% subscription')
    plt.show()

    '''duration'''
    # introduce new column 'duration_buckets' to  ''
    df_new['duration_buckets'] = pd.qcut(df_new['duration'], 20, labels=False, duplicates='drop')

    # group by 'duration_buckets' and find average campaign outcome per duration bucket
    mean_deposit = df_new.groupby(['duration_buckets'])['deposit'].mean()

    # plot
    plt.plot(mean_deposit.index, mean_deposit.values)
    plt.title('Mean % subscription depending on current contact duration')
    plt.xlabel('duration bucket')
    plt.ylabel('% subscription')
    plt.show()

    '''campaign'''
    # introduce new column 'age_buckets' to  ''
    df_new['campaign_buckets'] = pd.qcut(df_new['campaign'], 20, labels=False, duplicates='drop')

    # group by 'balance_buckets' and find average campaign outcome per balance bucket
    mean_campaign = df_new.groupby(['campaign_buckets'])['deposit'].mean()

    # plot average campaign outcome per bucket
    plt.plot(mean_campaign.index, mean_campaign.values)
    plt.title('Mean % subscription depending on number of contacts')
    plt.xlabel('number of contacts bucket')
    plt.ylabel('% subscription')
    plt.show()


''' Data Inspection '''
df = readDataset()
datasetInfo(df)
statisticalInfo(df)
featureInfo(df)
targetValueInfo(df)
visualizeCatCol(df.dropna())
visualizeNumCol(df.dropna())
'''
On the diagram we see that counts for 'yes' and 'no' values for 'deposit' are close,
so we can use accuracy as a metric for a model, which predicts the campaign outcome.
'''

''' Data Preprocessing '''
replaced_df = replaceUnusableData(df)
replaced_df = replaceWrongData(replaced_df)
verifyMissingValue(replaced_df)

cleaned_df = cleaningData(replaced_df)
processed_df = encoder(cleaned_df)

''' Data Analysis & Evaluation '''
X = processed_df.drop(columns='deposit')
y = processed_df['deposit']
X = scaling(X)

# holdout method
results = pd.DataFrame()
# k-fold cross validation
cvResults = pd.DataFrame()
# bagging method
baggingResults = pd.DataFrame()
# majority voting
model_results = pd.DataFrame()
bestEstimator = []
results, cvResults, baggingResults, bestEstimator1 = trainKNN(X, y, results, cvResults, baggingResults)
results, cvResults, baggingResults, bestEstimator2 = trainDecisionTree(X, y, results, cvResults, baggingResults)
# results, cvResults, baggingResults, bestEstimator3 = trainRandomForest(X, y, results, cvResults, baggingResults)
results, cvResults, baggingResults, bestEstimator4 = trainXGBoost(X, y, results, cvResults, baggingResults)

bestEstimator.append(bestEstimator1)
bestEstimator.append(bestEstimator2)
# bestEstimator.append(bestEstimator3)
bestEstimator.append(bestEstimator4)

model_results = majorityVoting(X, y, bestEstimator)

print('========== Holdout method(Parameter tuning) ==========')
print(results)
print('\n')
print('========== Cross validation ==========')
print(cvResults)
print('\n')
print('========== Bagging Method ==========')
print(baggingResults)
print('\n')
print('========== Majority voting ==========')
print(model_results)

outcome(processed_df)
