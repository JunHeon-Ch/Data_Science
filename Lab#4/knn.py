import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def scaling(df):
    sc = StandardScaler()
    ndf = pd.DataFrame(sc.fit_transform(df))
    ndf.columns = df.columns.values
    ndf.index = df.index.values

    return ndf


def encoder(df):
    features = list(df)
    for feature in features:
        df[feature] = preprocessing.LabelEncoder().fit_transform(df[feature])


warnings.filterwarnings(action='ignore')

# Read dataset
df = pd.read_csv('knn_data.csv', encoding='utf-8')
df['longitude'] = pd.to_numeric(df['longitude'])
df['latitude'] = pd.to_numeric(df['latitude'])

# Independent feature
X = df.drop(['lang'], axis=1)
# Target feature
y = df['lang'].to_frame()
# Scaling data before using model
X = scaling(X)
# Encoding data before using model
encoder(y)

# GridSearchCV is used to find the optimal k-value of KNN
classifier = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, len(df) // 2, 2)}

cv = KFold(n_splits=5, shuffle=True, random_state=1)
knn_gscv = GridSearchCV(classifier, param_grid, cv=cv)
knn_gscv.fit(X, y)
# Find the optimal k-value
bestk = knn_gscv.best_params_['n_neighbors']
# Use a KNN with the optimal k-value
classifier = KNeighborsClassifier(bestk)

# Evaluate the model using 5-fold cross validation method
accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=cv)
print('Accuracy list')
for accuracy in accuracies:
    print(accuracy)
print("Accuracy mean: %0.3f" % accuracies.mean())