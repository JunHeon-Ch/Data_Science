import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

df = pd.concat([train, test])

'''a'''
df_AF = df.loc[:, ('Age', 'Fare')]
# Standard Scaler
scaler_s = preprocessing.StandardScaler()
scaled_s = scaler_s.fit_transform(df_AF)
scaled_s = pd.DataFrame(scaled_s, columns=['Age', 'Fare'])
print('----------Standard Scaler----------')
print(scaled_s.head(n=20))
# MinMax Scaler
scaler_m = preprocessing.MinMaxScaler()
scaled_m = scaler_m.fit_transform(df_AF)
scaled_m = pd.DataFrame(scaled_m, columns=['Age', 'Fare'])
print('----------MinMax Scaler----------')
print(scaled_m.head(n=20))
# Robust Scaler
scaler_r = preprocessing.RobustScaler()
scaled_r = scaler_r.fit_transform(df_AF)
scaled_r = pd.DataFrame(scaled_r, columns=['Age', 'Fare'])
print('----------Robust Scaler----------')
print(scaled_r.head(n=20))

'''b'''
enc = preprocessing.OrdinalEncoder()
X = df.loc[:, ('Sex', 'Embarked', 'SibSp', 'Parch')]
X = X.dropna(how='any')
elements, counts = np.unique(X['Sex'], return_counts=True)
print(elements)
elements, counts = np.unique(X['Embarked'], return_counts=True)
print(elements)
elements, counts = np.unique(X['SibSp'], return_counts=True)
print(elements)
elements, counts = np.unique(X['Parch'], return_counts=True)
print(elements)

enc.fit(X)
Ord = pd.DataFrame(enc.transform(X), columns=['Sex', 'Embarked', 'SibSp', 'Parch'])
print(Ord.head(20))

# Using OneHotEncoder
enc2 = preprocessing.OneHotEncoder()
enc2.fit(X)
One = pd.DataFrame(enc2.transform(X).toarray())
print(One.head(20))

'''c'''
# # Comb_1(Standard_Ordinal)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# train_sOr = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# train_sOr.loc[:, ('Age', 'Fare')] = scaled_s
# train_sOr.loc[:, ('Sex', 'Embarked', 'SibSp', 'Parch')] = Ord
# train_sOr.loc[:, 'Age'] = train_sOr.loc[:, 'Age'] + 2.287647990657291  # must be non-neagtive
# train_sOr.loc[:, 'Fare'] = train_sOr.loc[:, 'Fare'] + 0.6438090896418941
# train_sOr = train_sOr.dropna()
#
# X = train_sOr.iloc[:, 2:9]
# y = train_sOr.loc[:, 'Survived']
# bestfeatures = SelectKBest(score_func=chi2, k=5)
# fit = bestfeatures.fit(X, y)
# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)
#
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Feature', 'Score']
#
# print('----------Comb_1(Standard_Ordinal)----------\n')
# print('----------Use SelectKBest----------')
# print(featureScores.nlargest(3, 'Score'), '\n\n')
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X, y)
#
# print('----------Use ExtraTreesClassifier----------')
# print('feature_importances :', model.feature_importances_)  # feature_importances of tree-based classifiers
# # plot graph of feature importances
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(3).plot(kind='barh')
# plt.show()
#
# print('----------Use ExtraTreesClassifier----------')
# import seaborn as sns
#
# cormat = train_sOr.corr()  # corr() computes pairwise
# top_corr_features = cormat.index
# plt.figure(figsize=(8, 8))
# # plot the heat map
# g = sns.heatmap(train_sOr.corr(), annot=True, cmap='RdYlGn')

# # Comb_2(Standrd_OneHot)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# train_sOn = pd.get_dummies(train, columns=['Sex', 'Embarked', 'SibSp', 'Parch'],
#                            prefix=('Sex', 'Embarked', 'SibSp', 'Parch'))
# train_sOn = train_sOn.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# train_sOn.loc[:, ('Age', 'Fare')] = scaled_s
# train_sOn.loc[:, 'Age'] = train_sOn.loc[:, 'Age'] + 2.287647990657291  # must be non-negative
# train_sOn.loc[:, 'Fare'] = train_sOn.loc[:, 'Fare'] + 0.6438090896418941
# train_sOn
#
# X = train_sOn.iloc[:, 2:23]
# y = train_sOn.loc[:, 'Survived']
# bestfeatures = SelectKBest(score_func=chi2, k=5)
# fit = bestfeatures.fit(X, y)
# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)
#
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Feature', 'Score']
#
# print('----------Comb_2(Standard_OneHot)----------\n')
# print('----------Use SelectKBest----------')
# print(featureScores.nlargest(3, 'Score'), '\n\n')
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X, y)
#
# print('----------Use ExtraTreesClassifier----------')
# print('feature_importances :', model.feature_importances_)  # feature_importances of tree-based classifiers
# # plot graph of feature importances
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(3).plot(kind='barh')
# plt.show()
#
# print('----------Use ExtraTreesClassifier----------')
# import seaborn as sns
#
# cormat = train_sOn.corr()  # corr() computes pairwise
# top_corr_features = cormat.index
# plt.figure(figsize=(15, 15))
# # plot the heat map
# g = sns.heatmap(train_sOn[top_corr_features].corr(), annot=True, cmap='RdYlGn')
#
# # Comb_3(MinMax_Ordinal)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# train_mOr = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# train_mOr.loc[:, ('Age', 'Fare')] = scaled_m
# train_mOr.loc[:, ('Sex', 'Embarked', 'SibSp', 'Parch')] = Ord
#
# X = train_mOr.iloc[:, 2:9]
# y = train_mOr.loc[:, 'Survived']
# bestfeatures = SelectKBest(score_func=chi2, k=5)
# fit = bestfeatures.fit(X, y)
# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)
#
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Feature', 'Score']
#
# print('----------Comb_1(Standard_Ordinal)----------\n')
# print('----------Use SelectKBest----------')
# print(featureScores.nlargest(3, 'Score'), '\n\n')
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X, y)
#
# print('----------Use ExtraTreesClassifier----------')
# print('feature_importances :', model.feature_importances_)  # feature_importances of tree-based classifiers
# # plot graph of feature importances
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(3).plot(kind='barh')
# plt.show()
#
# print('----------Use ExtraTreesClassifier----------')
# import seaborn as sns
#
# cormat = train_mOr.corr()  # corr() computes pairwise
# top_corr_features = cormat.index
# plt.figure(figsize=(8, 8))
# # plot the heat map
# g = sns.heatmap(train_mOr[top_corr_features].corr(), annot=True, cmap='RdYlGn')
#
# # Comb_4(MinMax_OneHot)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# train_mOn = pd.get_dummies(train, columns=['Sex', 'Embarked', 'SibSp', 'Parch'],
#                            prefix=('Sex', 'Embarked', 'SibSp', 'Parch'))
# train_mOn = train_mOn.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# train_mOn.loc[:, ('Age', 'Fare')] = scaled_m
# train_mOn
#
# X = train_mOn.iloc[:, 2:23]
# y = train_mOn.loc[:, 'Survived']
# bestfeatures = SelectKBest(score_func=chi2, k=5)
# fit = bestfeatures.fit(X, y)
# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)
#
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Feature', 'Score']
#
# print('----------Comb_2(Standard_OneHot)----------\n')
# print('----------Use SelectKBest----------')
# print(featureScores.nlargest(3, 'Score'), '\n\n')
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X, y)
#
# print('----------Use ExtraTreesClassifier----------')
# print('feature_importances :', model.feature_importances_)  # feature_importances of tree-based classifiers
# # plot graph of feature importances
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(3).plot(kind='barh')
# plt.show()
#
# print('----------Use ExtraTreesClassifier----------')
# import seaborn as sns
#
# cormat = train_mOn.corr()  # corr() computes pairwise
# top_corr_features = cormat.index
# plt.figure(figsize=(15, 15))
# # plot the heat map
# g = sns.heatmap(train_mOn[top_corr_features].corr(), annot=True, cmap='RdYlGn')
#
# # Comb_4(MinMax_OneHot)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# train_mOn = pd.get_dummies(train, columns=['Sex', 'Embarked', 'SibSp', 'Parch'],
#                            prefix=('Sex', 'Embarked', 'SibSp', 'Parch'))
# train_mOn = train_mOn.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# train_mOn.loc[:, ('Age', 'Fare')] = scaled_m
# train_mOn
#
# X = train_mOn.iloc[:, 2:23]
# y = train_mOn.loc[:, 'Survived']
# bestfeatures = SelectKBest(score_func=chi2, k=5)
# fit = bestfeatures.fit(X, y)
# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)
#
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Feature', 'Score']
#
# print('----------Comb_2(Standard_OneHot)----------\n')
# print('----------Use SelectKBest----------')
# print(featureScores.nlargest(3, 'Score'), '\n\n')
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X, y)
#
# print('----------Use ExtraTreesClassifier----------')
# print('feature_importances :', model.feature_importances_)  # feature_importances of tree-based classifiers
# # plot graph of feature importances
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(3).plot(kind='barh')
# plt.show()
#
# print('----------Use ExtraTreesClassifier----------')
# import seaborn as sns
#
# cormat = train_mOn.corr()  # corr() computes pairwise
# top_corr_features = cormat.index
# plt.figure(figsize=(15, 15))
# # plot the heat map
# g = sns.heatmap(train_mOn[top_corr_features].corr(), annot=True, cmap='RdYlGn')
#
# # Comb_6(Robust_OneHot)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# train_rOn = pd.get_dummies(train, columns=['Sex', 'Embarked', 'SibSp', 'Parch'],
#                            prefix=('Sex', 'Embarked', 'SibSp', 'Parch'))
# train_rOn = train_rOn.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# train_rOn.loc[:, ('Age', 'Fare')] = scaled_r
# train_rOn.loc[:, 'Age'] = train_rOn.loc[:, 'Age'] + 2.2522398190045245  # must be non-neagtive
# train_rOn.loc[:, 'Fare'] = train_rOn.loc[:, 'Fare'] + 0.6182504106214072
#
# X = train_rOn.iloc[:, 2:23]
# y = train_rOn.loc[:, 'Survived']
# bestfeatures = SelectKBest(score_func=chi2, k=5)
# fit = bestfeatures.fit(X, y)
# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)
#
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Feature', 'Score']
#
# print('----------Comb_2(Standard_OneHot)----------\n')
# print('----------Use SelectKBest----------')
# print(featureScores.nlargest(3, 'Score'), '\n\n')
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X, y)
#
# print('----------Use ExtraTreesClassifier----------')
# print('feature_importances :', model.feature_importances_)  # feature_importances of tree-based classifiers
# # plot graph of feature importances
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(3).plot(kind='barh')
# plt.show()
#
# print('----------Use ExtraTreesClassifier----------')
# import seaborn as sns
#
# cormat = train_rOn.corr()  # corr() computes pairwise
# top_corr_features = cormat.index
# plt.figure(figsize=(15, 15))
# # plot the heat map
# g = sns.heatmap(train_rOn[top_corr_features].corr(), annot=True, cmap='RdYlGn')

'''d'''
from sklearn.preprocessing import StandardScaler

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# Separate out the features
train_o = train
train_o.loc[:, ('Sex', 'Embarked', 'SibSp', 'Parch')] = Ord
train_o = train_o.dropna(how='any')
x = train_o.loc[:, features].values
# Separate out the target
y = train_o.loc[:, ['Survived']].values
# Standardize the features
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, train_o[['Survived']]], axis=1)

# code to plot the graph
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
Survived = [0, 1]
colors = ['r', 'b']
for survive, color in zip(Survived, colors):
    indicesToKeep = finalDf['Survived'] == survive
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color, s=50)
ax.legend(Survived)
ax.grid()
plt.show()
