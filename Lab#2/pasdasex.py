import pandas as pd
import numpy as np

# Create a DataFrame (4,4)
df = pd.DataFrame({'a':[3., '?', 2., 5.],
                   'b':['*', 4., 5., 6.],
                   'c':['+', 3., 2., '&'],
                   'd':[5., '?', 7., '!']})
# Display DataFrame
print(df)

# Replace any non-numeric value with NaN
df.replace({'?':np.nan, '*':np.nan, '+':np.nan, '&':np.nan, '!':np.nan}, inplace=True)
# print('Replace =============================================')
# print(df)
#
# # using isna function with any
# print('isna with any =============================================')
# print(df.isna().any())
#
# # using isna function with sum
# print('isna with sum =============================================')
# print(df.isna().sum())
#
# # using dropna function with how 'all'
# # axis is row
# print('dropna function')
# print('how=all, axis=0 =============================================')
# print(df.dropna(axis=0, how='all'))
# # axis is column
# print('how=all, axis=1 =============================================')
# print(df.dropna(axis=1, how='all'))
#
#
# # using dropna function with how 'any'
# # axis is row
# print('how=any, axis=0 =============================================')
# print(df.dropna(axis=0, how='any'))
# # axis is column
# print('how=any, axis=1 =============================================')
# print(df.dropna(axis=1, how='any'))
#
# # using dropna function with thresh=1
# # axis is row
# print('thresh = 1, axis 0 =============================================')
# print(df.dropna(axis=0, thresh=1))
# # axis is column
# print('thresh = 1, axis 1 =============================================')
# print(df.dropna(axis=1, thresh=1))
#
# # using dropna function with thresh=2
# # axis is row
# print('thresh = 2, axis 0 =============================================')
# print(df.dropna(axis=0, thresh=2))
# # axis is column
# print('thresh = 2, axis 1 =============================================')
# print(df.dropna(axis=1, thresh=2))
#
# # using fillna function
# # fill NaN with 100
# print('fillna function')
# print('100 =============================================')
# print(df.fillna(100))

# fill NaN with mean of column
print('mean =============================================')
# mean = df.mean()
# print(mean)
# print(df.fillna(df.mean()))
# print(df.fillna(df.mean(axis=0)))
mean = df.mean(axis=1)
print(df.fillna(mean))
# print(df.fillna(axis=1, value=df.mean(axis=1, skipna=True)))

# # fill NaN with median of column
# print('median =============================================')
# median = df.median()
# print(df.fillna(median))
#
# # fill NaN with forward value
# # axis is row
# print('method = ffill, axis 0 =============================================')
# print(df.fillna(axis=0, method='ffill').fillna(axis=1, method='bfill'))
# # axis is column
# print('method = ffill, axis 1 =============================================')
# print(df.fillna(axis=1, method='ffill').fillna(axis=0, method='bfill'))
#
# # fill NaN with backward value
# # axis is row
# print('method = bfill, axis 0 =============================================')
# print(df.fillna(axis=0, method='bfill').fillna(axis=1, method='ffill'))
# # axis is column
# print('method = bfill, axis 1 =============================================')
# print(df.fillna(axis=1, method='bfill').fillna(axis=0, method='ffill'))
