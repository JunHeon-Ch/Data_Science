import numpy as np

np.set_printoptions(precision=3, suppress=True)
age = [30, 40, 50, 60, 40]
income = [200, 300, 800, 600, 300]
yrsWorked = [10, 20, 20, 20, 20]
vacation = [4, 4, 1, 2, 5]
data = np.array([age, income, yrsWorked, vacation])
covMatrix = np.cov(data, bias=False)
print(covMatrix)
