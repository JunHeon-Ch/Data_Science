import numpy as np


def make_bmi():
    wt = np.random.uniform(40.0, 90.0, 100)
    ht = np.random.randint(140, 200, 100)

    bmi = np.zeros(100)
    for i in range(len(bmi)):
        bmi[i] = wt[i] / np.power(ht[i] * 0.01, 2)

    return bmi


bmi = make_bmi()

print(bmi)