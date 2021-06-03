import numpy as np
import matplotlib.pyplot as plt


def make_bmi(wt, ht):
    bmi = np.zeros(100)
    for i in range(len(bmi)):
        bmi[i] = wt[i] / np.power(ht[i] * 0.01, 2)

    return bmi


def number_by_bmi(bmi):
    students = [0, 0, 0, 0]

    for i in range(len(bmi)):
        if bmi[i] < 18.5:
            students[0] += 1
        elif 18.5 <= bmi[i] < 24.9:
            students[1] += 1
        elif 25.0 <= bmi[i] < 29.9:
            students[2] += 1
        else:
            students[3] += 1

    return students


def make_bar_chart(status, students):
    plt.bar(status, students)
    plt.title('student distribution of bmi level')
    plt.show()


def make_pie_chart(status, students):
    plt.pie(students, labels=status, autopct='%1.2f%%')
    plt.show()


def make_scatter_plot(wt, ht):
    plt.scatter(ht, wt)
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.show()


def make_histogram(bmi):
    plt.hist(bmi, bins=[0, 18.5, 25.0, 30.0, 50.0])
    plt.xticks([0, 18.5, 25.0, 30.0, 50.0])
    plt.title('student distribution of bmi level')
    plt.xlabel('bmi')
    plt.ylabel('student')
    plt.show()


wt = np.random.uniform(40.0, 90.0, 100)
ht = np.random.randint(140, 200, 100)
bmi = make_bmi(wt, ht)

status = ['Underweight', 'Healthy', 'Overweight', 'Obese']
students = number_by_bmi(bmi)

make_bar_chart(status, students)
make_histogram(bmi)
make_pie_chart(status, students)
make_scatter_plot(wt, ht)
