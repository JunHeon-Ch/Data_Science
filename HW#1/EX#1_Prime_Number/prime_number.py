import math


def is_prime(n):
    num = math.floor(math.sqrt(n)) + 1

    for i in range(2, num):
        if n % i == 0:
            print("{} isn't prime.".format(n))
            print('{} is divided by {}'.format(n, i))
            return i
    print("{} is prime.".format(n))


n = int(input())
is_prime(n)