def make_dict(k, v):
    d = dict(zip(k, v))

    return d


def check_correct(k, v, d):
    for i in range(len(v)):
        if d[k[i]] is not v[i]:
            print('incorrect')
            return
    print('correct')


k = ('Korean', 'Math', 'English', 'Science')
v = (90.1, 90.1, 96.7, 88.2)

d = make_dict(k, v)
check_correct(k, v, d)