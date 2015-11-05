__author__ = 'Jmexe'
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.special import erfinv
from math import sqrt,log

def score_project(score):
    prop =  (-1) * log(score / (float)(400))

    if prop <= 1 and prop >=0:
        return (int)((1 - prop) * score)
    else:
        return score

def score_proportion(score):
    prop =  (-1) * log(score / (float)(400))

    if prop <= 1 and prop >=0:
        return 1 - prop
    else:
        return 1


def load_data(filename):
    file = open(filename)

    t_miu = 0

    data = []
    for line in file:
        split = line.strip().split("\t")

        for i in range((int)(split[1])):
            t_miu += (int)(split[0])
            data.append((int)(split[0]))

    return data, t_miu

def load_data_count(filename):
    file = open(filename)
    score = []
    cnt = []
    for line in file:
        split = line.strip().split("\t")
        score.append((int)(split[0]))
        cnt.append((int)(split[1]))
    return score, cnt

def load_partial_data(filename):
    file = open(filename)

    t_miu = 0

    data = []
    for line in file:
        split = line.strip().split("\t")
        prop = score_proportion((int)(split[0]))

        for i in range((int)(prop * (int)(split[1]))):
            t_miu += (int)(split[0])
            data.append((int)(split[0]))

    return data, t_miu

def load_data_with_projection(filename):
    file = open(filename)

    t_miu = 0

    data = []
    for line in file:
        split = line.strip().split("\t")
        nscore = score_project((int)(split[0]))

        for i in range((int)(split[1])):
            t_miu += nscore
            data.append(nscore)

    return data, t_miu


def show_data(data, miu, sigma):
    import matplotlib.pyplot as plt
    num_bins = 1000
    n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, miu, sigma)
    plt.plot(bins, y, 'r--')


    plt.xlabel('Scores')
    plt.ylabel('Probability')
    plt.title(r'Histogram of Shandong 2015 : $\mu='+str(miu)+'$, $\sigma='+str(sigma)+'$')


    plt.show()

def estimate(s1, s2, c1, c2, total):
    p1 = (float) (c1 / ((float) (total)))
    p2 = (float) (c2 / ((float) (total)))

    z1 = erfinv(2 * p1 - 1)
    z2 = erfinv(2 * p2 - 1)

    miu = (z1 * s2 - z2 * s1) / (z1 - z2)

    sigma = (s1 - miu) / (z1 * sqrt(2))
    #theta2 = (s2 - miu) / (z2 * sqrt(2))

    return miu, sigma

def guess(data):
    import matplotlib.pyplot as plt
    num_bins = 1000
    n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor='green', alpha=0.5)



    y = mlab.normpdf(bins, 560, 55)
    plt.plot(bins, y, 'r--')


    plt.xlabel('Scores')
    plt.ylabel('Probability')
    plt.title(r'Histogram of guess')

    plt.show()

if __name__ == '__main__':

    data, t_miu = load_data_with_projection("data.txt")
    miu, sigma = estimate(562, 490, 80062, 150651, 257728)

    #data, t_miu = load_data("wk.txt")
    #miu, sigma = estimate(568, 510, 21243, 53556, 138374)

    #estimate using the 750
    #data, t_miu = load_data("data.txt")
    #miu, sigma = estimate(562, 750, 80062, 1, 257728)


    #data, t_miu = load_data("data.txt")
    #miu, sigma = estimate(490, 750, 150651, 1, 150652)

    #Use partial data to estimate
    #data, t_miu = load_partial_data("data.txt")
    #miu, sigma = estimate(562, 490, 80062, 150651, len(data))


    guess(data)

    #print miu, sigma, t_miu / len(data)
    #show_data(data, miu, abs(sigma))


    #print score_project(230)
