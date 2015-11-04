__author__ = 'Jmexe'
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.special import erfinv
from math import sqrt

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

def show_data(data, miu, sigma):
    import matplotlib.pyplot as plt
    num_bins = 50
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

if __name__ == '__main__':
    #data, t_miu = load_data("data.txt")
    #miu, sigma = estimate(562, 490, 80062, 150651, 257728)

    #data, t_miu = load_data("wk.txt")
    #miu, sigma = estimate(568, 510, 21243, 53556, 138374)

    data, t_miu = load_data("data.txt")
    miu, sigma = estimate(562, 750, 80062, 1, 257728)

    print miu, sigma, t_miu / (int)(138374)
    show_data(data, miu, abs(sigma))