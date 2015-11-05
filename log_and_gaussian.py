__author__ = 'Jmexe'

import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def generate_gaussian(miu, sigma, num):
    x = miu + sigma * np.random.randn(num)
    return x

def plot_gaussian(miu, sigma, data):
    num_bins = 100
    # the histogram of the data
    n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, miu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')

    plt.show()

def weib(x,n,a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
def weibull(a, num):
    s = np.random.weibull(a, num)

    s = s / s.max() * 750

    count, bins, ignored = plt.hist(s, 100)

    #scale = count.max()/weib(x, 1., 5.).max()

    #plt.plot(x, weib(x, 750, 5.)*scale)
    plt.show()


if __name__ == '__main__':
    #data = list(generate_gaussian(80, 520, 5000)) + list(generate_gaussian(1000, 300, 1000))
    data = list(generate_gaussian(195, 40, 500)) + list(generate_gaussian(450, 92, 8800)) + list(generate_gaussian(600, 33, 700))
    plot_gaussian(200, 200, data)