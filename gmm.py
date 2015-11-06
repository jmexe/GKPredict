__author__ = 'Jmexe'
from sklearn.mixture import GMM
from predict import load_data_for_gmm
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def train_gmm(filename):
    g =GMM(n_components=3)

    data,t_miu = load_data_for_gmm(filename)
    g.fit(data)

    return g, data

def generate_test():
    lspts = np.linspace(180, 750, 1000)
    data = [[pt] for pt in lspts]
    return data, lspts

def generate_samples():
    samples = g.sample(100000)
    pts = []
    for pt in samples:
        pts.extend(pt)

    num_bins = 1000
    n, bins, patches = plt.hist(pts, num_bins, normed=1, facecolor='green', alpha=0.5)

    plt.xlabel('Scores')
    plt.ylabel('Probability')
    plt.title(r'Histogram of guess')

    plt.show()



if __name__ == '__main__':
    print "TBD"
    """
    g, data = train_gmm("data.txt")

    x, pts = generate_test()
    y = - g.score_samples(x)[0]
    print x
    print y
    num_bins = 1000
    n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor='green', alpha=0.5)
    plt.plot(pts, y, 'r--')
    plt.show()
    """