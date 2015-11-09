__author__ = 'Jmexe'
#vim: set fileencoding:utf-8
from scipy.special import erfinv
from math import sqrt,log
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm

def load_data_in_dict(filename):
    """
    Given file path ,load the score data, the file should be in following format:
    1st column -- score
    2nd column -- count of students that got the score in column1
    3rd column (optional) -- rank of the score
    4th colunm (optional) -- accumulated count
    :param filename: file path
    :return: data
    """
    file = open(filename)

    data = {}
    acc_cnt = 0
    for line in file:
        split = line.strip().split("\t")
        acc_cnt += (int)(split[1])
        data[(int)(split[0])] = acc_cnt

    return data

def estimate_guassian(s1, c1, s2, c2, total):
    """
    Given 2 data points, estimate the guassian parameters
    """
    #Calculate
    p1 = 1 - (float) (c1 / ((float) (total)))
    p2 = 1 - (float) (c2 / ((float) (total)))

    z1 = erfinv(2 * p1 - 1)
    z2 = erfinv(2 * p2 - 1)

    miu = (z1 * s2 - z2 * s1) / (z1 - z2)

    sig1 = (s1 - miu) / (z1 * sqrt(2))
    sig2 = (s2 - miu) / (z2 * sqrt(2))

    return miu, (abs(sig1) + abs(sig2)) / 2

def high_score_estimate(s1, s2, total):
    """
    Estimate the guassian in high score area
    Assume there is only 1 student get 720
    Assume 90% of the students get higher score that s1
    """
    #miu = (s1 + 750) / 2
    #sigma = (750 - miu) / (erfinv(2 * 0.99999 - 1) * sqrt(2))

    c1 = 0.9 * total
    c2 = 1

    return estimate_guassian(s1, c1, s2, c2, total)

def all_score_estimate(s1, c1, s2, c2, total):
    """
    Estimate the guassian of the overall data
    Use the score and students count at s1&s2
    """
    return estimate_guassian(s1, c1, s2, c2, total)

def plot_his(data, num_bins):
    n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor='green', alpha=0.5)
    return bins


def sample(miu, sig, num):
    """
    Given N and parameters of guassian, generate N samples from the guassian.
    """
    x = miu + sig * np.random.randn(num)
    return np.array(x).astype(int)

def plot_guassian(miu, sig, bins, color, ratio):
    """
    Given parameters of guassian, plot curve of the guassian on plt
    """
    y = mlab.normpdf(bins, miu, sig) * ratio
    plt.plot(bins, y, color)


def gmm_score_rank(score, a_miu, a_sig, h_miu, h_sig, total, h_num, corr_portion):
    a_rank = (1 - norm(a_miu, a_sig).cdf(score)) * (total * (1 - corr_portion) - h_num)
    h_rank = (1 - norm(h_miu, h_sig).cdf(score)) * h_num

    return (int) (a_rank + h_rank)

def single_score_rank(score, miu1, sig1, total, h_num, corr_portion):
    return (int) ((1 - norm(miu1, sig1).cdf(score)) * h_num)


def score_points(data, a_miu, a_sig, h_miu, h_sig, total, h_num, corr_portion, thrh):
    rslt = []
    for pt in data:
        if pt > thrh:
            rslt.append(single_score_rank(pt, h_miu, h_sig, total, h_num, corr_portion))
        else:
            rslt.append(gmm_score_rank(pt, a_miu, a_sig, h_miu, h_sig, total, h_num, corr_portion))

    return rslt


def draw_accumulate(a_miu, a_sig, h_miu, h_sig, total, h_num, corr_portion, thrh):
    lspts = np.linspace(180, 750, 100)
    y = [1 - rk / (float) (total) for rk in score_points(lspts, a_miu, a_sig, h_miu, h_sig, total, h_num, corr_portion, thrh)]

    return lspts, y

if __name__ == '__main__':

    #shandong - 2015 - lk
    s1 = 562
    c1 = 80062
    s2 = 490
    c2 = 150651

    total = 257728


    #
    h_total = 20000

    #
    corr_portion = 0

    h_miu_corr = 50

    h_score_corr = 40

    filename = "./data/sd-2015-l.txt"


    """
    #shandong 2015 wk
    s1 = 568
    c1 = 21243
    s2 = 510
    c2 = 53556

    total = 138374
    h_total = 5000

    corr_portion = 0
    h_miu_corr = 0
    h_score_corr = 5
    filename = "./data/sd-2015-w.txt"
    """

    """
    #guangxi 2015 lk
    s1 = 480
    c1 = 25123
    s2 = 320
    c2 = 97807

    total = 145318
    h_total = 10000

    corr_portion = 0.2
    h_miu_corr = 50
    h_score_corr = 0
    filename = "./data/gx-2015-l.txt"
    """

    """
    #hebei 2015 lk

    s1 = 480
    c1 = 25123
    s2 = 320
    c2 = 97807

    total = 145318
    h_total = 10000

    corr_portion = 0.2
    h_miu_corr = 50
    h_score_corr = 0
    filename = "./data/gx-2015-l.txt"
    """

    #estimate guassian in high score area
    h_miu, h_sig = high_score_estimate(s1, 700 + h_score_corr, h_total)

    #estimate overall guassian -- remove the portion in high score area
    a_miu, a_sig = all_score_estimate(s1, c1 - h_total, s2, c2 - h_total, total * (1- corr_portion) - h_total)

    print a_miu, a_sig, h_miu, h_sig

    #from predict import load_data
    #data, tmiu = load_data("./data/sd-2015-l.txt")

    #calculate the ration of the high score guassian
    #ratio = (float)(total - h_total) / (float)(total)

    pts = [500, 510, 540, 560, 600, 620, 640, 660, 680, 700, 710]

    ranks = score_points(pts, a_miu, a_sig, h_miu, h_sig, total, h_total, corr_portion, h_miu + h_miu_corr)


    for i,v in enumerate(pts):
        print v, ranks[i]



    ddict = load_data_in_dict(filename)
    x, y = draw_accumulate(a_miu, a_sig, h_miu, h_sig, total, h_total, corr_portion, h_miu + h_miu_corr)

    y2 = []
    for i, pt in enumerate(x):
        pt = (int) (pt)
        if ddict.has_key(pt):

            accu_prob = 1 - ddict[pt] / (float) (total)

            y2.append(accu_prob)
        else:
            if i == 0:
                y2.append(0)
            else:
                y2.append(y2[i - 1])

    line, = plt.plot(x, y, '--', linewidth=2)

    plt.plot(x, y2, '-', linewidth=2)

    plt.show()

    """

    bins = plot_his(data, 200)
    plot_guassian(h_miu, h_sig, bins, "r--", 1 - ratio)
    plot_guassian(a_miu, a_sig, bins, "b--", ratio)

    print tmiu, h_miu, h_sig, a_miu, a_sig

    plt.show()
    """


    """
    #plot data
    data = list(sample(h_miu, h_sig, h_total)) + list(sample(a_miu, a_sig, total - h_total))
    bins = plot_his(data, 200)
    plt.show()
    """