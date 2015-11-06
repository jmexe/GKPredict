__author__ = 'Jmexe'
from scipy.special import erfinv
from math import sqrt,log

def estimate(s1):
    miu = (s1 + 700) / 2
    sigma = (750 - miu) / (erfinv(2 * 0.0001 - 1) * sqrt(2))


    return miu, sigma


if __name__ == '__main__':
    miu, sig