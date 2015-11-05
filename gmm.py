__author__ = 'Jmexe'
from sklearn.mixture import GMM

if __name__ == '__main__':
    from predict import load_data_count
    g =GMM(n_components=3)
    g.fit()