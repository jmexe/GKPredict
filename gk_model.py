__author__ = 'Jmexe'
#vim: set fileencoding:utf-8
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.special import erfinv
from math import sqrt
import numpy as np

class GKModel(object):
    """--------------Initial---------------------------"""
    def __init__(self, s1, c1, s2, c2, total=-1, low_cut=200, h_total_prop=0.08, l_total_prop=0,  h_stop_corr=0, highest_score=0):
        """
        Initialize the model
        :param file_path: File path of the score data
        :param delimiter: The delimiter between each column
        :return:
        """
        self.given_total = total
        self.total = total

        self.s1 = s1
        self.s2 = s2
        self.c1 = c1
        self.c2 = c2


        #The lowest cutting score, the score under 200 will not be considered
        self.cut = low_cut

        #Proportion of high score students, should be in range(0.005, 0.23)
        if h_total_prop < 0:
            self.h_total_prop = 0.005
        elif h_total_prop > 0.25:
            self.h_total_prop = 0.25
        else:
            self.h_total_prop = h_total_prop

        #Proportion of high score students, in range(0~0.2)
        if l_total_prop < 0:
            self.l_total_prop = 0
        elif l_total_prop > 0.2:
            self.l_total_prop = 0.2
        else:
            self.l_total_prop = l_total_prop

        #Correct the model change score in range(-100, 100)
        if h_stop_corr < -100:
            self.h_stop_corr = h_stop_corr
        elif h_stop_corr > 100:
            self.h_stop_corr = h_stop_corr
        else:
            self.h_stop_corr = h_stop_corr

        #Highest score, in range(650, 749)
        if highest_score < 650:
            self.highest_score = 650
        elif highest_score > 749:
            self.highest_score = 749
        else:
            self.highest_score  =  highest_score

    def fit(self, file_path, delimiter='\t'):
        self.load_data(file_path, delimiter)
        self.estimate()

    def re_fit(self, s1=-1, c1=-1, s2=-1, c2=-1, total=-1):
        if self.check_model() is not True:
            print "Estimate model first!"
            return

        old_total = self.total
        if total > 0:
            self.given_total = total
            self.total = total


        if s1 > 0:
            self.s1 = s1

        if s2 > 0:
            self.s2 = s2

        if c1 > 0:
            self.c1 = c1
        else:
            self.c1 = (int) (self.c1 / float(old_total) * self.total)




        if c2 > 0:
            self.c2 = c2
        else:
            self.c2 = (int) (self.c2 / float(old_total) * self.total)


        self.estimate()


    def load_data(self, file_path, delimiter='\t'):
        """
        Given file path ,load the score data, the file should be in following format:
        1st column -- score
        2nd column -- count of students that got the score in column1
        3rd column (optional) -- rank of the score
        4th colunm (optional) -- accumulated count

        Will cut the score that's under 200!!!

        :param filename: file path
        :return: data
        """
        file = open(file_path)

        self.data = {}
        acc_cnt = 0
        for line in file:
            #Split the row
            split = line.strip().split(delimiter)
            #Get the score(1st column)
            score = (int)(split[0])

            if score < self.cut or score > 750:
                continue
            else:
                #Accumulate the count of students
                acc_cnt += (int)(split[1])
                #Add the count and score to the dictionary
                self.data[(int)(split[0])] = [(int)(split[1]), acc_cnt]
        self.total = acc_cnt

    def show_data(self, num_bins):
        """
        Plot the data
        :return:
        """
        if self.check_data() is not True:
            print "Please fit data first!"
            return

        self.pts = []

        for key in self.data.keys():
            for i in range(self.data[key][0]):
                self.pts.append(key)

        plt.hist(self.pts, num_bins, normed=1, facecolor='green', alpha=0.5)
        plt.show()

    def estimate_gmm_score_rank(self, score):
        """
        Given score, estimate the rank using the mixture of the overall guassian and high-score gussian models.
        :param score: score
        :return: rank and the proportion of the rank
        """
        #estimate count using low score guassian
        l_norm = norm(self.a_miu, self.a_sig)
        a_rank = self.total * (1 - self.l_total_prop - self.h_total_prop) * (1 - l_norm.cdf(score) / l_norm.cdf(self.h_miu + self.h_stop_corr))

        #estimate count using high score guassian
        h_norm = norm(self.h_miu, self.h_sig)
        h_rank = self.total * self.h_total_prop * (1 - h_norm.cdf(score))


        return (int) (a_rank + h_rank), (a_rank + h_rank) / self.total

    def estimate_higscore_score_rank(self, score):
        """
        Give score, estimate the rank using the high-score guassian model.
        :param score: score
        :return: rank and the proportion of the rank
        """

        proption = ((1 - norm(self.h_miu, self.h_sig).cdf(score)) * self.h_total_prop)

        if proption < 0:
            proption = 0

        return (int) (proption * self.total), proption

    def estimate_rank(self, score, total = -1):
        if total < 0:
            total = self.total

        if score < self.cut or score > 750:
            return -1, 0
        if score > self.h_miu + self.h_stop_corr:
            return self.estimate_higscore_score_rank(score)[1] * total
        else:
            return self.estimate_gmm_score_rank(score)[1] * total


    def estimate_rank_for_scores(self, score_arr):
        """
        Give an array of scores, return the estimation of the ranks for each score
        :param score_arr: The array of scores
        :return: An array of ranks
        """
        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return
        ranks = []
        for score in score_arr:
            if score > self.h_miu + self.h_stop_corr:
                ranks.append(self.estimate_higscore_score_rank(score)[0])
            else:
                ranks.append(self.estimate_gmm_score_rank(score)[0])
        return ranks

    def show_model(self, num_bins):
        """
        Given the number of the bins of the figure, show tow subfigures,
        left is the histogram of the real data and the pmf of two guassians
        right is the cumulative probabilites of the real data and estimated model
        :param num_bins:
        :return:
        """
        if self.check_data() is not True:
            print "Please fit data first!"
            return

        if self.check_model() is not True:
            self.estimate()

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

        #Subfigure 1 -- Histogram and two guassian distributions

        #Prepare data
        if hasattr(self, "data") is not True:
            #load data
            self.load_data()
        if hasattr(self, "pts") is not True:
            #Convert cumulative rank to data points
            self.pts = []
            for key in self.data.keys():
                for i in range(self.data[key][0]):
                    self.pts.append(key)


        #Plot histogram
        n, bins, patches = ax0.hist(self.pts, num_bins, normed=1, facecolor='green', alpha=0.5)

        #Plot overall guassian
        y11 = mlab.normpdf(bins, self.a_miu, self.a_sig) * (1 - self.h_total_prop)
        ax0.plot(bins, y11, "red")

        #Plot overall guassian
        y12 = mlab.normpdf(bins, self.h_miu, self.h_sig) * self.h_total_prop
        ax0.plot(bins, y12, "blue")

        #Subfigure2 -- cumulative probability
        x_pts = np.linspace(self.cut, 750, 100)
        y21 = [1 - rk / (float) (self.total) for rk in self.estimate_rank_for_scores(x_pts)]

        #Load the real cumulative proportion
        y22 = []
        for i, pt in enumerate(x_pts):
            pt = (int) (pt)
            if self.data.has_key(pt):
                y22.append(1 - self.data[pt][1] / (float) (self.total))
            else:
                #If the score is not in the dict, use the nearest rank
                if i == 0:
                    y22.append(0)
                elif pt < self.cut:
                    y22.append(0)
                elif pt > 750:
                    y22.append(1)
                else:
                    #TODO -- optimize
                    tmp = pt
                    rank = -1
                    while tmp > pt - 20:
                        if self.data.has_key(tmp):
                            rank = self.data[tmp][1]
                            break
                        tmp = tmp - 1

                    if rank > 0:
                        y22.append(1 - rank / (float) (self.total))
                    else:
                        y22.append(y22[i - 1])

        ax1.plot(x_pts, y21, '--', linewidth=2)
        ax1.plot(x_pts, y22, '-', linewidth=2)

        plt.show()

    def compare_data(self, score_arr):
        """
        Given an array of scores, estimate the ranks.
        :param score_arr: An array of scores
        :return: The estimated ranks and real ranks of the scores
        """
        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return

        ranks = self.estimate_rank_for_scores(score_arr)
        if hasattr(self, "data") is not True:
            #load data
            print "Data are not loaded yet!"
            return

        real_ranks = []
        for score in score_arr:
            if self.data.has_key(score):
                real_ranks.append(self.data[score][1])
            else:
                real_ranks.append("not exist")

        return ranks, real_ranks


    """--------------Estimation------------------------"""

    def estimate_guassian(self, s1, c1, s2, c2, total):
        """
        Given 2 data points, estimate the guassian parameters
        """
        #Calculate cumulative probability of each point
        p1 = 1 - (float) (c1 / ((float) (total)))
        p2 = 1 - (float) (c2 / ((float) (total)))

        #Calculate inverse erf value
        z1 = erfinv(2 * p1 - 1)
        z2 = erfinv(2 * p2 - 1)

        #Calculate miu
        miu = (z1 * s2 - z2 * s1) / (z1 - z2)

        #Calculate sigma using two points, return the mean of the two sigmas
        sig1 = (s1 - miu) / (z1 * sqrt(2))
        sig2 = (s2 - miu) / (z2 * sqrt(2))

        return miu, (abs(sig1) + abs(sig2)) / 2

    def high_score_estimate(self, s1, s2, total):
        """
        Estimate the guassian in high score area
        Assume there is only 1 student get 720
        Assume 90% of the students get higher score that s1
        """
        c1 = 0.9 * total
        c2 = 1

        return self.estimate_guassian(s1, c1, s2, c2, total)

    def all_score_estimate(self, s1, c1, s2, c2, total):
        """
        Estimate the guassian of the overall data
        Use the score and students count at s1&s2
        """
        return self.estimate_guassian(s1, c1, s2, c2, total)

    def estimate(self):
        """
        Estimate the model using the given parameters and correction parameters
        Generating the score to rank table after estimating the model
        :return:
        """
        self.a_miu, self.a_sig = self.all_score_estimate(self.s1, self.c1, self.s2, self.c2, self.total)
        self.h_miu, self.h_sig = self.high_score_estimate(self.s1, self.highest_score, self.total * self.h_total_prop)
        self.sample_score_to_rank_table()


    """--------------Check parameters------------------"""
    def check_file(self):
        """
        Check whther the file path is defined
        :return:
        """
        return hasattr(self, "file_path")

    def check_data(self):
        """
        Check whether the data is loaded
        """
        return hasattr(self, "data")


    def check_model(self):
        """
        Check whether the models are estimated
        """
        return hasattr(self, "a_miu") and hasattr(self, "a_sig") and hasattr(self, "h_miu") and hasattr(self, "h_sig")

    def check_rank(self):
        return hasattr(self, "prop_table")

    """--------------Print parameters------------------"""
    def print_model(self):
        """
        Print the parameters of the guassian models
        """
        if self.check_model():
            print "Overall guassian parameters -- miu:", self.a_miu, " sigma:", self.a_sig
            print "High-score guassian parameters -- miu:", self.h_miu, " sigma:", self.h_sig
        else:
            print "Models are not estimated yet!"


    def print_parameters(self):
        """
        Print the parameters and correction values
        """
        if self.check_parameters() is True:
            print "一本线:", self.s1, "排名", self.c1, " -- 二本线", self.s2, "排名", self.c2, " -- 总人数", self.total
        else:
            print "请输入一本线、一本排名、二本线、二本排名、总人数!"

        print "修正值 -- 最高分:", self.highest_score, " 高分人数比例:", self.h_total_prop, "低分人数比例:", self.l_total_prop, "模型切换分数线修正:", self.h_stop_corr

    """--------------Predict score/rank data-----------"""
    def sample_score_to_rank_table(self, total=-1):
        """
        Create the score-to-rank table and the score-to-proportion table
        Return the score-to-rank table
        The total is optional
        """

        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return

        total = self.total
        score_to_rank_table = {}
        self.prop_table = {}
        for score in range(self.cut, 751):

            if score > self.h_miu + self.h_stop_corr:
                #If the score is higher the model changing line, estimating the rank using the high-score guassian
                score_to_rank_table[score] = (int) (self.estimate_higscore_score_rank(score)[1] * total)
                self.prop_table[score] = self.estimate_higscore_score_rank(score)[1]
            else:
                #If the score is lower than the model changing line, estimating the rank using the mixture guassian
                score_to_rank_table[score] = (int) (self.estimate_gmm_score_rank(score)[1] * total)
                self.prop_table[score] = self.estimate_gmm_score_rank(score)[1]


        return score_to_rank_table


    def get_rank_to_score(self, rank, total=-1):
        """
        Given rank ,get the score
        """

        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return

        if self.check_rank() is not True:
            self.sample_score_to_rank_table()

        if total < 0:
            total = self.total

        if rank < 0 or rank > total:
            print "Rank must be in range(1, %d)" % total
            return

        if rank < self.prop_table[(int)(self.h_miu + self.h_stop_corr)] * total:
            return (int) (erfinv(1 - 2 * rank / float(total * self.h_total_prop)) * sqrt(2) * self.h_sig + self.h_miu)
        else:
            lo = self.cut
            hi = self.prop_table[(int)(self.h_miu + self.h_stop_corr)] * total

            prop = rank / float(total)

            while lo < hi:
                mid = (int) ((lo + hi) / 2)
                if self.prop_table[mid] > prop:
                    lo = mid
                else:
                    hi = mid

                if lo >= hi - 1:
                    return lo

            return -1

        #return (int) (self.prop_table[score] * total)

if __name__ == '__main__':

    """ ---------------------------2015 山东文科数据----------------------------------"""
    #-----------------初始化模型--------------------#
    #必须给定的参数：
    #一本线， 一本线排名， 二本线，二本线排名 -- (568, 21243, 510, 53556)
    #总人数可以不给出，如果给定，可以直接调用estimate()函数去估计参数，不然必须通过fit()函数去读取文件，同时estimate
    #low_cut是最低分数线，低于此分数线的分数不予考虑，默认值是200
    #h_total_prop   --  高分考生比例，默认值0.08，范围(0, 0.25)
    #l_total_prop   --  低分考生比例，默认值0, 范围(0, 0.2)
    #h_stop_corr    --  切换模型分数线修正值, 修正范围(-100, 100)，默认为高分高斯的miu值，此处设置为-10， 则模型中模型切换的分数线为 h_miu - 10
    #highest_score  --  最高分，默认是700，范围(650, 749)
    model = GKModel(516, 13556, 439, 27275, total=-1, low_cut=200, h_total_prop=0.08, l_total_prop=0,  h_stop_corr=-10, highest_score=705)
    #Fit数据，读取数据，重新计算总人数，fit完后可以调用show_model()函数输出图表
    model.fit("./data/tj-2014-l.txt.csv",delimiter='\t')
    model.show_model(200)
    #print model.compare_data([500, 600])
    #print model.sample_score_to_rank_table()

    #Re-fit函数，通过往年模型，给定当年部分参数，重新估计当年参数
    #s1 -- 一本线，默认采用往年一本线
    #c1 -- 一本排名，默认采用 往年一本线所占考生比例 * 当年总人数
    #s2 -- 二本线，默认采用往年二本线
    #c2 -- 二本排名，默认采用 往年二本线所占考生比例 * 当年总人数
    #total -- 总人数，默认值会采用往年总人数
    #model.re_fit(s1=550, s2=500)



    #print model.sample_score_to_rank_table()
