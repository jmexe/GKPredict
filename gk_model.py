__author__ = 'Jmexe'
#vim: set fileencoding:utf-8
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.special import erfinv
from math import sqrt
import numpy as np

class GKModel(object):
    def __init__(self, file_path="", delimiter='\t'):
        """
        Initialize the model
        :param file_path: File path of the score data
        :param delimiter: The delimiter between each column
        :return:
        """
        if file_path is not "":
            self.file_path = file_path

        #The lowest cutting score, the score under 200 will not be considered
        self.cut = 200

        #Proportion of high score students
        self.h_total_prop = 0.08
        #The proportion of the students that can be eliminated
        self.l_total_prop = 0
        #Correction for high-score guassian model
        self.h_stop_corr = 0
        #Highest score
        self.highest_score = 700

        self.delimiter = delimiter

    def check_file(self):
        """
        Check whther the file path is defined
        :return:
        """
        return hasattr(self, "file_path")

    def load_data(self):
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
        if self.check_file() is not True:
            print "No data to load"
            return

        file = open(self.file_path)

        self.data = {}
        acc_cnt = 0
        for line in file:
            #Split the row
            split = line.strip().split(self.delimiter)
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
        if self.check_file() is not True:
            print "File path not defined!"
            return

        if self.check_data() is not True:
            self.load_data()

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
        if self.check_file() is not True:
            print "File path not defined!"
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
                if pt < self.cut:
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
        if self.check_file() is not True:
            print "File path not defined!"
            return

        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return

        ranks = self.estimate_rank_for_scores(score_arr)
        if hasattr(self, "data") is not True:
            #load data
            self.load_data()

        real_ranks = []
        for score in score_arr:
            if self.data.has_key(score):
                real_ranks.append(self.data[score][1])
            else:
                real_ranks.append("not exist")

        return ranks, real_ranks



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
        if self.check_parameters():
            self.a_miu, self.a_sig = self.all_score_estimate(self.s1, self.c1, self.s2, self.c2, self.total)
            self.h_miu, self.h_sig = self.high_score_estimate(self.s1, self.highest_score, self.total * self.h_total_prop)
            self.score_to_rank_table()
        else:
            print "Parameter are not all set!"

    def set_parameters(self, s1, c1, s2, c2, total=-1, low_cut=200):
        """
        Set the score points for estimating the model
        :param s1: 一本线
        :param c1: rank of s1
        :param s2: 二本线
        :param c2: rank of s2
        :param total: total students count, it's optional, if it's not given, must call the load data before starting the estimation
        :param low_cut: The lowest cutting score, which means the score under that line will not be considered in the model, default value is 200
        :return:
        """
        self.s1 = s1
        self.c1 = c1
        self.s2 = s2
        self.c2 = c2

        self.given_total = total
        self.total = total

        self.cut = low_cut

    def set_corrections(self, h_total_corr=0, l_total_corr=0,  h_stop_corr=0, highest_score_corr=0):
        """
        Set the parameters for model correction
        :param h_total:             Correction for the proportion of students modeled by the high-score guassian, default value is 8%
        :param l_total_corr:        Correction for the proportion of the students that can be eliminated, default value is 0.
        :param h_stop_corr:         Correction for the miu of the high-score guassian (used for changing model when estimate the rank)
        :param highest_score_corr:  Correction for the highest score student, used for estimating the high-score guassian. Default value is 700
        :return:
        """

        #Correct the high-score proportion in range(0.005, 0.23)
        if h_total_corr < -0.075:
            self.h_total_prop = 0.005
        elif h_total_corr > 0.15:
            self.h_total_prop = 0.23
        else:
            self.h_total_prop += h_total_corr

        #Correct the low-score proportion in range(0~0.2)
        if l_total_corr < 0:
            self.l_total_prop = 0
        elif l_total_corr > 0.2:
            self.l_total_prop = 0.2
        else:
            self.l_total_prop += l_total_corr

        #Correct the model change score in range(-100, 100)
        if h_stop_corr < -100:
            self.h_stop_corr = h_stop_corr
        elif h_stop_corr > 100:
            self.h_stop_corr = h_stop_corr

        #Correct the highest score in range(-50, 50)
        if highest_score_corr < -50:
            self.highest_score = 650
        elif highest_score_corr > 50:
            self.highest_score = 749
        else:
            self.highest_score  +=  highest_score_corr


    def print_parameters(self):
        """
        Print the parameters and correction values
        """
        if self.check_parameters() is True:
            print "一本线:", self.s1, "排名", self.c1, " -- 二本线", self.s2, "排名", self.c2, " -- 总人数", self.total
        else:
            print "请输入一本线、一本排名、二本线、二本排名、总人数!"

        print "修正值 -- 最高分:", self.highest_score, " 高分人数比例:", self.h_total_prop, "低分人数比例:", self.l_total_prop, "模型切换分数线修正:", self.h_stop_corr

    def check_data(self):
        """
        Check whether the data is loaded
        """
        return hasattr(self, "data")

    def check_parameters(self):
        """
        Check whether all the required parameters are all set
        """
        if hasattr(self, "s1") and hasattr(self, "c1") and hasattr(self, "s2") and hasattr(self, "c2") and hasattr(self, "total"):
            return self.total > 0
        else:
            return False

    def check_model(self):
        """
        Check whether the models are estimated
        """
        return hasattr(self, "a_miu") and hasattr(self, "a_sig") and hasattr(self, "h_miu") and hasattr(self, "h_sig")

    def check_rank(self):
        return hasattr(self, "prop_table")

    def print_model(self):
        """
        Print the parameters of the guassian models
        """
        if self.check_model():
            print "Overall guassian parameters -- miu:", self.a_miu, " sigma:", self.a_sig
            print "High-score guassian parameters -- miu:", self.h_miu, " sigma:", self.h_sig
        else:
            print "Models are not estimated yet!"

    def score_to_rank_table(self, total=-1):
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



    def rank_to_score(self, rank, total=-1):
        """
        Given rank ,estimate the score
        """


        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return

        if self.check_rank() is not True:
            self.score_to_rank_table()

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
    #三个例程, 删掉注释可用



    """ ---------------------------2015 山东文科数据----------------------------------"""
    """
    #初始化模型
    #数据文件路径 -- 数据格式在代码中有注释
    #省份、年份、文理科
    model = GKModel("./data/sd-2015-w.txt", "山东", 2015, "文科")

    #设置参数
    #一本线， 一本线排名， 二本线，二本线排名， 考生总人数
    model.set_parameters(568, 21243, 510, 53556, 138374)

    #设置修正值(optional)
    #此处设置的是修正值，都是对模型中参数进行修正
    #模型中默认参数：高分考生比例8%, 模型切换分数是高分高斯的miu值，低分考生比例为0，考生最高分为700
    #
    #h_total_corr:       高分人数修正值，修正范围(-0.08, 0.15)，系统默认值0.08，此处设为0，则模型中高分考生比例为 8%+0 = 8%
    #h_stop_corr:        切换模型分数线修正值, 修正范围(-100, 100)，默认为高分高斯的miu值，此处设置为-10， 则模型中模型切换的分数线为 h_miu - 10
    #l_total_corr:       低分考生比例修正值，修正范围(0, 0.3)，系统默认值0，此处设为0，则模型中高分考生比例为 0%+0 = 0%
    #highest_score_corr: 考生最高分修正值，修正范围(-50, 50)，系统默认值为700，此处设为0，则模型中最高分考生为700 + 0 = 700
    model.set_corrections(h_total_corr=0, h_stop_corr=-10, l_total_corr=0, highest_score_corr=5)

    #根据输入数据估计模型
    model.estimate()

    #测试模型
    #给定一组分数，估计排名
    test_scores = [500, 540, 580, 620, 660, 700, 720]
    estimate_ranks, real_ranks = model.compare_data(test_scores)
    for i, score in enumerate(test_scores):
        print score, "-- est rank:", estimate_ranks[i], "-- real rank:", real_ranks[i]

    #画图,参数为x轴坐标密度
    model.show_model(100)
    """

    """ ---------------------------2015 山东理科数据----------------------------------"""

    """
    #Initialize the model
    model = GKModel("./data/sd-2015-l.txt", "山东", 2015, "理科")

    #Set the parameters
    model.set_parameters(562, 80062, 490, 150651, 257728)
    #Set the correction parameters
    model.set_corrections(h_total_corr=0, h_stop_corr=50, l_total_corr=0, highest_score_corr=40)
    #Estimate the model
    model.estimate()
    #Given an array of scores, compare the estimate ranks with real ranks
    test_scores = [500, 540, 580, 620, 660, 700, 720]
    estimate_ranks, real_ranks = model.compare_data(test_scores)
    for i, score in enumerate(test_scores):
        print score, "-- est rank:", estimate_ranks[i], "-- real rank:", real_ranks[i]
    #Plot the model
    model.show_model(100)
    """


    """ ---------------------------2015 广西理科数据----------------------------------"""
    """
    #Initialize the model
    model = GKModel("./data/gx-2015-l.txt", "广西", 2015, "理科")

    #Set the parameters
    model.set_parameters(480, 25123, 320, 97807, 145318)
    #Set the correction parameters
    model.set_corrections(h_total_corr=0, h_stop_corr=0, l_total_corr=0, highest_score_corr=0)
    #Estimate the model
    model.estimate()
    #Given an array of scores, compare the estimate ranks with real ranks
    test_scores = [500, 540, 580, 620, 660, 700, 720]
    estimate_ranks, real_ranks = model.compare_data(test_scores)
    for i, score in enumerate(test_scores):
        print score, "-- est rank:", estimate_ranks[i], "-- real rank:", real_ranks[i]
    #Plot the model
    model.show_model(100)
    """

    """ ---------------------------2015 广西理科数据----------------------------------"""
    """
    #Initialize the model
    model = GKModel("./data/yn-2014-l.txt", "云南", 2014, "理科")

    #Set the parameters
    model.set_parameters(525, 21363, 445, 54007, 124963)

    model.load_data()
    model.show_data(100)

    #Set the correction parameters
    model.set_corrections(h_total_corr=0, h_stop_corr=39, l_total_corr=0, highest_score_corr=43)
    #Estimate the model
    model.estimate()
    #Given an array of scores, compare the estimate ranks with real ranks
    test_scores = [220, 500, 540, 580, 620, 660, 700, 720]
    estimate_ranks, real_ranks = model.compare_data(test_scores)
    for i, score in enumerate(test_scores):
        print score, "-- est rank:", estimate_ranks[i], "-- real rank:", real_ranks[i]
    #Plot the model
    model.show_model(100)
    """

    """ ---------------------------2015 内蒙文科数据----------------------------------"""
    #Initialize the model
    model = GKModel(file_path = "./data/neimenggu-2015-w.csv", delimiter=',')
    #Set the parameters
    model.set_parameters(525, 4755, 455, 14273, total=50469)
    #Set the correction parameters
    model.set_corrections(h_total_corr=-0.06, h_stop_corr=0, l_total_corr=0.1, highest_score_corr=-30)
    #Estimate the model
    model.estimate()
