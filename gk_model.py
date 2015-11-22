__author__ = 'Jmexe'
#vim: set fileencoding:utf-8
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.special import erfinv
from math import sqrt
import numpy as np
import matplotlib.lines as mlines

class GKModel(object):
    """--------------Initial---------------------------"""
    def __init__(self, s1, c1, s2, c2, total=-1, low_cut=200, high_score_students_proportion=0.08, low_score_students_proportion=0,  model_change_score_corr=0, highest_score=0, left_high_portion = 0.3, high_start_corr=0):
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
        if high_score_students_proportion < 0:
            self.high_score_students_proportion = 0.005
        elif high_score_students_proportion > 0.25:
            self.high_score_students_proportion = 0.25
        else:
            self.high_score_students_proportion = high_score_students_proportion

        #Proportion of high score students, in range(0~0.2)
        if low_score_students_proportion < 0:
            self.low_score_students_proportion = 0
        elif low_score_students_proportion > 0.2:
            self.low_score_students_proportion = 0.2
        else:
            self.low_score_students_proportion = low_score_students_proportion

        #Correct the model change score in range(-100, 100)
        if model_change_score_corr < -100:
            self.model_change_score_corr = model_change_score_corr
        elif model_change_score_corr > 100:
            self.model_change_score_corr = model_change_score_corr
        else:
            self.model_change_score_corr = model_change_score_corr

        #Highest score, in range(650, 749)
        if highest_score < 650:
            self.highest_score = 650
        elif highest_score > 749:
            self.highest_score = 749
        else:
            self.highest_score  =  highest_score

        self.left_high_portion = left_high_portion
        self.prepared = False

        self.high_start_corr = high_start_corr

    def fit(self, file_path, delimiter='\t'):
        """
        Fit the model to the data, will load the data from file and recount the total number of studens, then re-estimate the model
        :param file_path:
        :param delimiter:
        :return:
        """
        #Load data
        self.load_data(file_path, delimiter)

        #Re-estimate
        self.estimate()

    def re_fit(self, s1=-1, c1=-1, s2=-1, c2=-1, total=-1):
        """
        Re-fit the model to the new parameters, will use the parameter that are adjusted in the old model
        :param s1:      一本线
        :param c1:      一本线对应排名
        :param s2:      二本线
        :param c2:      二本线对应排名
        :param total:   考生总人数
        :return:
        """
        if self.check_model() is not True:
            print "Estimate model first!"
            return

        #If total number is not give, will use the total number of the old number
        old_total = self.total
        if total > 0:
            self.given_total = total
            self.total = total

        #If 一本线 is given, will use the new 一本线
        if s1 > 0:
            self.s1 = s1

        #If 二本线 is given, will use the new 二本线
        if s2 > 0:
            self.s2 = s2

        #If the rank of s1 is not given, will use the portion of the old c1 to estimate new c1
        #Otherwise will use the new c1
        if c1 > 0:
            self.c1 = c1
        else:
            self.c1 = (int) (self.c1 / float(old_total) * self.total)

        #If the rank of s2 is not given, will use the portion of the old c2 to estimate new c2
        #Otherwise will use the new c2
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

    def show_data(self, num_bins, start, end):
        """
        Plot the data
        :return:
        """
        if self.check_data() is not True:
            print "Please fit data first!"
            return

        self.pts = []

        for key in self.data.keys():
            if key >= start and key <= end:
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
        portion_in_normal_students = (1 - self.a_norm.cdf(score) / self.a_norm.cdf(self.model_change))
        a_rank = (int) (self.total * (1 - self.low_score_students_proportion - self.high_score_students_proportion) * portion_in_normal_students)

        #estimate count using high score guassian
        portion_in_left_part_high_guassian = self.h_norm.cdf(self.model_change) - self.h_norm.cdf(score)
        h_rank = (int) (portion_in_left_part_high_guassian * self.left_high_total)

        return a_rank + h_rank + self.right_high_total

    def estimate_higscore_score_rank(self, score):
        """
        Give score, estimate the rank using the high-score guassian model.
        :param score: score
        :return: rank and the proportion of the rank
        """

        higher_part_portion_in_high_guassian = (1 - self.h_norm.cdf(score)) / (1 - self.h_norm.cdf(self.model_change))
        rank = (int) (self.right_high_total * higher_part_portion_in_high_guassian)

        return rank

    def prepare_estimate(self):
        """
        Preparation function, used for estimate the students count for each score range
        :return:
        """
        self.model_change = (int) (self.h_miu + self.model_change_score_corr)

        self.h_norm = norm(self.h_miu, self.h_sig)
        self.a_norm = norm(self.a_miu, self.a_sig)

        self.left_high_total = (int) (self.left_high_portion * self.high_score_students_proportion * self.total) * (self.h_norm.cdf(self.model_change))
        self.right_high_total = (int)(self.total * self.high_score_students_proportion) - self.left_high_total

        self.prepared = True

    def estimate_rank(self, score):
        """
        Given score, estimate the position(rank) of the score in the total students
        :param score:
        :return:
        """

        #return self.estimate_rank_folded_guassian(score)
        if self.prepared is not True:
            self.prepare_estimate()

        if score < self.cut or score > 750:
            return -1, 0
        if score > self.h_miu + self.model_change_score_corr:
            return (int) (self.estimate_higscore_score_rank(score))
        else:
            return (int) (self.estimate_gmm_score_rank(score))


    def estimate_rank_for_scores(self, score_arr):
        """
        Give an array of scores, return the estimation of the ranks for each score
        :param score_arr: The array of scores
        :return: An array of ranks
        """
        if self.prepared is not True:
            self.prepare_estimate()

        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return
        ranks = []
        for score in score_arr:
            ranks.append(self.estimate_rank(score))
        return ranks

    def show_model(self):
        """
        Plotting model data, there are four sub-figures

        up-left:    Error rate
        up-right:   PDF of the real data and model
        down-left:  Real data ppf
        down-right: Model ppf

        :return:
        """
        if self.check_data() is not True:
            print "Please fit data first!"
            return

        if self.check_model() is not True:
            self.estimate()

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)


        info = "highest_score:" + str(self.highest_score) + " low_score_students_proportion:" + str(self.low_score_students_proportion) + " high_score_students_proportion:" + str(self.high_score_students_proportion) + " left_high_portion:" + str(self.left_high_portion) + "model_change_score_corr:" + str(self.model_change_score_corr)
        x0, y0, err_rate = self.error_rate(self.cut + 1, 750)
        ax0.plot(x0, y0)
        ax0.set_title(err_rate)
        ax0.grid(True)

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

        #Get the pdf of the real data
        #Plot the data in the down-right sub-figure
        x2, y2, max_x2, max_y2 = self.real_pdf(self.cut + 1, 750)
        ax2.plot(x2, y2)

        #Mark the peak of the ppf
        real_ppf_line = mlines.Line2D([max_x2, max_x2], [0, max_y2], lw=2., alpha=0.5)
        ax2.add_line(real_ppf_line)
        ax2.annotate("peak - %d" % (max_x2), (max_x2, max_y2 / 10))



        #Get the pdf of the model
        #Plot the data in the down-right sub-figure
        x3, y3, max_x3, max_y3 = self.pdf(self.cut + 1, 750)
        ax3.plot(x3, y3)

        #Mark the peak
        model_ppf_line = mlines.Line2D([max_x3, max_x3], [0, max_y3], lw=2., alpha=0.5)
        ax3.add_line(model_ppf_line)
        ax3.annotate("peak - %d" % (max_x3), (max_x3, max_y3 / 10))

        plt.suptitle(info)
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
        """
        Estimate the normal-student-guassian model with two points,
        (s1, c1), (s2, c2)
        Low-score proportion will be substracted from the total number
        """
        self.a_miu, self.a_sig = self.all_score_estimate(self.s1, self.c1, self.s2, self.c2, self.total * (1 - self.low_score_students_proportion))

        """
        Estimate the high-score-students-guassian mode with two points,
        (s1+correction, 0.9 * total-high-score-students-number), (highest-score, 1)
        """
        self.h_miu, self.h_sig = self.high_score_estimate(self.s1 - self.high_start_corr, self.highest_score, self.total * self.high_score_students_proportion)
        self.sample_score_to_rank_table()



    def sample(self, num, lo, hi):
        """
        Generate samples from range(lo, hi) using the model
        :param num:     Total number of sample points
        :param lo:      Lowest bound
        :param hi:      Highest bound
        :return:
        """
        samples = []
        for score in range(lo, hi):
            #Calculate the count of the given score
            #Substract the rank of (score - 1) from the rank of (score)
            cnt = (self.estimate_rank(score - 1) - self.estimate_rank(score)) * num / self.total

            for i in range(cnt):
                samples.append(score)
        return samples



    """--------------Check parameters------------------"""
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


    """--------------Predict score/rank data-----------"""
    def sample_score_to_rank_table(self):
        """
        Create the score-to-rank table and the score-to-proportion table
        Return the score-to-rank table
        The total is optional
        """

        if self.check_model() is not True:
            print "Models are not estimated yet!"
            return

        score_to_rank_table = {}
        self.prop_table = {}
        for score in range(self.cut, 751):
            score_to_rank_table[score] = self.estimate_rank(score)

        return score_to_rank_table




    def samplesss(self):
        return 0

    def error_rate(self, lo, hi):
        """
        Calculating the error rate for each data point in given range
        :param lo   lowest bound
        :param hi   highest bound
        """

        x = []
        y = []

        err_rate = 0
        err_cnt = 0

        for score in range(lo, hi):
            if self.data.has_key(score):

                real_rank = self.data[score][1]
                estimate_rank = self.estimate_rank(score)

                #Calculate the error rate using folloing formula
                # (monitoring rank) / (real rank) - 1
                err = (estimate_rank - real_rank) / float(real_rank)
                x.append(score)
                y.append(err)

                #Cumulative error rate
                err_rate += err
                err_cnt += 1

        #Calculate the average error rate
        if err_cnt != 0:
            err_rate = err_rate / err_cnt
        else:
            err_rate = 0

        return x, y, err_rate

    def pdf(self, lo, hi):
        """
        PDF of the model in given score range
        The pdf of the model is the ppf of a set of discrete data points(score)
        :param lo
        :param hi
        """
        x = []
        y = []

        max_ppf = 0
        max_point = 0

        total = self.estimate_rank(lo) - self.estimate_rank(hi)
        for score in range(lo, hi):
            cnt = self.estimate_rank(score - 1) - self.estimate_rank(score)
            ppf = cnt / (float) (total)

            x.append(score)
            y.append(ppf)

            if ppf > max_ppf:
                max_ppf = ppf
                max_point = score


        return x, y, max_point, max_ppf

    def real_pdf(self, lo, hi):
        """
        The pdf of the real data
        """
        x = []
        y = []

        low_rank = self.total
        if self.data.has_key(lo):
            low_rank = self.data[lo][1]
        hi_rank = 0
        if self.data.has_key(hi):
            low_rank = self.data[hi][1]

        total = low_rank - hi_rank
        max_ppf = 0
        max_pos = 0
        for score in range(lo, hi):
            ppf = 0
            if self.data.has_key(score):
                ppf = self.data[score][0] / (float) (total)
            if ppf != 0:
                x.append(score)
                y.append(ppf)

                if ppf > max_ppf:
                    max_ppf = ppf
                    max_pos = score

        #plt.plot(x, y)
        #plt.show()

        return x, y, max_pos, max_ppf

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

        if rank < self.prop_table[(int)(self.h_miu + self.model_change_score_corr)] * total:
            return (int) (erfinv(1 - 2 * rank / float(total * self.high_score_students_proportion)) * sqrt(2) * self.h_sig + self.h_miu)
        else:
            lo = self.cut
            hi = self.prop_table[(int)(self.h_miu + self.model_change_score_corr)] * total

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

        return (int) (self.prop_table[score] * total)


    def err_rate_span(self, lo, hi):
        """
        Given the range of score, calculate the maximum span of the error rates of the score in the range,
        which is the (maximum error rate) - (minimum error rate)
        """
        max_rate = 0
        min_rate = 0

        err_rate = 0
        err_cnt = 0


        for score in range(lo, hi):
            if self.data.has_key(score):

                real_rank = self.data[score][1]
                estimate_rank = self.estimate_rank(score)

                err = (estimate_rank - real_rank) / float(real_rank)

                if err > max_rate:
                    max_rate = err
                if err < min_rate:
                    min_rate = err

                err_rate += err
                err_cnt += 1


        if err_cnt != 0:
            err_rate = err_rate / err_cnt
        else:
            err_rate = 0

        err_span = max_rate - min_rate

        return err_span, err_rate

    def adjust(self, a_miu_corr, a_sig_corr, h_miu_corr, h_sig_corr):
        self.a_miu += a_miu_corr
        self.a_sig += a_sig_corr

        self.h_miu += h_miu_corr
        self.h_sig += h_sig_corr

        self.prepare_estimate()

def bath_process(s1, c1, s2, c2, total, filpath, h_total_prop_range, l_total_prop_range, h_stop_corr_range, highest_score_range, left_high_portion_range, hi_start_corr_range):
    """
    Process a batch of parameters, find the parameters that meet the requirements
    """
    min_rate = 10
    min_parameters = []
    low_error_parameters = []
    for high_score_students_proportion in h_total_prop_range:
        for low_score_students_proportion in l_total_prop_range:
            for model_change_score_corr in h_stop_corr_range:
                for highest_score in highest_score_range:
                    for left_high_portion in left_high_portion_range:
                        for high_start_corr in hi_start_corr_range:

                            model = GKModel(s1, c1, s2, c2, total, low_cut=200, high_score_students_proportion=high_score_students_proportion, low_score_students_proportion=low_score_students_proportion,  model_change_score_corr=model_change_score_corr, highest_score=highest_score, left_high_portion = left_high_portion, high_start_corr=high_start_corr)
                            model.fit(filpath,delimiter='\t')

                            err_span, err_rate = model.err_rate_span(300, 750)
                            print err_span, '\r',

                            if err_rate < min_rate:
                                min_rate = err_rate
                                min_parameters = [high_score_students_proportion, low_score_students_proportion, model_change_score_corr, highest_score, left_high_portion, high_start_corr]

                            if err_span < 0.5:
                                print ""
                                low_error_parameters.append([err_rate, err_span, high_score_students_proportion, low_score_students_proportion, model_change_score_corr, highest_score, left_high_portion, high_start_corr])

    print min_parameters
    print "low error rate models:"
    for low_rate in low_error_parameters:
        print low_rate


if __name__ == '__main__':

    """ ---------------------------2015 山东文科数据----------------------------------"""
    #-----------------初始化模型--------------------#
    #必须给定的参数：
    #一本线， 一本线排名， 二本线，二本线排名 -- (568, 21243, 510, 53556)
    #总人数可以不给出，如果给定，可以直接调用estimate()函数去估计参数，不然必须通过fit()函数去读取文件，同时estimate
    #low_cut是最低分数线，低于此分数线的分数不予考虑，默认值是200
    #h_total_prop           --  高分考生比例，默认值0.08，范围(0, 0.25)
    #l_total_prop           --  低分考生比例，默认值0, 范围(0, 0.2)
    #h_stop_corr            --  切换模型分数线修正值, 修正范围(-100, 100)，默认为高分高斯的miu值，此处设置为-10， 则模型中模型切换的分数线为 h_miu - 10
    #highest_score          --  最高分，默认是700，范围(650, 749)
    #left_high_portion      --  高分高斯模型切换前考生缩放比例，范围[0, 1], 默认值1
    #hi_start_corr          --  高分高斯模型起始分数修正值，范围(-50, 50), 默认值0
    #model = GKModel(572, 68820, 460, 173685, 263447, low_cut=200, h_total_prop=0.04, l_total_prop=0.04,  h_stop_corr=40, highest_score=705, left_high_portion = 0.45, hi_start_corr=-30)

    """Set the ranges of the paraemters"""
    """
    h_total_prop_range = np.linspace(0.05, 0.1, 3)

    #l_total_prop_range = np.linspace(0.01, 0.2, 10)
    l_total_prop_range = [0.06]

    #h_stop_corr_range = np.linspace(-10, 50, 60)
    h_stop_corr_range = [35, 40, 45]

    highest_score_range = np.linspace(700, 740, 5)

    left_high_portion_range = np.linspace(0, 1, 5)

    hi_start_corr_range = np.linspace(-10, 30, 5)

    bath_process(572, 68820, 460, 173685, 263447, h_total_prop_range, l_total_prop_range, h_stop_corr_range, highest_score_range, left_high_portion_range, hi_start_corr_range)
    """

    """
    model = GKModel(572, 68820, 460, 173685, 263447, low_cut=200, high_score_students_proportion=0.05, low_score_students_proportion=0.06,  model_change_score_corr=40, highest_score=720, left_high_portion = 0.5, high_start_corr=-10)
    #Fit数据，读取数据，重新计算总人数，fit完后可以调用show_model()函数输出图表
    model.fit("./data/sd-2014-l.txt",delimiter='\t')
    #model.re_fit(562, 80062, 490, 150651, 257728)
    model.show_model()
    #给定一组分数，输出模拟排名
    print model.estimate_rank_for_scores([400, 450, 500, 550, 600, 620, 660, 680, 700])
    #输出一分一档表
    print model.sample_score_to_rank_table()
    print model.err_rate_span(300, 750)
    """


    """
    广西2015理科数据
    """

    """Set the ranges of the paraemters"""

    """
    high_score_students_proportion_range = np.linspace(0.01, 0.03, 3)

    #l_total_prop_range = np.linspace(0.01, 0.2, 10)
    low_score_students_proportion_range = [0.06]

    #h_stop_corr_range = np.linspace(-10, 50, 60)
    model_change_score_corr = np.linspace(10, 50, 5)

    highest_score_range = np.linspace(700, 740, 5)

    left_high_portion_range = np.linspace(0, 1, 5)

    high_start_corr_range = np.linspace(-10, 30, 5)


    bath_process(480, 25476, 320, 98296, 145318, "./data/gx-2015-l.txt", high_score_students_proportion_range, low_score_students_proportion_range, model_change_score_corr, highest_score_range, left_high_portion_range, high_start_corr_range)
    """
    """
    model = GKModel(480, 25476, 320, 98296, 145318, low_cut=200, high_score_students_proportion=0.03, low_score_students_proportion=0.06,  model_change_score_corr=50, highest_score=700, left_high_portion = 0, high_start_corr=10)
    #Fit数据，读取数据，重新计算总人数，fit完后可以调用show_model()函数输出图表
    model.fit("./data/gx-2015-l.txt",delimiter='\t')
    #model.re_fit(562, 80062, 490, 150651, 257728)
    model.show_model()
    """


    """
    广西2014理科数据
    """

    """Set the ranges of the paraemters"""

    """
    high_score_students_proportion_range = np.linspace(0.01, 0.03, 3)
    #high_score_students_proportion_range = [0.03]
    low_score_students_proportion_range = np.linspace(0.01, 0.2, 5)
    #low_score_students_proportion_range = [0.06]

    #h_stop_corr_range = np.linspace(-10, 50, 60)
    model_change_score_corr = np.linspace(10, 50, 5)

    highest_score_range = np.linspace(700, 730, 3)

    left_high_portion_range = np.linspace(0, 1, 5)

    high_start_corr_range = np.linspace(-10, 30, 5)


    bath_process(520, 23763, 407, 72210, 144019, "./data/gx-2014-l.txt", high_score_students_proportion_range, low_score_students_proportion_range, model_change_score_corr, highest_score_range, left_high_portion_range, high_start_corr_range)


    """
    """
    #model = GKModel(520, 23763, 407, 72210, 157000, low_cut=200, high_score_students_proportion=0.02, low_score_students_proportion=0.2,  model_change_score_corr=30, highest_score=710, left_high_portion = 0.5, high_start_corr=20)
    model = GKModel(520, 23763, 407, 72210, 144019, low_cut=200, high_score_students_proportion=0.25, low_score_students_proportion=0.05,  model_change_score_corr=-25, highest_score=700, left_high_portion = 0.1, high_start_corr=30)
    #Fit数据，读取数据，重新计算总人数，fit完后可以调用show_model()函数输出图表min_rate
    model.fit("./data/gx-2015-l.txt",delimiter='\t')
    model.adjust(-50, -5, -60, 20)
    #model.show_data(200, 200, 750)
    print model.err_rate_span(300, 750)
    #model.re_fit(562, 80062, 490, 150651, 257728)
    model.show_model()
    """




    """ ---------------------------2015 山东文科数据----------------------------------"""
    #-----------------初始化模型--------------------#
    #必须给定的参数：
    #一本线， 一本线排名， 二本线，二本线排名 -- (568, 21243, 510, 53556)
    #总人数可以不给出，如果给定，可以直接调用estimate()函数去估计参数，不然必须通过fit()函数去读取文件，同时estimate
    #low_cut是最低分数线，低于此分数线的分数不予考虑，默认值是200
    #h_total_prop           --  高分考生比例，默认值0.08，范围(0, 0.25)
    #l_total_prop           --  低分考生比例，默认值0, 范围(0, 0.2)
    #h_stop_corr            --  切换模型分数线修正值, 修正范围(-100, 100)，默认为高分高斯的miu值，此处设置为-10， 则模型中模型切换的分数线为 h_miu - 10
    #highest_score          --  最高分，默认是700，范围(650, 749)
    #left_high_portion      --  高分高斯模型切换前考生缩放比例，范围[0, 1], 默认值1
    #hi_start_corr          --  高分高斯模型起始分数修正值，范围(-50, 50), 默认值0
    #model = GKModel(572, 68820, 460, 173685, 263447, low_cut=200, h_total_prop=0.04, l_total_prop=0.04,  h_stop_corr=40, highest_score=705, left_high_portion = 0.45, hi_start_corr=-30)

    """Set the ranges of the paraemters"""
    """
    h_total_prop_range = np.linspace(0.05, 0.1, 3)

    #l_total_prop_range = np.linspace(0.01, 0.2, 10)
    l_total_prop_range = [0.06]

    #h_stop_corr_range = np.linspace(-10, 50, 60)
    h_stop_corr_range = [35, 40, 45]

    highest_score_range = np.linspace(700, 740, 5)

    left_high_portion_range = np.linspace(0, 1, 5)

    hi_start_corr_range = np.linspace(-10, 30, 5)

    bath_process(572, 68820, 460, 173685, 263447, h_total_prop_range, l_total_prop_range, h_stop_corr_range, highest_score_range, left_high_portion_range, hi_start_corr_range)
    """


    model = GKModel(572, 68820, 460, 173685, 263447, low_cut=200, high_score_students_proportion=0.05, low_score_students_proportion=0.06,  model_change_score_corr=40, highest_score=720, left_high_portion = 0.5, high_start_corr=-10)
    #Fit数据，读取数据，重新计算总人数，fit完后可以调用show_model()函数输出图表
    model.fit("./data/sd-2014-l.txt",delimiter='\t')
    #model.re_fit(562, 80062, 490, 150651, 257728)
    model.show_model()
    #给定一组分数，输出模拟排名
    print model.estimate_rank_for_scores([400, 450, 500, 550, 600, 620, 660, 680, 700])
    #输出一分一档表
    print model.sample_score_to_rank_table()
    print model.err_rate_span(300, 750)
