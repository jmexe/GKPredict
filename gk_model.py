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
    def __init__(self, s1, c1, s2, c2, total=-1, low_cut=200, average_guassian_pos_corr=0, average_guassian_shape_corr=0, high_guassian_pos_corr=0, high_guassian_shape_corr=0, model_change_score=600, high_score_proportion=0.05, low_score_proportion=0, high_guassian_left_coef=0):
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
        self.low_score_proportion = low_score_proportion
        self.high_score_proportion = high_score_proportion
        self.model_change_score = model_change_score

        self.high_guassian_left_coef = high_guassian_left_coef

        #Correction parameters
        self.average_guassian_pos_corr = average_guassian_pos_corr
        self.average_guassian_shape_corr = average_guassian_shape_corr
        self.high_guassian_pos_corr = high_guassian_pos_corr
        self.high_guassian_shape_corr = high_guassian_shape_corr

        self.prepared = False

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
        self.prepare_estimate()


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
        portion_in_normal_students = (1 - self.a_norm.cdf(score) / self.a_norm.cdf(self.model_change_score))
        a_rank = (int) (self.total * (1 - self.low_score_proportion - self.high_score_proportion) * portion_in_normal_students)

        #estimate count using high score guassian

        portion_in_left_part_high_guassian = self.h_norm.cdf(self.model_change_score) - self.h_norm.cdf(score)
        h_rank = (int) (portion_in_left_part_high_guassian * self.left_high_total)

        return a_rank + h_rank + self.right_high_total

    def estimate_higscore_score_rank(self, score):
        """
        Give score, estimate the rank using the high-score guassian model.
        :param score: score
        :return: rank and the proportion of the rank
        """

        higher_part_portion_in_high_guassian = (1 - self.h_norm.cdf(score)) / (1 - self.h_norm.cdf(self.model_change_score))
        rank = (int) (self.right_high_total * higher_part_portion_in_high_guassian)

        return rank

    def prepare_estimate(self):
        """
        Preparation function, used for estimate the students count for each score range
        :return:
        """

        self.h_norm = norm(self.h_miu + self.high_guassian_pos_corr, self.h_sig + self.high_guassian_shape_corr)
        self.a_norm = norm(self.a_miu + self.average_guassian_pos_corr, self.a_sig + self.average_guassian_shape_corr)

        self.left_high_total = (int) (self.high_guassian_left_coef * self.high_score_proportion * self.total) * (self.h_norm.cdf(self.model_change_score))
        self.right_high_total = (int)(self.total * self.high_score_proportion) - self.left_high_total

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
        if score > self.model_change_score:
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


        info = "high_score_proportion=" + str(self.high_score_proportion) + " low_score_proportion=" + str(self.low_score_proportion) + " high_guassian_shape_corr=" + str(self.high_guassian_shape_corr) + \
               "\r\nhigh_guassian_pos_corr=" + str(self.high_guassian_pos_corr) + "average_guassian_pos_corr=" + str(self.average_guassian_pos_corr) + "average_guassian_shape_corr=" + str(self.average_guassian_shape_corr) +\
               "model_change_score=" + str(self.model_change_score)
        x0, y0, err_rate = self.error_rate(self.cut + 1, 750)

        max_rate = max(y0)
        min_rate = min(y0)

        #Mark the peak
        model_change_line0  = mlines.Line2D([self.model_change_score, self.model_change_score],                                         [min_rate, max_rate], lw=2., alpha=0.5, color='r')
        hi_miu_line0        = mlines.Line2D([self.a_miu + self.average_guassian_pos_corr, self.a_miu + self.average_guassian_pos_corr], [min_rate, max_rate], lw=2., alpha=0.5, color='g')
        a_miu_line0         = mlines.Line2D([self.h_miu + self.high_guassian_pos_corr, self.h_miu + self.high_guassian_pos_corr],       [min_rate, max_rate], lw=2., alpha=0.5, color='b')
        ax0.add_line(model_change_line0)
        ax0.add_line(hi_miu_line0)
        ax0.add_line(a_miu_line0)

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

        model_change_line1  = mlines.Line2D([self.model_change_score, self.model_change_score],                                         [0, 1], lw=2., alpha=0.5, color='r')
        hi_miu_line1        = mlines.Line2D([self.a_miu + self.average_guassian_pos_corr, self.a_miu + self.average_guassian_pos_corr], [0, 1], lw=2., alpha=0.5, color='g')
        a_miu_line1         = mlines.Line2D([self.h_miu + self.high_guassian_pos_corr, self.h_miu + self.high_guassian_pos_corr],       [0, 1], lw=2., alpha=0.5, color='b')
        ax1.add_line(model_change_line1)
        ax1.add_line(hi_miu_line1)
        ax1.add_line(a_miu_line1)


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


        model_change_line3  = mlines.Line2D([self.model_change_score, self.model_change_score],                                         [0, max_y3], lw=2., alpha=0.5, color='r')
        hi_miu_line3        = mlines.Line2D([self.a_miu + self.average_guassian_pos_corr, self.a_miu + self.average_guassian_pos_corr], [0, max_y3], lw=2., alpha=0.5, color='g')
        a_miu_line3         = mlines.Line2D([self.h_miu + self.high_guassian_pos_corr, self.h_miu + self.high_guassian_pos_corr],       [0, max_y3], lw=2., alpha=0.5, color='b')
        model_ppf_line      = mlines.Line2D([max_x3, max_x3],                                                                           [0, max_y3], lw=2., alpha=0.5, color='y')
        #Mark the peak

        ax3.add_line(model_ppf_line)
        ax3.add_line(model_change_line3)
        ax3.add_line(hi_miu_line3)
        ax3.add_line(a_miu_line3)


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
        self.a_miu, self.a_sig = self.all_score_estimate(self.s1, self.c1, self.s2, self.c2, self.total * (1 - self.low_score_proportion))

        """
        Estimate the high-score-students-guassian mode with two points,
        (s1+correction, 0.9 * total-high-score-students-number), (highest-score, 1)
        """
        self.h_miu, self.h_sig = self.high_score_estimate(self.s1, 750, self.total * self.high_score_proportion)
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

    #TODO Rank-to-score estimate
    """
    def get_rank_to_score(self, rank, total=-1):

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

        if rank < self.prop_table[self.model_change_score] * total:
            return (int) (erfinv(1 - 2 * rank / float(total * self.high_score_proportion)) * sqrt(2) * self.h_sig + self.h_miu)
        else:
            lo = self.cut
            hi = self.prop_table[self.model_change_score] * total

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
    """


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

    def show_prmo_model(self):
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

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

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

        ax1.grid(True)

        model_change_line1  = mlines.Line2D([self.model_change_score, self.model_change_score],                                         [0, 1], lw=2., alpha=0.5, color='r')
        hi_miu_line1        = mlines.Line2D([self.a_miu + self.average_guassian_pos_corr, self.a_miu + self.average_guassian_pos_corr], [0, 1], lw=2., alpha=0.5, color='g')
        a_miu_line1         = mlines.Line2D([self.h_miu + self.high_guassian_pos_corr, self.h_miu + self.high_guassian_pos_corr],       [0, 1], lw=2., alpha=0.5, color='b')
        ax1.add_line(model_change_line1)
        ax1.add_line(hi_miu_line1)
        ax1.add_line(a_miu_line1)


        #Get the pdf of the real data
        #Plot the data in the down-right sub-figure
        x21, y21, max_x2, max_y2 = self.real_pdf(self.cut + 1, 750)

        x22, y22, max_x3, max_y3 = self.pdf(self.cut + 1, 750)

        ax2.plot(x21, y21, color='r')
        ax2.plot(x22, y22, color='b')

        plt.show()

def auto_parameter(s1, c1, s2, c2, filpath, change_line_auto_range=[500, 600, 5], low_score_proportion_auto_range=[0, 0.02, 3],
                   average_guassian_pos_auto_range=[-30, 30, 5], average_guassian_shape_corr_auto_range=[-20, 20, 1],
                   high_guassian_pos_corr_auto_range=[-30, 30, 5], high_guassian_shape_corr_auto_range=[-20, 20, 2]):
    file = open(filpath)

    data = {}
    acc_cnt = 0
    for line in file:
        #Split the row
        split = line.strip().split("\t")
        #Get the score(1st column)
        score = (int)(split[0])
        #Accumulate the count of students
        acc_cnt += (int)(split[1])
        #Add the count and score to the dictionary
        data[(int)(split[0])] = [(int)(split[1]), acc_cnt]

    total = acc_cnt


    model_change_scores = np.arange(change_line_auto_range[0], change_line_auto_range[1], change_line_auto_range[2])

    for model_change_score in model_change_scores:
        if data.has_key(model_change_score):
            high_score_proportion = model_change_score / (float)(total)
            cnt = 0
            #Finding Left
            low_score_proportion_range          = np.linspace(  low_score_proportion_auto_range[0],         low_score_proportion_auto_range[1],         low_score_proportion_auto_range[2])
            average_guassian_pos_corr_range     = np.arange(    average_guassian_pos_auto_range[0],         average_guassian_pos_auto_range[1],         average_guassian_pos_auto_range[2])
            average_guassian_shape_corr_range   = np.arange(    average_guassian_shape_corr_auto_range[0],  average_guassian_shape_corr_auto_range[1],  average_guassian_shape_corr_auto_range[2])

            lowest_span = 10
            lowest_parameter = {}
            for low_score_proportion in low_score_proportion_range:
                for average_guassian_pos_corr in average_guassian_pos_corr_range:
                    for average_guassian_shape_corr in average_guassian_shape_corr_range:

                        print 'left-',cnt, '\r',
                        cnt += 1

                        model = GKModel(s1, c1, s2, c2, total, low_score_proportion=low_score_proportion, high_score_proportion=high_score_proportion, model_change_score=model_change_score,
                                        average_guassian_pos_corr=average_guassian_pos_corr, average_guassian_shape_corr=average_guassian_shape_corr)
                        model.fit(filpath,delimiter='\t')
                        err_span, err_rate = model.err_rate_span(300, model.model_change_score)

                        if err_span < lowest_span:
                            lowest_span = err_span
                            lowest_parameter["low_span"]                    = lowest_span
                            lowest_parameter["model_change_score"]          = model_change_score
                            lowest_parameter["high_score_proportion"]       = high_score_proportion
                            lowest_parameter["low_score_proportion"]        = low_score_proportion
                            lowest_parameter["average_guassian_pos_corr"]   = average_guassian_pos_corr
                            lowest_parameter["average_guassian_shape_corr"] = average_guassian_shape_corr

            cnt = 0
            #Finding right
            high_guassian_pos_corr_range    = np.arange(high_guassian_pos_corr_auto_range[0],   high_guassian_pos_corr_auto_range[1], high_guassian_pos_corr_auto_range[2])
            high_guassian_shape_corr_range  = np.arange(high_guassian_shape_corr_auto_range[0], high_guassian_shape_corr_auto_range[1], high_guassian_shape_corr_auto_range[2])
            lowest_span = 10
            for high_guassian_pos_corr in high_guassian_pos_corr_range:
                for high_guassian_shape_corr in high_guassian_shape_corr_range:

                    print 'right-', cnt, '\r',
                    cnt += 1

                    model = GKModel(s1, c1, s2, c2, total, high_score_proportion=high_score_proportion, model_change_score=model_change_score,
                                        high_guassian_pos_corr=high_guassian_pos_corr, high_guassian_shape_corr=high_guassian_shape_corr)
                    model.fit(filpath,delimiter='\t')
                    err_span, err_rate = model.err_rate_span(model.model_change_score, 750)

                    if err_span < lowest_span:
                        lowest_span = err_span
                        lowest_parameter["high_span"]                   = err_span
                        lowest_parameter["high_guassian_pos_corr"]      = high_guassian_pos_corr
                        lowest_parameter["high_guassian_shape_corr"]    = high_guassian_shape_corr

            print lowest_parameter




def bath_process(s1, c1, s2, c2, total, filpath, model_change_score_range, average_guassian_pos_corr_range, average_guassian_shape_corr_range, high_guassian_pos_corr_range, high_guassian_shape_corr_range, high_score_proportion_range, low_score_proportion_range, average_or_high):
    """
    Process a batch of parameters, find the parameters that meet the requirements
    """
    min_rate = 10
    min_parameters = []
    low_error_parameters = []

    if average_or_high == 1:
        for average_guassian_pos_corr in average_guassian_pos_corr_range:
            for average_guassian_shape_corr in average_guassian_shape_corr_range:
                for low_score_proportion in low_score_proportion_range:
                    model = GKModel(s1, c1, s2, c2, total, low_cut=200, model_change_score=model_change_score_range[0], average_guassian_pos_corr=average_guassian_pos_corr, average_guassian_shape_corr=average_guassian_shape_corr,
                                                    high_guassian_pos_corr=high_guassian_pos_corr_range[0], high_guassian_shape_corr=high_guassian_shape_corr_range[0], high_score_proportion=high_score_proportion_range[0], low_score_proportion=low_score_proportion)
                    model.fit(filpath,delimiter='\t')
                    err_span, err_rate = model.err_rate_span(300, model.model_change_score)
                    print err_span, '\r',

                    if err_rate < min_rate:
                        min_rate = err_rate
                        min_parameters = [err_rate, err_span, model_change_score_range[0], average_guassian_pos_corr, average_guassian_shape_corr, high_guassian_pos_corr_range[0], high_guassian_shape_corr_range[0], high_score_proportion_range[0], low_score_proportion]

                    if err_span < 0.1:
                        print ""
                        low_error_parameters.append([err_rate, err_span, model_change_score_range[0], average_guassian_pos_corr, average_guassian_shape_corr, high_guassian_pos_corr_range[0], high_guassian_shape_corr_range[0], high_score_proportion_range[0], low_score_proportion])



    elif average_or_high == 2:
        for high_guassian_pos_corr in high_guassian_pos_corr_range:
            for high_guassian_shape_corr in high_guassian_shape_corr_range:
                model = GKModel(s1, c1, s2, c2, total, low_cut=200, model_change_score=model_change_score_range[0], average_guassian_pos_corr=average_guassian_pos_corr_range[0], average_guassian_shape_corr=average_guassian_shape_corr_range[0],
                                high_guassian_pos_corr=high_guassian_pos_corr, high_guassian_shape_corr=high_guassian_shape_corr, high_score_proportion=high_score_proportion_range[0], low_score_proportion=low_score_proportion_range[0])
                model.fit(filpath,delimiter='\t')

                err_span, err_rate = model.err_rate_span(model.model_change_score, 750)
                print err_span, '\r',

                if err_rate < min_rate:
                    min_rate = err_rate
                    min_parameters = [err_rate, err_span, model_change_score_range[0], average_guassian_pos_corr_range[0], average_guassian_shape_corr_range[0], high_guassian_pos_corr, high_guassian_shape_corr, high_score_proportion_range[0], low_score_proportion_range[0]]

                if err_span < 0.5:
                    print ""
                    low_error_parameters.append([err_rate, err_span, model_change_score_range[0], average_guassian_pos_corr_range[0], average_guassian_shape_corr_range[0], high_guassian_pos_corr, high_guassian_shape_corr, high_score_proportion_range[0], low_score_proportion_range[0]])


    else:
        for model_change_score in model_change_score_range:
            for average_guassian_pos_corr in average_guassian_pos_corr_range:
                for average_guassian_shape_corr in average_guassian_shape_corr_range:
                    for high_guassian_pos_corr in high_guassian_pos_corr_range:
                        for high_guassian_shape_corr in high_guassian_shape_corr_range:
                            for high_score_proportion in high_score_proportion_range:
                                for low_score_proportion in low_score_proportion_range:

                                    model = GKModel(s1, c1, s2, c2, total, low_cut=200, model_change_score=model_change_score, average_guassian_pos_corr=average_guassian_pos_corr, average_guassian_shape_corr=average_guassian_shape_corr,
                                                    high_guassian_pos_corr=high_guassian_pos_corr, high_guassian_shape_corr=high_guassian_shape_corr, high_score_proportion=high_score_proportion, low_score_proportion=low_score_proportion)
                                    model.fit(filpath,delimiter='\t')

                                    err_span, err_rate = model.err_rate_span(300, 750)
                                    print err_span, '\r',

                                    if err_rate < min_rate:
                                        min_rate = err_rate
                                        min_parameters = [err_rate, err_span, model_change_score, average_guassian_pos_corr, average_guassian_shape_corr, high_guassian_pos_corr, high_guassian_shape_corr, high_score_proportion, low_score_proportion]

                                    if err_span < 0.5:
                                        print ""
                                        low_error_parameters.append([err_rate, err_span, model_change_score, average_guassian_pos_corr, average_guassian_shape_corr, high_guassian_pos_corr, high_guassian_shape_corr, high_score_proportion, low_score_proportion])

    print min_parameters
    print "low error rate models:"
    for low_rate in low_error_parameters:
        print low_rate


if __name__ == '__main__':
    """
    自动调参，可以手动设置调参范围

    代码对于每一个模型切换线:
        1、自动找到高分人数比例
        2、在限定范围内找到左边高斯的最优参数
        3、在限定范围内找到右边高斯的最优参数
        4、输出最优的参数组合
    change_line_auto_range                      -- 模型切换线的调节范围
    low_score_proportion_auto_range             -- 低分人数调节范围
    average_guassian_pos_auto_range             -- 左边高斯位置调节范围
    average_guassian_shape_corr_auto_range      -- 左边高斯形状调节范围
    high_guassian_pos_corr_auto_range           -- 右边高斯位置调节范围
    high_guassian_shape_corr_auto_range         -- 右边高斯形状调节范围

    low_score_proportion_auto_range的数组内容为 (范围开始，范围结束，采样个数)
    其余每个一个调解范围需要设定一个数组，数组内容为 [范围开始，范围结束，步长]


    请在设定范围前考虑运算次数:

        假设
            n1 = change_line_auto_range中参数个数
            n2 = low_score_proportion_auto_range
            n3 = average_guassian_pos_auto_range
            n4 = average_guassian_shape_corr_auto_range
            n5 = high_guassian_pos_corr_auto_range
            n6 = high_guassian_shape_corr_auto_range

        代码运算次数
            n1 * (n2 * n3 * n4 + n5 * n6)

    以下是一组建议参数，代码会运行 7 * (3 * 13 * 41 + 13 * 21) = 13104次
    """
    auto_parameter(572, 68820, 460, 173685, "./data/sd-2014-l.txt", change_line_auto_range=[520, 580, 10], low_score_proportion_auto_range=[0, 0.02, 3],
                   average_guassian_pos_auto_range=[-30, 30, 5], average_guassian_shape_corr_auto_range=[-20, 20, 1],
                   high_guassian_pos_corr_auto_range=[-30, 30, 5], high_guassian_shape_corr_auto_range=[-20, 20, 2])