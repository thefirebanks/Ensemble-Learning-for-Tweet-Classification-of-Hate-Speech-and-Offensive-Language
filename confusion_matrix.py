from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import math
import CEN

class ConfusionMatrix:

    #TODO: Change all the initial lists of the metrics to [0, 0, 0] in case a classifier returns only one result (random)
    #TODO: Also use indexing instead of appending

    def __init__(self, actual, predicted, name):
        self.pred = predicted
        self.actual = actual
        self.number_cm= self.build_cm()
        self.normalized_cm = self.build_normalized_cm()
        self.name = name


    def build_cm(self):
        '''Builds confusion matrix as a 2D numpy array, using number of guesses'''
        cm = metrics.confusion_matrix(self.actual, self.pred)
        #print(cm)
        return cm

    def build_normalized_cm(self):
        '''Builds confusion matrix as a 2D numpy array, using percentage of guesses'''
        p_cm = metrics.confusion_matrix(self.actual, self.pred)
        matrix_proportions = np.zeros((3, 3))
        for i in range(0, 3):
            matrix_proportions[i, :] = p_cm[i, :] / float(p_cm[i, :].sum())

        return matrix_proportions

    def get_CEN_score(self):
        '''Returns CEN score for this confusion matrix'''
        return CEN.calcCEN(self.number_cm)

    def get_number_cm(self):
        '''Returns number confusion matrix'''
        return self.number_cm

    def get_normalized_cm(self):
        '''Returns percentage confusion matrix'''
        return self.normalized_cm

    def get_precision(self):
        '''Returns the percentage of right guesses per each label in a list'''
        precision = [0, 0, 0]

        ncm = self.number_cm

        pred_0 = ncm[0][0] + ncm[1][0] + ncm[2][0]
        pred_1 = ncm[0][1] + ncm[1][1] + ncm[2][1]
        pred_2 = ncm[0][2] + ncm[1][2] + ncm[2][2]

        precision[0] = ncm[0][0] / (pred_0 or not pred_0)
        precision[1] = ncm[1][1] / (pred_1 or not pred_1)
        precision[2] = ncm[2][2] / (pred_2 or not pred_2)

        return precision


    def get_accuracy(self):
        return metrics.accuracy_score(self.actual, self.pred)

    def get_mcc(self):
        '''Returns the Matthew's Correlation Coefficient Score'''

        TP = self.get_true_pos()
        TN = self.get_true_neg()
        FP = self.get_false_pos()
        FN = self.get_false_neg()

        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (FN + TN) * (FP + TN) * (TP + FN))

        return mcc
        #return metrics.matthews_corrcoef(self.actual, self.pred)

    def get_recall(self):
        recall = []
        ncm = self.number_cm

        recall.append(ncm[0][0] / sum(ncm[0]))
        recall.append(ncm[1][1] / sum(ncm[1]))
        recall.append(ncm[2][2] / sum(ncm[2]))

        return recall

    def get_true_pos(self): #Recall
        true_pos = []
        ncm = self.number_cm

        true_pos.append(ncm[0][0])
        true_pos.append(ncm[1][1])
        true_pos.append(ncm[2][2])

        return sum(true_pos)

    def get_true_neg(self):
        true_neg = []
        ncm = self.number_cm

        pred_0 = ncm[1][1] + ncm[1][2] + ncm[2][1] + ncm[2][2]
        pred_1 = ncm[0][0] + ncm[0][2] + ncm[2][0] + ncm[2][2]
        pred_2 = ncm[0][0] + ncm[0][1] + ncm[1][0] + ncm[1][1]

        true_neg.append(pred_0)
        true_neg.append(pred_1)
        true_neg.append(pred_2)

        return sum(true_neg)

    def get_false_pos(self):
        false_pos = []

        ncm = self.number_cm

        pred_0 = ncm[1][0] + ncm[2][0]
        pred_1 = ncm[0][1] + ncm[2][1]
        pred_2 = ncm[0][2] + ncm[1][2]

        false_pos.append(pred_0)
        false_pos.append(pred_1)
        false_pos.append(pred_2)

        return sum(false_pos)

    def get_false_neg(self):
        false_neg = []
        ncm = self.number_cm

        false_neg.append(sum(ncm[0]) - ncm[0][0])
        false_neg.append(sum(ncm[1]) - ncm[1][1])
        false_neg.append(sum(ncm[2]) - ncm[2][2])

        return sum(false_neg)

    def get_name(self):
        return self.name

    def store_cm(self):
        '''Stores confusion matrix in pdf format'''

        names = ['Hate', 'Offensive', 'Neither']
        confusion_df = pd.DataFrame(self.normalized_cm, index=names, columns=names)
        plt.figure(figsize=(5, 5))
        seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', cbar=False, square=True,
                        fmt='.2f')
        plt.ylabel(r'True categories', fontsize=14)
        plt.xlabel(r'Predicted categories', fontsize=14)
        plt.tick_params(labelsize=12)

        # print(cfm)
        # print(matrix_proportions)
        plt.savefig(self.name + "_ConfusionMatrix.pdf")
        print("Stored confusion matrix!")



