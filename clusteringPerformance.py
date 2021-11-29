'''
   This program is to evaluate clustering performance

   Code author: Shide Du
   Email: shidedums@163.com
   Date: Dec 4, 2019.
'''

import scipy.io as scio
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy.stats import mode
import numpy as np
from loadMatData import loadData
import torch
from hangarian import Hungarian
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
import warnings
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.utils.linear_assignment_ import linear_assignment

warnings.filterwarnings("ignore")


### K-means clustering
def KMeansClustering(features, gnd, clusterNum, randNum):
    kmeans = KMeans(n_clusters=clusterNum, n_init=1, max_iter=500,
                    random_state=randNum)
    estimator = kmeans.fit(features)
    clusters = estimator.labels_
    label_pred = estimator.labels_  

    labels = np.zeros_like(clusters)
    for i in range(clusterNum):
        mask = (clusters == i)
        labels[mask] = mode(gnd[mask])[0]
    # Return the preditive clustering label
    return labels


def similarity_function(points):
    """

    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def cluster_f(y_true, y_pred):
    N = len(y_true)
    numT = 0
    numH = 0
    numI = 0
    for n in range(0, N):
        C1 = [y_true[n] for x in range(1, N - n)]
        C1 = np.array(C1)
        C2 = y_true[n + 1:]
        C2 = np.array(C2)
        Tn = (C1 == C2)*1

        C3 = [y_pred[n] for x in range(1, N - n)]
        C3 = np.array(C3)
        C4 = y_pred[n + 1:]
        C4 = np.array(C4)
        Hn = (C3 == C4)*1

        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(np.multiply(Tn, Hn))
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0;
    else:
        f = 2 * p * r / (p + r);
    return f, p, r


def clustering_purity(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    :return: Purity
    """
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins = range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct/len(y_pred)

    return Purity


### Evaluation metrics of clustering performance
def clusteringMetrics(trueLabel, predictiveLabel):
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, predictiveLabel)

    # Normalized mutual information
    NMI = metrics.v_measure_score(trueLabel, predictiveLabel)

    # Purity
    Purity = clustering_purity(trueLabel.reshape((-1, 1)), predictiveLabel.reshape(-1, 1))

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, predictiveLabel)

    # Fscore, Precision, Recall score
    Fscore, Precision, Recall = cluster_f(trueLabel, predictiveLabel)

    return ACC, NMI, Purity, ARI, Fscore, Precision, Recall


### Report mean and std of 10 experiments
def StatisticClustering(features, gnd, clusterNum, repnum):
    ### Input the mean and standard diviation with 10 experiments
    repNum = repnum
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    PurityList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    FscoreList = np.zeros((repNum, 1))
    PrecisionList = np.zeros((repNum, 1))
    RecallList = np.zeros((repNum, 1))

    #clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    # print("cluster number: ", clusterNum)
    for i in range(repNum):
        randNum = random.randint(1,999999)
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, randNum)
        if (i==0):
            data = {}
            data['ind'] = np.array(predictiveLabel)
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, predictiveLabel)
        ACCList[i] = ACC
        NMIList[i] = NMI
        PurityList[i] = Purity
        ARIList[i] = ARI
        FscoreList[i] = Fscore
        PrecisionList[i] = Precision
        RecallList[i] = Recall
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    Puritymean_std = np.around([np.mean(PurityList), np.std(PurityList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    Fscoremean_std = np.around([np.mean(FscoreList), np.std(FscoreList)], decimals=4)
    Precisionmean_std = np.around([np.mean(PrecisionList), np.std(PrecisionList)], decimals=4)
    Recallmean_std = np.around([np.mean(RecallList), np.std(RecallList)], decimals=4)

    return ACCmean_std, NMImean_std, ARImean_std


def StatisticClustering1(features, gnd):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 7
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    # print("cluster number: ", clusterNum)
    for i in range(repNum):
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, i)
        ACC, NMI, ARI = clusteringMetrics(gnd, predictiveLabel)
        ACCList[i] = ACC
        NMIList[i] = NMI
        ARIList[i] = ARI
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    return ACCmean_std, NMImean_std, ARImean_std


### Real entrance to this program
if __name__ == '__main__':
    # Step 1: load data
    features, gnd = loadData('./data/Yale_32x32.mat')
    print("The size of data matrix is: ", features.shape)
    gnd = gnd.flatten()
    print("The size of data label is: ", gnd.shape)
    clusterNum = 10
    # Print clustering results
    [ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std] = StatisticClustering(
        features, gnd)
    print("ACC, NMI, Purity, ARI, Fscore, Precision, Recall: ", ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std)
