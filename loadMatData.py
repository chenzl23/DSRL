'''
   This program is to load .mat data 

   Code author: Shiping Wang
   Email: shipingwangphd@163.com,
   Date: July 24, 2019.

 '''

import scipy.io
import numpy as np

### Load data with .mat
def loadData(data_name):
    data = scipy.io.loadmat(data_name) 
    #print(data.keys()) 
    features = data['X']#.dtype = 'float32'
    gnd = data['Y']
    gnd = gnd.flatten()
    #print("The size of data matrix is: ", features.shape)
    return features, gnd


if __name__ == '__main__':
    # Step 1: load data
    features, gnd = loadData('./_multiview datasets/ALOI.mat')
    print("The size of data matrix is: ", features.shape)
    for i in range(features.shape[1]):
        V = features.shape[1]
        nc = np.unique(gnd).shape[0]
        nc = nc
        print(nc)
        features1 = features[0][i]
        print(features1.shape)
    gnd = gnd.flatten()
    print("The size of data label is: ", gnd.shape)

