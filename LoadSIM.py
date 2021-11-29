'''
   This program is to load .mat data

   Code author: Shiping Wang
   Email: shipingwangphd@163.com,
   Date: July 24, 2019.

 '''

import scipy.io

### Load data with .mat
def loadSIM(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    similaritis = data['W']
    #print("The size of data matrix is: ", features.shape)
    return similaritis


if __name__ == '__main__':
    # Step 1: load data
    features, gnd = loadSIM('./data/ALOI.mat')
    print("The size of data matrix is: ", features.shape)
    features1 = features[0][3]
    print(features1.shape)
    gnd = gnd.flatten()
    print("The size of data label is: ", gnd.shape)
