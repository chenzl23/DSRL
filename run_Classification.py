
import torch
import os
import numpy as np
from loadMatData import loadData
from LoadSIM import loadSIM
import math
import time
from sklearn.metrics import accuracy_score
import random
from paraparser import parameter_parser_classification, tab_printer
from DSRL import DSRL

np.set_printoptions(suppress=True)



if __name__ == "__main__":
## parameter parser
    args = parameter_parser_classification()
    tab_printer(args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)

    data_dir = "./_multiview datasets"
    dataW_dir = "./datasetW"

    layer = args.layer
    lr = args.lr
    epoch = args.epoch
    dataset_name = args.dataset_name

    random_ratio = args.ratio

    repeatNum = 10
    
    acc_list = []
    total_time = 0.0

    
    for i in range(repeatNum):
        print("Repeat no.", str(i + 1))
        features, gnd = loadData(os.path.join(data_dir,dataset_name+".mat"))
        W = loadSIM(os.path.join(dataW_dir,dataset_name+"W.mat"))
        gnd = gnd - 1
        
        nc = np.unique(gnd).shape[0] # number of classes
        
        N = gnd.shape[0] # number of samples
        n = math.floor(N*random_ratio) # number of labeled data
        
        # Generate random permutation
        p_all = np.random.permutation(N)
        p = p_all[:n]
        p_test = p_all[n:]
        

        for i in p:
            for j in p:
                if (abs(int(gnd[i]) - int(gnd[j])) < 0.1):
                    W[i,j] = 1

        L = np.zeros([N,nc])
        for i in p:
            L[i][gnd[i]] = 1

        alpha = 1
        beta = 1

        model1 = DSRL(layer, epoch, lr, W, nc, gnd).to(device)
        S = model1.para_init()
        start = time.perf_counter()
        model1.train()
        WW = model1.get_ans()
        print("Sparsity:",np.sum(WW < 0.01) / (WW.shape[0] * WW.shape[1]))
        D = np.diag(np.sum(WW,1))
        I = np.eye(WW.shape[0],WW.shape[1])
        predY = np.linalg.inv(I+(alpha/beta)*(D-WW)).dot(L)
        pred_label = np.argmax(predY, axis=1)
        acc = accuracy_score(gnd[p_test],pred_label[p_test])
        print("Accuracy", acc)
        acc_list.append(acc)
        elapsed = (time.perf_counter() - start)
        total_time += elapsed
        print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("Avg Accuracy:",np.mean(acc_list), "Std:",np.std(acc_list))
    print("Runtime:", total_time/repeatNum)