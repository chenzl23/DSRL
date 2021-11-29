import torch
import os
import numpy as np
from loadMatData import loadData
from LoadSIM import loadSIM
import time
import random
from paraparser import parameter_parser_clustering, tab_printer
from DSRL import DSRL


np.set_printoptions(suppress=True)

## parameter parser
args = parameter_parser_clustering()
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


features, gnd = loadData(os.path.join(data_dir,dataset_name+".mat"))
W = loadSIM(os.path.join(dataW_dir,dataset_name+"W.mat"))
nc = np.unique(gnd).shape[0]



model = DSRL(layer, epoch, lr, W, nc, gnd).to(device)
model.para_init()
start = time.perf_counter()
model.train()
elapsed = (time.perf_counter() - start)
print("Time used:", elapsed)
WW = model.get_ans()
print('DSRL Performance:')
print("Sparsity: ", np.sum(WW < 0.01) / (WW.shape[0] * WW.shape[1]))
model.spectral_clustering(WW, nc, repnum=10)


