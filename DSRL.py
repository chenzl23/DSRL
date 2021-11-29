import torch
from torch import nn
import copy
from tqdm import tqdm
from clusteringPerformance import StatisticClustering
from clusteringPerformance import similarity_function
from numpy import linalg as LA
from sklearn.preprocessing import normalize
import numpy as np

class DSRL(nn.Module):
    def __init__(self, block_num, epoch, lr, W, nc, gnd):
        super(DSRL, self).__init__()
        self.block_num = block_num
        self.epoch = epoch
        self.lr = lr
        self.nc = nc
        self.W = torch.from_numpy(W).float()                             
        self.gnd = gnd

        self.active_num = 2

        init_para = [1,2] 
        self.b = nn.ParameterList([nn.Parameter(torch.FloatTensor([init_para[i]]), requires_grad=True) for i in range (self.active_num)])

        init_w = [1.0 for i in range(self.active_num)] 
        self.w = nn.ParameterList([nn.Parameter(torch.FloatTensor([init_w[i]]), requires_grad=True) for i in range (self.active_num)])


    def para_init(self):
        self.L = nn.Parameter(torch.FloatTensor([1.0]))
        
    def activation(self, x, w, b):
        mask_1 = (x >= b[1]).float()
        mask_2 = ((x < b[1]) * (x >= b[0])).float()
        mask_3 = ((x < b[0]) * (x >= (-b[0]))).float()
        mask_4 = ((x >= (-b[1])) * (x < (-b[0]))).float()
        mask_5 = (x < (-b[1])).float()

        new_x_1 = ((w[1] *  (x - b[1]) + w[0] * (b[1] - b[0]))) * mask_1
        new_x_2 = (w[0] *  (x - b[0])) * mask_2
        new_x_3 = 0 * mask_3
        new_x_4 = (w[0] *  (x + b[0])) * mask_4
        new_x_5 = (w[1] *  (x + b[1]) + w[0] * (b[0] - b[1])) * mask_5
        return new_x_1 + new_x_2 + new_x_3 + new_x_4 + new_x_5

    def my_loss1(self, x, pred_x):
        return 0.5 * (torch.norm(x - pred_x) ** 2)

    def projection_b(self, w, b):
        return_b = torch.FloatTensor([0,0])
        if w[0]>0 and w[0]<=1:
            if (b[1] >= b[0]) and (b[0]>=0) and (b[1]>=0):
                return_b[0] = b[0]
                return_b[1] = b[1]
            elif (b[0] < 0) and (b[1] > 0):
                return_b[0] = 0
                return_b[1] = b[1]
            elif (b[1] <= min(0, -b[0])):
                return_b[0] = 0
                return_b[1] = 0
            elif (b[0] >= abs(b[1])):
                return_b[0] = (b[0] + b[1]) / 2
                return_b[1] = (b[0] + b[1]) / 2 
        else:
            denominator = (w[0] * w[0]) + ((w[0]-1) * (w[0]-1))
            xi = [((w[0]-1)*(w[0]-1))/denominator, (w[0]*(w[0]-1))/denominator, (w[0] * w[0])/denominator]
            if  (b[1]>=0) and (b[1] >= b[0]) and (b[0] >= ( ((w[0]-1)/w[0]) * b[1]) ):
                return_b[0] = b[0]
                return_b[1] = b[1]
            elif (b[0] < ( ((w[0]-1)/w[0])*b[1] )) and ( b[0] > ( (w[0]/(1-w[0]))*b[1] ) ):
                return_b[0] = xi[0] * b[0] + xi[1] * b[1]
                return_b[1] = xi[1] * b[0] + xi[2] * b[1]
            elif ( b[0] <= (w[0]/(1-w[0]))*b[1] ) and (b[1] >= 0):
                return_b[0] = 0
                return_b[1] = 0
            elif (b[1] <= min(0, -b[0])):  
                return_b[0] = 0
                return_b[1] = 0
            elif (b[0] >= abs(b[1])):
                return_b[0] = (b[0] + b[1]) / 2
                return_b[1] = (b[0] + b[1]) / 2 
        return return_b

    def projection_w(self, w):
        return_w = torch.FloatTensor([0,0])
        for i in range(self.active_num):
            if w[i] > 1e-4:
                return_w[i] = w[i]
            else:
                return_w[i] = 1e-4
        if return_w[1] > 1:
            return_w[1] = 1
        return return_w

    def forward(self, W):
        W_list = []
        cur_W = copy.deepcopy(W)
        projected_w =self.projection_w(self.w)
        projected_b =self.projection_b(projected_w, self.b)
        cur_W = self.activation(cur_W, projected_w, projected_b)
        W_list.append(cur_W)

        for i in range(self.block_num):
            cur_W = W_list[-1] - ((W_list[-1] - W) / self.L)
            cur_W = self.activation(cur_W, projected_w, projected_b)
            W_list.append(cur_W)
        return W_list

    def train(self):
        self.loss_list = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.90, 0.92), weight_decay=0.15)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3, verbose = True, min_lr=1e-6) 
        with tqdm(total=self.epoch, desc="Training") as pbar:
            for epoch_id in range(self.epoch):
                W_list = self(self.W)
                loss = self.my_loss1(W_list[-1], self.W)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = loss.cpu().detach().numpy()
                if  len(train_loss.shape) == 1:
                    train_loss = train_loss[0]
                scheduler.step(loss)
                self.loss_list.append(train_loss)
                if (optimizer.param_groups[0]['lr'] <= 2e-7):
                    print("early stopped")
                    break
                pbar.update(1)


    # Refactoring W
    def get_ans(self):
        W_list = self(self.W)
        return W_list[-1].cpu().detach().numpy()


    def spectral_clustering(self, points, k, repnum=10):
        W = similarity_function(points)
        Dn = np.diag(1 / np.power(np.sum(W, axis=1), -0.5))
        L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
        eigvals, eigvecs = LA.eig(L)
        eigvecs = eigvecs.astype(float)
        indices = np.argsort(eigvals)[:k]
        k_smallest_eigenvectors = normalize(eigvecs[:, indices])

        [ACC, NMI, ARI] = StatisticClustering(k_smallest_eigenvectors, self.gnd, k, repnum)
        print("ACC, NMI, ARI: ", ACC, NMI, ARI)
        return [ACC, NMI, ARI]
