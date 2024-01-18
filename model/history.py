import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Sequential, LogSoftmax
import numpy as np
from torch_geometric.utils import to_dense_adj,dense_to_sparse


class SATA(Module):
    def __init__(self, n, nclass, nfeat, nlayer, lambda_1, lambda_2, dropout,adj,blancealpha):
        super(SATA, self).__init__()
        self.n = n
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.nclass = nclass
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.dropout = dropout
        self.w1 = Parameter(torch.FloatTensor(self.n, self.n), requires_grad=True)
        self.EA = Parameter(torch.FloatTensor(self.n, self.n), requires_grad=True)
        self.w22= torch.ones(self.n, self.n).cuda()
        # self.s= Parameter(torch.ones(self.n, self.n), requires_grad=True)
        self.alpha =  Parameter(blancealpha*torch.ones(1), requires_grad=False)
        self.w2 = Sequential(Linear(2 * nfeat, nclass), LogSoftmax(dim=1))
        self.params1 = [self.w1,self.alpha]
        self.params12 = [self.EA]
        self.params2 = list(self.w2.parameters())
        self.laplacian = None
        self.reset_parameter()
        indic =adj.coalesce().indices()
        self.w22[indic[0],indic[1]]= 0

    def reset_parameter(self):
        nn.init.uniform_(self.w1, 0, 1)
        nn.init.uniform_(self.EA, 0, 1)

    def ans(self,z,adj):
        # M: Tensor = torch.zeros(self.n, self.n).to(adj.device)
        # edgevalue,p =dense_to_sparse(adj)
        # print(edgevalue)
        # print(adj)
        #
        # for i in range(self.n):
        #     for j in range(0,i+1):
        #         if adj[i,j]!=0:
        #          M[i,j]=M[j,i] = adj[i,j]*torch.norm(z[i]-z[j])
        indic =adj.coalesce().indices()
        value = adj.coalesce().values()
        P = z[indic[0]]-z[indic[1]]
        P = torch.norm(P,dim=1)
        P = P * value
        M = torch.sparse_coo_tensor(indices=indic, values=P, size=[adj.shape[0], adj.shape[0]]).to(adj.device)
        return M
    def forward(self, feat, adj):
        # adj1 = torch.mul(adj,adj)
        # adj2 = torch.mul(adj1,adj)
        adj = adj
        # pp = to_dense_adj(edge_index=adj.coalesce().indices(),edge_attr=values).to(adj.device)
        # adj = dense_to_sparse(pp)
        if self.laplacian is None:
            n = adj.shape[0]
            indices = torch.Tensor([list(range(n)), list(range(n))])
            values = torch.FloatTensor([1.0] * n)
            eye = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n]).to(adj.device)
            self.laplacian = eye - adj
        adj1 = adj
        adj = adj.to_dense()
        lap = self.laplacian.to_dense()
        z: Tensor = torch.rand(self.n, self.nfeat).to(adj.device)
        EP: Tensor = torch.ones(self.n, self.n).to(adj.device)
        # t = self.s
        y: Tensor = feat.to(adj.device)
        for i in range(self.nlayer):
            # self.w22 = torch.where(self.alpha * self.w12 > 0.5, self.w22, torch.tensor(0.0).cuda())
            feat = F.dropout(feat, self.dropout, training=self.training)
            # t = torch.mm(self.w12,t)
            temp=torch.mm(EP.t(),y)
            # temp = torch.mm(torch.mm(y,y.t()), y)
            temp = temp+torch.mm(adj*EP+self.alpha*self.EA*self.w22,y)-torch.mm(torch.mm(y,y.t()),y)
            temp = torch.mm(self.w1,temp)
            temp = torch.sigmoid(temp)
            y_n = self.lambda_2/2*temp+feat
            temp = torch.eye(self.n).cuda()-(adj*EP+self.alpha*self.EA*self.w22)
            # temp = torch.sigmoid(temp)
            z_n = torch.spmm(temp, z)
            temp = self.lambda_2*(torch.mm(y,y.t())-self.alpha*self.EA*self.w22)
            m = self.lambda_1*0.5*self.ans(z,adj1).to_dense()
            EP_n = (temp -m)/self.lambda_2


            # feat = F.dropout(feat, self.dropout, training=self.training)
            # # t = torch.mm(self.w12,t)
            # temp=torch.mm(t.t(),y)
            # # temp = torch.mm(torch.mm(y,y.t()), y)
            # temp = temp+torch.mm(self.alpha*adj*t+(1-self.alpha)*self.w12.mul(self.w22),y)-2*torch.mm(torch.mm(y,y.t()),y)
            # temp = torch.mm(self.w1,temp)
            # temp = torch.sigmoid(temp)
            # y_n = self.lambda_2*temp+feat
            # temp = self.alpha*adj*t+(1-self.alpha)*self.w12.mul(self.w22)
            # # temp = torch.sigmoid(temp)
            # z_n = torch.spmm(temp, z)
            # temp = torch.mm(y,y.t())-(1-self.alpha)*self.w12.mul(self.w22)
            # temp = self.lambda_2*temp
            # m = 2*self.lambda_1*0.5*self.ans(z,adj1).to_dense()
            # adj = adj +1e-7
            # t_n = (temp*adj -m)/self.lambda_2/self.alpha/adj/adj/2


            # m = self.lambda_1*0.5*self.ans(z,adj1)
            # p0 = torch.mm(y,y.t())+self.w12.mul(self.w22)
            # p1 = 2*self.lambda_2*adj.mul(p0)-m
            # t_n = p1/self.lambda_2/2/adj
            # t_n = np.nan_to_num(t_n.cpu().detach().numpy(), copy=False)
            # t_n = torch.from_numpy(t_n).cuda()

            y = y_n
            z = z_n
            EP = EP_n
        y = F.normalize(y, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        p = torch.cat((z, y), dim=1)
        p = F.dropout(p, self.dropout, training=self.training)
        return self.w2(p)

        # w21 = self.w12.mul(self.w22)
        # feat = F.dropout(feat, self.dropout, training=self.training)
        # # t = torch.mm(self.w12,t)
        # # temp = torch.mm(torch.mm(y,y.t()), y)
        # # t = torch.mm(self.w12,t)
        # # temp = torch.mm(torch.mm(self.w21,t.t()), y)
        # temp = torch.mm(torch.mm(w21, t), y) - torch.mm(torch.mm(y, y.t()), y)
        # # temp = temp+torch.mm(t,y)-torch.mm(torch.mm(y,y.t()),y)
        # temp = torch.mm(self.w1, temp)
        # temp = torch.sigmoid(temp)
        # y_n = self.lambda_2 * 2 * temp + feat
        # temp = lap * t
        # # temp = torch.sigmoid(temp)
        # z_n = torch.spmm(temp, z)
        # temp = 2 * self.lambda_2 * torch.mm(torch.mm(w21.t(), y), y.t())
        # m = self.lambda_1 * 0.5 * self.ans(z, adj)
        # t_n = (temp - m) / 2 / self.lambda_2 / torch.mm(w21.t(), w21)














        # z: Tensor = torch.rand(self.n, self.nfeat).to(adj.device)
        # y: Tensor = feat
        # for i in range(self.nlayer):
        #     feat = F.dropout(feat, self.dropout, training=self.training)
        #     temp = torch.mm(self.w1, y)
        #     temp = torch.mm(y, temp.t())
        #     temp = torch.sigmoid(temp)
        #     temp = torch.mm(temp, z)
        #     temp = (self.lambda_2 / self.lambda_1) * temp
        #     z_n = torch.spmm(lap, z) - temp
        #     temp = torch.mm(z.t(), y)
        #     temp = torch.mm(z, temp)
        #     temp = torch.sigmoid(temp)
        #     y_n = feat - self.lambda_2 * temp
        #     y = y_n
        #     z = z_n
        # y = F.normalize(y, p=2, dim=1)
        # z = F.normalize(z, p=2, dim=1)
        # p = torch.cat((z, y), dim=1)
        # p = F.dropout(p, self.dropout, training=self.training)
        # return self.w2(p)
