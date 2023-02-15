import torch.nn.functional as F
import torch
import torch.nn as nn
import scipy.io as scio

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from Modules import FilterLinear
import math
import numpy as np
import MAHGCN
import torch
from collections import OrderedDict

gpu = 1


class MAHGCNNET(nn.Module):
    def __init__(self,ROInum,layer, num_class=2):
        super(MAHGCNNET, self).__init__()
        self.ROInum=ROInum
        self.layer = layer
        self.paranum=0
        for i in range(100,ROInum+100,100):
            self.paranum=self.paranum+i
        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.MAHGCN = MAHGCN.MAHGCN(nn.ReLU(),0.3,self.layer)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.paranum)
        self.fl1 = nn.Linear(self.paranum,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g1, g2, g3, g4, g5):
        batch_size = g1.shape[0]
        ROInum = self.ROInum

        fea = torch.zeros(batch_size, ROInum, ROInum)
        for s in range(batch_size):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g1 = g1.cuda()
        g2 = g2.cuda()
        g3 = g3.cuda()
        g4 = g4.cuda()
        g5 = g5.cuda()
        out = torch.zeros(batch_size, self.paranum)

        for s in range(batch_size):
            temp = self.MAHGCN(g1[s, :, :], g2[s, :, :], g3[s, :, :],g4[s, :, :], g5[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out

class GCN_base(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_base, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GCN(ROInum, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(ROInum)
        self.fl1 = nn.Linear(ROInum,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, ROInum)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out

class GCN_gpool(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_gpool, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GraphUnet([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out

class GCN_diffpool(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_diffpool, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GraphDiif([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out
    def Loss(self):
        return self.gcn.Loss()

class GCN_SAG(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_SAG, self).__init__()

        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.gcn = MAHGCN.GraphSAG([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out
