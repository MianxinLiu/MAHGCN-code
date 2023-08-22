import torch.nn.functional as F
import torch
import torch.nn as nn

import MAHGCN
from ops import linear
import dsbn
from collections import OrderedDict

gpu = 1

class GCNNET(nn.Module):
    def __init__(self,ROInum, num_class=2):
        super(GCNNET, self).__init__()
        self.ROInum=ROInum
        self.gcn = MAHGCN.GCN(ROInum, 1, nn.ReLU(),0.2)

        self.bn1 = torch.nn.BatchNorm1d(self.ROInum)
        self.fl1 = nn.Linear(self.ROInum,512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fl2 = nn.Linear(512,256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fl3 = nn.Linear(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.fl4 = nn.Linear(128, num_class)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = self.ROInum

        h = torch.zeros(batch_size, ROInum, ROInum)
        for s in range(batch_size):
            h[s,:,:] = torch.eye(ROInum)
        h = h.cuda()
        g = g.cuda()

        h = self.gcn(g,h)
        h = torch.squeeze(h)

        h = self.bn1.cuda()(h)
        h = F.relu(h)
        h = self.fl1(h)
        h = self.bn2.cuda()(h)
        h = F.relu(h)
        h = self.fl2(h)

        h = self.bn3.cuda()(h)
        h = F.relu(h)
        h = self.fl3(h)

        h = self.bn4.cuda()(h)
        h = F.relu(h)
        h = self.fl4(h)

        h = self.softmax(h)

        return h

class MAHGCNNET(nn.Module):
    def __init__(self,ROInum,layer, num_class=2):
        super(MAHGCNNET, self).__init__()
        self.ROInum=ROInum
        self.layer = layer
        self.paranum=0
        for i in range(100,ROInum+100,100):
            self.paranum=self.paranum+i

        self.mgunet = MAHGCN.MAHGCN(nn.ReLU(),0.2,self.layer)

        self.bn1 = torch.nn.BatchNorm1d(self.paranum)
        self.fl1 = nn.Linear(self.paranum,512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fl2 = nn.Linear(512,256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fl3 = nn.Linear(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.fl4 = nn.Linear(128, num_class)

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
            temp = self.mgunet(g1[s, :, :], g2[s, :, :], g3[s, :, :],g4[s, :, :], g5[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)
        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)
        out = self.fl2(out)

        out = self.bn3.cuda()(out)
        out = F.relu(out)
        out = self.fl3(out)

        out = self.bn4.cuda()(out)
        out = F.relu(out)
        out = self.fl4(out)

        out = self.softmax(out)

        return out

class GCNNET_gpool(nn.Module):
    def __init__(self,ROInum, num_class=2):
        super(GCNNET_gpool, self).__init__()
        self.ROInum=ROInum
        self.gcn = GUNET_2.GraphUnet([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.2)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fl2 = nn.Linear(512,256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fl3 = nn.Linear(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.fl4 = nn.Linear(128, num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = self.ROInum

        h = torch.zeros(batch_size, ROInum, ROInum)
        for s in range(batch_size):
            h[s,:,:] = torch.eye(ROInum)
        h = h.cuda()
        g = g.cuda()

        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], h[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        h = out.cuda()
        h = torch.squeeze(h)

        h = self.bn1.cuda()(h)
        h = F.relu(h)
        h = self.fl1(h)

        h = self.bn2.cuda()(h)
        h = F.relu(h)
        h = self.fl2(h)

        h = self.bn3.cuda()(h)
        h = F.relu(h)
        h = self.fl3(h)

        h = self.bn4.cuda()(h)
        h = F.relu(h)
        h = self.fl4(h)

        h = self.softmax(h)

        return h

class GCNNET_diffpool(nn.Module):
    def __init__(self,ROInum, num_class=2):
        super(GCNNET_diffpool, self).__init__()
        self.ROInum=ROInum
        self.gcn = GUNET_2.GraphDiif([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.2)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fl2 = nn.Linear(512,256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fl3 = nn.Linear(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.fl4 = nn.Linear(128, num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = self.ROInum

        h = torch.zeros(batch_size, ROInum, ROInum)
        for s in range(batch_size):
            h[s,:,:] = torch.eye(ROInum)
        h = h.cuda()
        g = g.cuda()

        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], h[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        h = out.cuda()

        h = torch.squeeze(h)

        h = self.bn1.cuda()(h)
        h = F.relu(h)
        h = self.fl1(h)

        h = self.bn2.cuda()(h)
        h = F.relu(h)
        h = self.fl2(h)

        h = self.bn3.cuda()(h)
        h = F.relu(h)
        h = self.fl3(h)

        h = self.bn4.cuda()(h)
        h = F.relu(h)
        h = self.fl4(h)

        h = self.softmax(h)

        return h


class GCNNET_SAG(nn.Module):
    def __init__(self,ROInum, num_class=2):
        super(GCNNET_SAG, self).__init__()
        self.ROInum=ROInum
        self.gcn = GUNET_2.GraphSAG([4/5,3/4,2/3,1/2],ROInum, 1, 1, nn.ReLU(),0.2)

        self.bn1 = torch.nn.BatchNorm1d(1500)
        self.fl1 = nn.Linear(1500,512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fl2 = nn.Linear(512,256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fl3 = nn.Linear(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.fl4 = nn.Linear(128, num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = self.ROInum

        h = torch.zeros(batch_size, ROInum, ROInum)
        for s in range(batch_size):
            h[s,:,:] = torch.eye(ROInum)
        h = h.cuda()
        g = g.cuda()

        out = torch.zeros(batch_size, 1500)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], h[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        h = out.cuda()
        h = torch.squeeze(h)

        h = self.bn1.cuda()(h)
        h = F.relu(h)
        h = self.fl1(h)

        h = self.bn2.cuda()(h)
        h = F.relu(h)
        h = self.fl2(h)

        h = self.bn3.cuda()(h)
        h = F.relu(h)
        h = self.fl3(h)

        h = self.bn4.cuda()(h)
        h = F.relu(h)
        h = self.fl4(h)

        h = self.softmax(h)

        return h


class MAPGCNNET(nn.Module):
    def __init__(self,ROInum,layer, num_class=2):
        super(MAPGCNNET, self).__init__()
        self.ROInum=ROInum
        self.layer = layer
        self.paranum=0
        for i in range(100,ROInum+100,100):
            self.paranum=self.paranum+i

        self.mgunet = GUNET_2.MultiresolutionGUnet_base(nn.ReLU(),0.2,self.layer)

        self.bn1 = torch.nn.BatchNorm1d(self.paranum)
        self.fl1 = nn.Linear(self.paranum, 512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fl2 = nn.Linear(512, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fl3 = nn.Linear(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.fl4 = nn.Linear(128, num_class)

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
            temp = self.mgunet(g1[s, :, :], g2[s, :, :], g3[s, :, :],g4[s, :, :], g5[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)
        out = self.fl1(out)

        out = self.bn2.cuda()(out)
        out = F.relu(out)
        out = self.fl2(out)

        out = self.bn3.cuda()(out)
        out = F.relu(out)
        out = self.fl3(out)

        out = self.bn4.cuda()(out)
        out = F.relu(out)
        out = self.fl4(out)

        out = self.softmax(out)

        return out