import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio


class MAHGCN(nn.Module):
    def __init__(self,  act, drop_p, layer, num_site):
        super(MAHGCN, self).__init__()
        self.layer=layer
        self.num_site = num_site
        self.net_gcn_down1 = GCN(500, 200, act, drop_p)
        self.bn1 = dsbn.DomainSpecificBatchNorm1d(200, num_site)
        self.net_pool1 = AtlasMap_batch(500, 400, drop_p)

        self.net_gcn_down2 = GCN(200, 200, act, drop_p)
        self.bn2 = dsbn.DomainSpecificBatchNorm1d(200, num_site)
        self.net_pool2 = AtlasMap_batch(400, 300, drop_p)

        self.net_gcn_down3 = GCN(200, 200, act, drop_p)
        self.bn3 = dsbn.DomainSpecificBatchNorm1d(200, num_site)
        self.net_pool3 = AtlasMap_batch(300, 200, drop_p)

        self.net_gcn_down4 = GCN(200, 200, act, drop_p)
        self.bn4 = dsbn.DomainSpecificBatchNorm1d(200, num_site)
        self.net_pool4 = AtlasMap_batch(200, 100, drop_p)

        self.net_gcn_bot = GCN(200, 200, act, drop_p)
        self.bn5 = dsbn.DomainSpecificBatchNorm1d(200, num_site)

        self.proj = nn.Linear(200,1)

    def forward(self, g1, g2, g3, g4, g5, h, site_id):
        if self.layer==5:
            h = self.net_gcn_down1(g5, h)

            h = h.permute([0,2,1])
            #print(h.shape)
            for site in range(1, self.num_site + 1):
                h[site_id == site, :, :] = self.bn1.cuda()(
                    h[site_id == site, :, :], site - 1)
            h = h.permute([0,2,1])
            #print(h.shape)
            downout1 = h
            h = self.net_pool1(h)

            h = self.net_gcn_down2(g4, h)
            h = h.permute([0,2,1])
            for site in range(1, self.num_site + 1):
                h[site_id == site, :, :] = self.bn2.cuda()(
                    h[site_id == site, :, :], site - 1)
            h = h.permute([0,2,1])
            downout2 = h
            h = self.net_pool2(h)
            h = self.net_gcn_down3(g3, h)
            h=h.permute([0, 2, 1])
            for site in range(1, self.num_site + 1):
                h[site_id == site, :, :] = self.bn3.cuda()(
                    h[site_id == site, :, :], site - 1)
            h = h.permute([0,2,1])
            downout3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g2, h)
            h = h.permute([0,2,1])
            for site in range(1, self.num_site + 1):
                h[site_id == site, :, :] = self.bn4.cuda()(
                    h[site_id == site, :, :], site - 1)
            h = h.permute([0,2,1])
            downout4 = h
            h = self.net_pool4(h)
            h = self.net_gcn_bot(g1, h)
            h = h.permute([0,2,1])
            for site in range(1, self.num_site + 1):
                h[site_id == site, :, :] = self.bn5.cuda()(
                    h[site_id == site, :,:], site - 1)
            h = h.permute([0,2,1])
            hh = torch.cat((h, downout1, downout2, downout3, downout4), 1)
            hh = self.proj(hh)
            hh = torch.squeeze(hh)

        return hh

class AtlasMap(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        filename = '/media/user/4TB/matlab/scripts/adni data/data/interlayermapping/mapping_'+str(self.indim) +'to' + str(self.outdim)+ '_b.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        #h = torch.diag(h)
        return h

class AtlasMap_batch(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap_batch, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):

        h = h.permute(0,2,1)
        filename = '/media/user/4TB/matlab/scripts/adni data/data/interlayermapping/mapping_' + str(
            self.indim) + 'to' + str(self.outdim) + '_b.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        h = torch.matmul(h, Map)
        h = h.permute(0,2,1)

        return h

class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h , meta_loss=None, meta_step_size=None, stop_gradient=False):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = linear(inputs=h,
                   weight=self.proj.weight,
                   bias=self.proj.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        h = self.act(h)
        return h