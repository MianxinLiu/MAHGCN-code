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
        filename = './interlayermapping/mapping_' + str(
            self.indim) + 'to' + str(self.outdim) + '_b.mat'
        Map = scio.loadmat(filename)
        Map = Map['mapping']
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        h = torch.matmul(h, Map)
        h = h.permute(0,2,1)

        return h


class MAPGCN(nn.Module):
    def __init__(self,  act, drop_p, layer):
        super(MAPGCN, self).__init__()
        self.layer=layer
        self.net_gcn_down1 = GCN(500, 1, act, drop_p)
        self.net_gcn_down2 = GCN(400, 1, act, drop_p)
        self.net_gcn_down3 = GCN(300, 1, act, drop_p)
        self.net_gcn_down4 = GCN(200, 1, act, drop_p)
        self.net_gcn_bot = GCN(100, 1, act, drop_p)

    def forward(self, g1, g2, g3, g4, g5, h):
        if self.layer==5:
            h0=h.cuda()
            h = self.net_gcn_down1(g5, h0)
            downout1 = h
            h = self.net_gcn_down2(g4, h0[0:400, 0:400])
            downout2 = h
            h = self.net_gcn_down3(g3, h0[0:300, 0:300])
            downout3 = h
            h = self.net_gcn_down4(g2, h0[0:200, 0:200])
            downout4 = h
            h = self.net_gcn_bot(g1, h0[0:100, 0:100])
            hh = torch.cat((h, downout1, downout2, downout3, downout4))

        return hh


class GraphDiif(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphDiif, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)

        self.pools.append(DiffPool(500, 400))
        self.pools.append(DiffPool(400, 300))
        self.pools.append(DiffPool(300, 200))
        self.pools.append(DiffPool(200, 100))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))

    def forward(self, g, h):
        hnext = self.top_gcn(g, h)
        h1 = hnext
        for i in range(self.l_n):
            g, h = self.pools[i](g, h, hnext)
            #print(g.shape)
            #print(h.shape)
            h=torch.diag(torch.squeeze(h))
            hnext = self.down_gcns[i](g, h)
            #print(hnext.shape)
            h1=torch.cat([h1, hnext], dim=0)

        return h1
    def Loss(self):
        L=0
        for i in range(self.l_n):
            L=L+self.pools[i].Loss()
        return L

class DiffPool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiffPool, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.gcn=GCN(self.in_dim, self.out_dim, nn.ReLU(), 0.2)
        self.softmax = nn.Softmax()
        self.assign = torch.zeros(in_dim, out_dim)
        self.g = torch.zeros(in_dim, in_dim)

    def forward(self, g, h, hnext):
        self.g = g
        self.assign=self.gcn(g,h)
        #print(assign.shape)
        self.assign=self.softmax(self.assign)
        newh = torch.matmul(torch.transpose(self.assign, 0, 1), hnext)
        newg = torch.transpose(self.assign, 0, 1) @ g @ self.assign
        return newg, newh
    def Loss(self):
        loss_LP=torch.norm(torch.cat([self.g,torch.matmul(self.assign,torch.transpose(self.assign, 0, 1))]))/self.in_dim
        loss_en=0
        eps = 1e-7
        NPassign=F.relu(self.assign)
        for row in range(self.in_dim):
            loss_en=loss_en-torch.sum(NPassign[row,:]*torch.log(NPassign[row,:])+eps)
        loss_en=loss_en/self.in_dim
        return (loss_LP+loss_en)/self.in_dim

class GraphSAG(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphSAG, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.pools.append(SAGPool(ks[i], dim, drop_p))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            h1=torch.cat([h1, h], dim=0)

        return h1

class SAGPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(SAGPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.in_dim = in_dim
        self.proj = GCN(self.in_dim, 1, nn.ReLU(), p)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(g, Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.pools.append(gPool(ks[i], dim, drop_p))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            h1=torch.cat([h1, h], dim=0)
        
        return h1

class gPool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(gPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)

def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

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