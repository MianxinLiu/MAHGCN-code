import torch.nn.functional as F
import torch
import torch.nn as nn

import MAHGCN
from ops import linear
import dsbn
from collections import OrderedDict

gpu = 1

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduce = reduce
        self.BCE_loss = nn.CrossEntropyLoss(reduce=False, reduction='none')
        self.WBCE_loss = nn.CrossEntropyLoss(weight=self.weight, reduce=False, reduction='none')

    def forward(self, inputs, targets):

        pt = torch.exp(-self.BCE_loss(inputs, targets))
        F_loss = (1-pt)**self.gamma * self.WBCE_loss(inputs, targets)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

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

    def forward(self, g, meta_loss=None, meta_step_size=None, stop_gradient=False):
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

    def forward(self, g1, g2, g3, g4, g5, meta_loss=None, meta_step_size=None, stop_gradient=False):
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
