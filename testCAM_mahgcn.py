import pandas as pd
import numpy as np
import scipy.io as scio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import MyModels
from sklearn import metrics
from random import shuffle
import GradCAM_mahgcn


neg=1
pos=2

subInfo = pd.read_csv('./subinfo.csv')
y_data1 = torch.from_numpy(subInfo['group'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)
y_data1[y_data1 == pos] = 9
y_data1[y_data1 == neg] = 0
y_data1[y_data1 == 9] = 1

age = torch.from_numpy(subInfo['age'].to_numpy())
age = torch.tensor(age, dtype=torch.float32)
gen = torch.from_numpy(subInfo['sex2'].to_numpy())
gen = torch.tensor(gen, dtype=torch.float32)

A1 = np.zeros((len(subInfo), 100, 100))
A2 = np.zeros((len(subInfo), 200, 200))
A3 = np.zeros((len(subInfo), 300, 300))
A4 = np.zeros((len(subInfo), 400, 400))
A5 = np.zeros((len(subInfo), 500, 500))
subcount = 0
for id in index:
    fn = subInfo['Subj Name'][id]
    fn = fn[:len(fn) - 4] + '_FC.mat'
    FCfile = scio.loadmat('/home/user/matlab/scripts/adni data/data/par' + str(100) + '/' + fn)
    A1[subcount, :, :] = FCfile['FC']
    FCfile = scio.loadmat('/home/user/matlab/scripts/adni data/data/par' + str(200) + '/' + fn)
    A2[subcount, :, :] = FCfile['FC']
    FCfile = scio.loadmat('/home/user/matlab/scripts/adni data/data/par' + str(300) + '/' + fn)
    A3[subcount, :, :] = FCfile['FC']
    FCfile = scio.loadmat('/home/user/matlab/scripts/adni data/data/par' + str(400) + '/' + fn)
    A4[subcount, :, :] = FCfile['FC']
    FCfile = scio.loadmat('/home/user/matlab/scripts/adni data/data/par' + str(500) + '/' + fn)
    A5[subcount, :, :] = FCfile['FC']
    subcount = subcount + 1

A1 = torch.tensor(A1, dtype=torch.float32)
A1.cuda()
A2 = torch.tensor(A2, dtype=torch.float32)
A2.cuda()
A3 = torch.tensor(A3, dtype=torch.float32)
A3.cuda()
A4 = torch.tensor(A4, dtype=torch.float32)
A4.cuda()
A5 = torch.tensor(A5, dtype=torch.float32)
A5.cuda()

for cv in ([1,2,3,4,5]):
    mask_all=np.zeros((len(subInfo),200))
    model = MyModels.MAHGCNNET(ROInum=ROInum, layer=int(ROInum / 100))
    model.cuda()
    model.load_state_dict(
        torch.load('./GCRN_copy/ckpt/ROI' + str(ROInum) + '_magcn_hc_' + status + '_catsum_cv' + str(cv) + '.pth'))
    model.eval()
    for i in range(len(subInfo)):
        target_index = 1
        gcam = GradCAM_magcn.GradCam(model=model.eval(), target_layer_names=['net_gcn_down4'], use_cuda=True)
        g1 = torch.zeros((1, 100, 100))
        g2 = torch.zeros((1, 200, 200))
        g3 = torch.zeros((1, 300, 300))
        g4 = torch.zeros((1, 400, 400))
        g5 = torch.zeros((1, 500, 500))
        g1[0, :, :] =A1[i, :, :]
        g2[0, :, :] = A2[i, :, :]
        g3[0, :, :] = A3[i, :, :]
        g4[0, :, :] = A4[i, :, :]
        g5[0, :, :] = A5[i, :, :]
        g1.requires_grad=True
        g2.requires_grad=True
        g3.requires_grad=True
        g4.requires_grad=True
        g5.requires_grad=True
        gen_input = torch.zeros((1, 1))
        gen_input[0, 0] = gen[i]
        age_input = torch.zeros((1, 1))
        age_input[0, 0] = age[i]
        mask_all[i, :] = np.squeeze(gcam(g1,g2,g3,g4,g5, gen[i],age[i],target_index))

    filename_save = './gradcam/'+status+'_gradcam_down4_cv'+str(cv)+'.mat'
    scio.savemat(filename_save,{'mask_all':mask_all})