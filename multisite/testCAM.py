import pandas as pd
import numpy as np
import scipy.io as scio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import MyModels
from sklearn import metrics
from random import shuffle
import GradCAM

def load_FCN(b_x, path_var):
    temp = b_x.numpy().tolist()
    batch_size = b_x.shape[0]
    A1 = np.zeros((batch_size, 100, 100))
    A2 = np.zeros((batch_size, 200, 200))
    A3 = np.zeros((batch_size, 300, 300))
    A4 = np.zeros((batch_size, 400, 400))
    A5 = np.zeros((batch_size, 500, 500))
    subcount = 0
    for id in temp:
        path = path_var['Combatpath'].values[int(id)]
        fn = path_var['filename'].values[int(id)]
        filepath = path + '/par' + str(100) + '/' + fn + '.mat'
        FCfile = scio.loadmat(filepath)
        A1[subcount, :, :] = FCfile['FC']

        filepath = path + '/par' + str(200) + '/' + fn + '.mat'
        FCfile = scio.loadmat(filepath)
        A2[subcount, :, :] = FCfile['FC']

        filepath = path + '/par' + str(300) + '/' + fn + '.mat'
        FCfile = scio.loadmat(filepath)
        A3[subcount, :, :] = FCfile['FC']

        filepath = path + '/par' + str(400) + '/' + fn + '.mat'
        FCfile = scio.loadmat(filepath)
        A4[subcount, :, :] = FCfile['FC']

        filepath = path + '/par' + str(500) + '/' + fn + '.mat'
        FCfile = scio.loadmat(filepath)
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

    return A1, A2, A3, A4, A5

ROInum=500

neg=0
pos=1

savepath='/media/user/Elements/all_disease_project'

subInfo = pd.read_csv('./subinfo_revised.csv')

x_data1 = torch.from_numpy(subInfo.index.to_numpy())
x_data1 = torch.tensor(x_data1, dtype=torch.float32)

y_data1 = torch.from_numpy(subInfo['group'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)

s_data1 = torch.from_numpy(subInfo['site'].to_numpy())
s_data1 = torch.tensor(s_data1, dtype=torch.float32)

fnum=[500,400,300,200,100]
lname=['net_gcn_down1','net_gcn_down2','net_gcn_down3','net_gcn_down4','net_gcn_bot']
for layer in range(1,6):
    for cv in range(1, 11):
        mask_all=np.zeros((len(x_data1),fnum[layer-1],10))
        model = MyModels.MAHGCNNET(ROInum=ROInum, layer=int(ROInum / 100))
        model.cuda()
        model.load_state_dict(torch.load('./model/model' + str(neg) + 'vs' + str(pos) +'_cv'+str(cv)+ '_mahgcn_Combat_fft.pth'))
        model.eval()
        for i in range(0,len(x_data1),100):
            g1, g2, g3, g4, g5 = load_FCN(x_data1[i:i+100], subInfo)
            target_index = 1
            gcam = GradCAM.GradCam(model=model.eval(), target_layer_names=lname[layer-1], use_cuda=True)

            g1.requires_grad=True
            g2.requires_grad=True
            g3.requires_grad=True
            g4.requires_grad=True
            g5.requires_grad=True
            mask_all[i:i+100, :, cv-1] = np.squeeze(gcam(g1,g2,g3,g4,g5, target_index))

    filename_save = savepath+'/gradcam/'+'_gradcam_down'+str(layer)+'.mat'
    scio.savemat(filename_save,{'mask_all':mask_all})
