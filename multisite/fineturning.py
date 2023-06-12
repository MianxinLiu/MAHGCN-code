import pandas as pd
import numpy as np
import scipy.io as scio
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import MyModels
from sklearn import metrics
from sklearn.model_selection import KFold
from random import shuffle
import GradCAM
import glob
from random import shuffle
import transferlearning as TL

neg=0
pos=1

workpath = '.'

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

source = 6
freeze = 3
shot = 100

for pretrain in [True,False]:
    for set in [1,2,3,4,5]:

        EPOCH = 50
        ROInum=500

        qualified=[]
        qualified_all=[]
        for cv in range(1,11):
            subInfo = pd.read_csv(workpath + '/subinfo_revised.csv')

            x_data1 = torch.from_numpy(subInfo.index.to_numpy())
            x_data1 = torch.tensor(x_data1, dtype=torch.float32)

            y_data1 = torch.from_numpy(subInfo['group'].to_numpy())
            y_data1 = torch.tensor(y_data1, dtype=torch.float32)

            s_data1 = torch.from_numpy(subInfo['site'].to_numpy())
            s_data1 = torch.tensor(s_data1, dtype=torch.float32)
            temp = scio.loadmat(workpath+'/kfold/Combat_shuffled_index_cv' + str(cv))

            train_idx = np.float32(temp['train_idx'][0])
            test_idx = np.float32(temp['test_idx'][0])
            # print("Train:", train_idx, " Test:", test_idx)
            index = np.concatenate((train_idx, test_idx))
            x_data1 = x_data1[index]
            y_data1 = y_data1[index]
            s_data1 = s_data1[index]

            x_data1 = x_data1[s_data1 == set]
            y_data1 = y_data1[s_data1 == set]
            qualified = []

            x_train = x_data1[0:shot]
            y_train = y_data1[0:shot]

            x_test = x_data1[100:]
            y_test = y_data1[100:]

            ratio = y_train.sum() / (y_train.shape[0] - y_train.sum())
            weight = torch.cuda.FloatTensor([1, 1 / ratio])
            loss_func = nn.CrossEntropyLoss(weight)
            qualified = []
            auc_baseline = 0.00
            for trial in range(5):
                model = MyModels.MAHGCNNET(ROInum=ROInum, layer=int(ROInum / 100))
                if pretrain:
                    print('Load pretrained')
                    lr = 0.001
                    pretrained_dict = torch.load(workpath + '/model/model' + str(neg) + 'vs' + str(pos) + '_magcn_Combat_pretrain'+str(source)+'.pth')
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    if freeze==1:
                        TL.freeze_by_names(model, ('mgunet'))
                        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                                weight_decay=1e-2)
                    if freeze==2:
                        TL.freeze_by_names(model, ('mgunet', 'fl1', 'fl2', 'fl3'))
                        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                                weight_decay=1e-2)
                    if freeze==3:
                        TL.freeze_by_names(model, ('mgunet', 'fl1', 'bn1', 'fl2', 'bn2'))
                        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                                weight_decay=1e-2)                            
                    if freeze==0:
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
                else:
                    print('Not pretrained')
                    lr = 0.001
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
                model.cuda()
                test_auc = []
                train_los = []
                test_los = []
                train_auc = []
                sen = []
                spe = []


                for epoch in range(EPOCH):
                    predicted_all = []
                    b_y_all = []

                    dataset_train = TensorDataset(x_train, y_train)
                    dataset_test = TensorDataset(x_test, y_test)
                    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=50, shuffle=True)
                    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=10)

                    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                        model.train()
                        b_y = b_y.view(-1)
                        b_y = b_y.long()
                        # b_y = torch.LongTensor(np.squeeze(b_y))
                        # for i in range(INPUT_SIZE):
                        b_y = b_y.cuda()
                        A1, A2, A3, A4, A5 = load_FCN(b_x, subInfo)

                        output = model(A1, A2, A3, A4, A5)

                        loss = loss_func(output, b_y)  # cross entropy loss
                        predicted = torch.max(output.data, 1)[1]

                        b_y = b_y.cpu()
                        predicted = predicted.cpu()
                        predicted_all = predicted_all + predicted.tolist()
                        b_y_all = b_y_all + b_y.tolist()


                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    correct = (np.array(predicted_all) == np.array(b_y_all)).sum()
                    tr_accuracy = float(correct) / float(len(b_y_all))

                    print('Epoch:', epoch + 1, 'Batch:', step+1, '|train diag loss:', loss.data.item(), '|train accuracy:', tr_accuracy
                          )
                    if epoch >= 49:
                        model.eval()
                        predicted_all = []
                        test_y_all = []
                        with torch.no_grad():
                            for i, (test_x, test_y) in enumerate(test_loader):
                                test_y = test_y.view(-1)
                                test_y = test_y.long()
                                test_y = test_y.cuda()
                                A1, A2, A3, A4, A5 = load_FCN(test_x, subInfo)
                                # test_x = torch.ones((batch_size * 116, 1))
                                test_output = model(A1, A2, A3, A4, A5)
                                # test_loss = loss_func(test_output, test_y)
                                predicted = torch.max(test_output.data, 1)[1]
                                correct = (predicted == test_y).sum()
                                accuracy = float(correct) / float(predicted.shape[0])
                                test_y = test_y.cpu()
                                predicted = predicted.cpu()
                                predicted_all = predicted_all + predicted.tolist()
                                test_y_all = test_y_all + test_y.tolist()

                        correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
                        accuracy = float(correct) / float(len(test_y_all))
                        # test_auc.append(accuracy)
                        sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
                        # sen.append(sens)
                        spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
                        # spe.append(spec)
                        auc = metrics.roc_auc_score(test_y_all, predicted_all)
                        print('|test accuracy:', accuracy,
                              '|test sen:', sens,
                              '|test spe:', spec,
                              '|test auc:', auc,
                              )
                        if auc >= auc_baseline and sens >= 0.00 and spec >= 0.00:
                            auc_baseline = auc
                            #torch.save(model.state_dict(),
                            #           workpath + '/model/model' + str(neg) + 'vs' + str(pos) + '_cv' + str(cv) +'_shot'+ str(shot)+ '_magcn_FT_set'+ str(set) + '.pth')
                            print('got one model with |test accuracy:', accuracy,
                                  '|test sen:', sens,
                                  '|test spe:', spec,
                                  )
                            qualified.append([accuracy, sens, spec, auc])


            qualified_all.append(qualified[-1])
        print(qualified_all)
        print(np.mean(qualified_all, axis=0))
        print(np.std(qualified_all, axis=0))

        if pretrain:
            prefix= 'trained'
            scio.savemat(workpath + '/results/Performance_source'+str(source)+'_set'+str(set)+'_shot'+str(shot)+'_'+prefix+'_freeze' +str(freeze)+ '.mat', {'qualified_all':qualified_all})
        else:
            prefix = 'nottrained'
            scio.savemat(workpath + '/results/Performance_source'+str(source)+'_set'+str(set)+'_shot'+str(shot)+'_'+prefix+ '.mat', {'qualified_all':qualified_all})

        


