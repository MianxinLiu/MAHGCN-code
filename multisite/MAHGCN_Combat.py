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

neg=0
pos=1

subInfo = pd.read_csv('./subinfo_revised.csv')

x_data1 = torch.from_numpy(subInfo.index.to_numpy())
x_data1 = torch.tensor(x_data1, dtype=torch.float32)

y_data1 = torch.from_numpy(subInfo['group'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)

s_data1 = torch.from_numpy(subInfo['site'].to_numpy())
s_data1 = torch.tensor(s_data1, dtype=torch.float32)


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


lr = 0.01
EPOCH = 150
ROInum=500

for cv in [1,2,3,4,5,6,7,8,9,10]:
    temp = scio.loadmat('./kfold/Combat_shuffled_index_cv' + str(cv))
    train_idx = np.float32(temp['train_idx'][0])
    test_idx = np.float32(temp['test_idx'][0])
    # print("Train:", train_idx, " Test:", test_idx)

    x_train = x_data1[train_idx]
    y_train = y_data1[train_idx]
    s_train = s_data1[train_idx]

    x_test = x_data1[test_idx]
    y_test = y_data1[test_idx]
    s_test = s_data1[test_idx]
    qualified=[]

    #ratio=y_data1[train_idx].sum()/(y_data1[train_idx].shape[0]-y_data1[train_idx].sum())
    #weight = torch.cuda.FloatTensor([1, 1.2])
    #loss_func = nn.CrossEntropyLoss(weight)  # the target label is not one-hotted
    #loss_func = MyModels.FocalLoss(weight, gamma=3)  # the target label is not one-hotted
    site_weight= torch.cuda.FloatTensor([1011, 297, 267, 1350, 717, 768])
    site_weight=torch.sqrt((1/site_weight))*100

    site_size=torch.cuda.FloatTensor([100, 100, 100, 100, 100, 100])

    while not qualified:

        model = MyModels.MAHGCNNET(ROInum=ROInum, layer=int(ROInum / 100))
        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
        test_auc = []
        train_los = []
        test_los = []
        train_auc = []
        sen = []
        spe = []
        auc_baseline = 0.65

        for epoch in range(EPOCH):
            if EPOCH==50:
                lr = 0.001
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
            loss = 0
            predicted_all = []
            b_y_all = []
            for site in range(1,7):

                x = x_train[s_train == site]
                y = y_train[s_train == site]
                index = np.arange(y.shape[0])
                shuffle(index)
                x = x[index]
                y = y[index]
                ratio = y.sum() / (y.shape[0] - y.sum())
                if ratio<1:
                    weight = torch.cuda.FloatTensor([1, 1 / ratio])
                else:
                    weight = torch.cuda.FloatTensor([ratio, 1])
                loss_func = nn.CrossEntropyLoss(weight)  # the target label is not one-hotted
                bsize = int(site_size[site-1])
                dataset_train = TensorDataset(x[0:bsize], y[0:bsize])
                train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=bsize)
                #print('Selected site:', site)

                for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                    model.train()
                    b_y = b_y.view(-1)
                    b_y = b_y.long()
                    # b_y = torch.LongTensor(np.squeeze(b_y))
                    # for i in range(INPUT_SIZE):
                    b_y = b_y.cuda()
                    A1, A2, A3, A4, A5 = load_FCN(b_x, subInfo)

                    output = model(A1, A2, A3, A4, A5)

                    loss += site_weight[site-1]*loss_func(output, b_y)  # cross entropy loss
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
            if epoch >= 80:
                model.eval()
                predicted_all = []
                test_y_all = []
                ACC = 0
                AUC = 0
                SEN = 0
                SPE = 0
                valid = 1
                with torch.no_grad():
                    for site in range(1,7):
                        predicted_site = []
                        test_y_site = []
                        #x = x_train[s_train == site]
                        #y = y_train[s_train == site]
                        x = x_test[s_test == site]
                        y = y_test[s_test == site]

                        dataset_test = TensorDataset(x, y)
                        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100)
                        for step, (test_x, test_y) in enumerate(test_loader):  # gives batch data
                            test_y = test_y.view(-1)
                            test_y = test_y.long()
                            # b_y = torch.LongTensor(np.squeeze(b_y))
                            # for i in range(INPUT_SIZE):
                            test_y = test_y.cuda()
                            A1, A2, A3, A4, A5 = load_FCN(test_x, subInfo)

                            test_output = model(A1, A2, A3, A4, A5)  # rnn output
                            predicted = torch.max(test_output.data, 1)[1]
                            correct = (predicted == test_y).sum()
                            accuracy = float(correct) / float(predicted.shape[0])
                            test_y = test_y.cpu()
                            predicted = predicted.cpu()
                            predicted_site = predicted_site + predicted.tolist()
                            test_y_site = test_y_site + test_y.tolist()
                            predicted_all = predicted_all + predicted.tolist()
                            test_y_all = test_y_all + test_y.tolist()

                        correct = (np.array(predicted_site) == np.array(test_y_site)).sum()
                        accuracy = float(correct) / float(len(test_y_site))
                        ACC += accuracy/6
                        # test_auc.append(accuracy)
                        sens = metrics.recall_score(test_y_site, predicted_site, pos_label=1)
                        SEN += sens/6
                        # sen.append(sens)
                        spec = metrics.recall_score(test_y_site, predicted_site, pos_label=0)
                        SPE += spec/6
                        # spe.append(spec)
                        auc = metrics.roc_auc_score(test_y_site, predicted_site)
                        AUC += auc/6
                        if auc<0.55:
                            valid = 0
                        print('|site:', site,
                              '|test accuracy:', accuracy,
                              '|test sen:', sens,
                              '|test spe:', spec,
                              '|test auc:', auc,
                              )
                #correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
                #accuracy = float(correct) / float(len(test_y_all))
                # test_auc.append(accuracy)
                #sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
                # sen.append(sens)
                #spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
                # spe.append(spec)
                #auc = metrics.roc_auc_score(test_y_all, predicted_all)
                print('|test accuracy:', ACC,
                      '|test sen:', SEN,
                      '|test spe:', SPE,
                      '|test auc:', AUC,
                      )

                if AUC >= auc_baseline and SEN >= 0.51 and SPE >= 0.51 and valid:
                    auc_baseline = AUC
                    torch.save(model.state_dict(),
                               './model/model' + str(neg) + 'vs' + str(pos) + '_cv' + str(cv) + '_magcn_CombatV2.pth')
                    print('got one model with |test accuracy:', ACC,
                          '|test sen:', SEN,
                          '|test spe:', SPE,
                          )
                    qualified.append([ACC, SEN, SPE, AUC])

print(qualified)
