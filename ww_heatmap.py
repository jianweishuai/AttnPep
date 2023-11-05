# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:43:22 2023

@author: liyulin
"""
import torch
import time
import optparse
import numpy as np
from data_utils import load_pin_return_featureMatrix, findInitDirection
from sklearn import preprocessing
from calc_qvalues import calcQ
from SingleTransformer import Batch, make_model
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from pyteomics import pepxml

scaler = torch.cuda.amp.GradScaler()

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

def run_model_on_data(data_iter, model, device):
    model.eval()
    Y_hat = []
    with torch.no_grad():
        for pep, pep_mask, _ in data_iter:
            y_hat = model.forward(
                pep.to(device),pep_mask.to(device)
            )
            
            if Y_hat != []:
                Y_hat = torch.cat((Y_hat, y_hat))
            else:
                Y_hat = y_hat
    model.train()
    return torch.sigmoid(Y_hat)

def single_train(model, optimizer, train_iter, valid_iter, loss_compute, num_epochs, device):
    train_loss = []
    valid_loss = []
    # best_valid_acc = 0
    # model_params_at_best_valid = model.state_dict()
    
    t0 = time.time()
    for epoch in range(num_epochs):
        losses = []
        valid_losses = []
        for pep, pep_mask, y in train_iter:

            y_hat = model.forward(
                pep.to(device), pep_mask.to(device)
            )
            y = y.type_as(y_hat).to(device)
            l = loss_compute(y_hat, y)
            optimizer.zero_grad()
            # l.backward()
            # optimizer.step()
            scaler.scale(l).backward()
            
            # # print(l)

            scaler.step(optimizer)
            scaler.update()
            losses.append(l.data.cpu().numpy())
            
        with torch.no_grad():
            for pep, pep_mask, y in train_iter:

                y_hat = model.forward(
                    pep.to(device), pep_mask.to(device)
                )
                y = y.type_as(y_hat).to(device)
                l = loss_compute(y_hat, y)
                # optimizer.zero_grad()
                # l.backward()
                # optimizer.step()
                # scaler.scale(l).backward()
                
                # # print(l)

                # scaler.step(optimizer)
                # scaler.update()
                valid_losses.append(l.data.cpu().numpy())
        print((time.time()-t0)/60) 
        t0 = time.time()
        print('Epoch {}/{} completed with average loss {:6.4f}'.format(epoch+1, num_epochs, np.mean(losses)))
        # if epoch % validation_check == 0:
        #     val_pred = run_model_on_data(valid_iter, model, device)
        #     val_acc = f1_score(val_labels, torch_tensor_to_np(val_pred.ge(0.5)))
        #     if val_acc > best_valid_acc:
        #         best_valid_acc = val_acc
        #         model_params_at_best_valid = model.state_dict()
        # else:
        #     val_acc = -1
        
        train_loss.append(np.mean(losses))
        valid_loss.append(np.mean(valid_losses))
        # valid_loss.append(val_acc)
    
    
    # model.load_state_dict(model_params_at_best_valid)
    
    return model, train_loss, valid_loss

def encode_features(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X1 = (min_max_scaler.fit_transform(X)*10000).astype(int)
    return torch.LongTensor(X1)


def init_score(X, Y, thresh, featureNames):
    initDirection, numIdentified, negBest = findInitDirection(X, Y, thresh, featureNames)
    if negBest:
        scores = -1. * X[:,initDirection]
    else:
        scores = X[:,initDirection]
    print("Could separate %d PSMs in feature \'%s\', q-vals < %f" % (numIdentified, featureNames[initDirection], thresh))
    return scores

def run_model_on_data(data_iter, model, device):
    model.eval()
    Y_hat = []
    with torch.no_grad():
        for pep, pep_mask, _ in data_iter:
            y_hat = model.forward(
                pep.to(device),pep_mask.to(device)
            )
            
            if Y_hat != []:
                Y_hat = torch.cat((Y_hat, y_hat))
            else:
                Y_hat = y_hat
    model.train()
    return Y_hat

path_pin = './test/search_ehgines/msfragger/LFQ_TTOF6600_DDA_QC_01'
pep, X, Y, featureNames, sids, expMasses = load_pin_return_featureMatrix(path_pin + '.pin')
encode = encode_features(X)
# encode = encode_features(X[:,[0,2,3,4,5,6,10,13]])

scores = init_score(X, Y, 0.01, featureNames)

Index = np.arange(0,len(X), 1)
taq, daq, _ = calcQ(scores, Y, 0.01, True)
td = [Index[i] for i in taq]
gd = [i for i in Index if Y[i] != 1]
print(" |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d" % ( len(Index) - len(gd), len(gd), len(taq), len(daq)))
trainSids0 = gd + td

def multi_train(trainSids0, encode, Y ,ratio = 0.8, thresh = 0.01):
    train_size = int(ratio * len(trainSids0))
    test_size = len(trainSids0) - train_size
    trainSids, testSids = torch.utils.data.random_split(
        dataset=trainSids0,
        lengths=[train_size, test_size]
    )


    trainEncode = encode[trainSids]
    train_batch = Batch(trainEncode)
    trainLabels = np.maximum(Y[trainSids], 0)
    train_dataset = TensorDataset(train_batch.src, train_batch.src_mask, torch.LongTensor(trainLabels))
    train_iter = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    
    testEncode = encode[testSids]
    testbatch = Batch(testEncode)
    testLabels = np.maximum(Y[testSids], 0)
    test_dataset = TensorDataset(testbatch.src, testbatch.src_mask, torch.LongTensor(testLabels))
    test_iter = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    validbatch = Batch(encode)
    validLabels = np.maximum(Y, 0)
    validDataset = TensorDataset(validbatch.src, validbatch.src_mask, torch.LongTensor(validLabels))
    valid_iter = DataLoader(
        dataset=validDataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = make_model(11000, encode.size()[1]).to(device)

    # torch.save(model.state_dict(), './test_model')

    optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8#, weight_decay= 0.0001c
        )
    loss_compute = torch.nn.BCEWithLogitsLoss()#pos_weight = torch.Tensor([len(gd)/len(td)]).to(device))

    num_epochs = 15

    model, train_loss, valid_loss = single_train(model, optimizer, train_iter, test_iter, loss_compute, num_epochs, device)
    
    y_hat = run_model_on_data(valid_iter, model, device)
    
    df = pd.DataFrame({'scores': y_hat.cpu(), 'y':Y})
    # target = df[df['y'] == 1]
    # decoy = df[df['y'] == -1]
    return df, model

def hook(module, input, output):
    featuresIn.append(input)
    featuresOut.append(output)
    return None

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

def softmax(x):
    """
    对输入x的每一行计算softmax。

    该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。
    
    代码利用softmax函数的性质: softmax(x) = softmax(x + c)

    参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.

    返回值:
    x -- 在函数内部处理后的x
    """
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x,axis=1) # 得到每行的最大值，用于缩放每行的元素，避免溢出。 shape为(x.shape[0],)
        x -= tmp.reshape((x.shape[0],1)) # 利用性质缩放元素
        x = np.exp(x) # 计算所有值的指数
        tmp = np.sum(x, axis = 1) # 每行求和        
        x /= tmp.reshape((x.shape[0], 1)) # 求softmax
    else:
        # 向量
        tmp = np.max(x) # 得到最大值
        x -= tmp # 利用最大值缩放数据
        x = np.exp(x) # 对所有元素求指数        
        tmp = np.sum(x) # 求元素和
        x /= tmp # 求somftmax
    return x


#%%
WW={}
for i in range(20):
    df, model = multi_train(trainSids0, encode, Y)
    model_params = model.state_dict()
    torch.save(model_params, './model.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model1 = make_model(11000, encode.size()[1]).to(device)
    model1.load_state_dict(torch.load('./model.pt'))

    x = encode[0].unsqueeze(0)

    validbatch = Batch(x)
    validDataset = TensorDataset(validbatch.src, validbatch.src_mask)
    valid_iter = DataLoader(
        dataset=validDataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for pep, pep_mask in valid_iter:
        pep = pep
        pep_mask = pep_mask
    
    featuresIn = []
    featuresOut = []
    net1 = model1

    #  设置钩子
    h1 = net1.encoder.layers[0].self_attn.linears[0].register_forward_hook(hook)

    out = net1.forward(
        pep.to(device),pep_mask.to(device)
    )

    h1.remove()

    f1 = featuresOut[0]
    
    featuresIn = []
    featuresOut = []
    net2 = model1

    #  设置钩子
    h2 = net2.encoder.layers[0].self_attn.linears[1].register_forward_hook(hook)
    # h1.remove()
    out = net2.forward(
        pep.to(device),pep_mask.to(device)
    )

    h2.remove()

    f2 = featuresOut[0]

    f1 = f1.squeeze(0).cpu().detach().numpy()
    f2 = f2.squeeze(0).cpu().detach().numpy()
    WW[i] = np.dot(f1, np.transpose(f2))/8
    
    
    
    with pd.ExcelWriter('./REVIEW/WW/heatmap_msfragger_WW2'+str(i)+'.xlsx') as writer:
        pd.DataFrame(softmax(WW[i])).to_excel(writer, sheet_name='heatmap')
    
#%%
test=softmax(WW[0])
for i in range(19):
    test+=softmax(WW[i+1])
    
#%%
test/=20