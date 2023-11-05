# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:25:29 2023

@author: liyulin
"""

import numpy as np
import pandas as pd
import time
import sys
import argparse
import os
import logging
from data_utils import load_pin_return_featureMatrix, findInitDirection
from SingleTransformer import Batch, make_model
from calc_qvalues import calcQ
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import torch
from torch.utils.data import DataLoader, TensorDataset


scaler = torch.cuda.amp.GradScaler()
device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))
    
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

def init_score(X, Y, thresh, featureNames):

    initDirection, numIdentified, negBest = findInitDirection(X, Y, thresh, featureNames)

    if negBest:
        scores = -1. * X[:,initDirection]
    else:
        scores = X[:,initDirection]

    logging.info("Could separate %d PSMs in feature \'%s\', q-vals < %f" % (numIdentified, featureNames[initDirection], thresh))
    
    return scores

def encode_features(X):

    min_max_scaler = preprocessing.MinMaxScaler()

    X1 = (min_max_scaler.fit_transform(X)*10000).astype(int)

    return torch.LongTensor(X1)
    
def single_train(model, train_iter, valid_iter, optimizer, loss_compute, num_epochs, device):
    train_loss = []
    valid_loss = []
    
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

            scaler.scale(l).backward()


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
                
                valid_losses.append(l.data.cpu().numpy())


        logging.info('Epoch {}/{} completed with average loss {:6.4f}\n'.format(epoch+1, num_epochs, np.mean(losses)))

        train_loss.append(np.mean(losses))
        valid_loss.append(np.mean(valid_losses))

    
    return model, train_loss, valid_loss

def train(X, Y, featureNames,num_epochs, LR):
 
    encode = encode_features(X)

    kf = StratifiedKFold(n_splits=3, shuffle=True)

    Model = []
    
    i = 0
    for train_index, test_index in kf.split(X,Y):

        
        trainEncode = encode[train_index]
        train_batch = Batch(trainEncode)
        trainLabels = np.maximum(Y[train_index], 0)
        
        train_dataset = TensorDataset(train_batch.src, train_batch.src_mask, torch.LongTensor(trainLabels))
        train_iter = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
        
        testEncode = encode[test_index]
        testbatch = Batch(testEncode)
        testLabels = np.maximum(Y[test_index], 0)
        test_dataset = TensorDataset(testbatch.src, testbatch.src_mask, torch.LongTensor(testLabels))
        test_iter = DataLoader(
            dataset=test_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
        )
        
        model = make_model(11000, encode.size()[1]).to(device)
        optimizer = torch.optim.Adam(
                model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-8
            )
        
        loss_compute = torch.nn.BCEWithLogitsLoss()

        logging.info('CV '+str(i+1)+'\n')
        model, train_loss, valid_loss = single_train(model, train_iter, test_iter, optimizer, loss_compute, num_epochs, device)
        
        Model.append(model)
        
        i+= 1

    return Model

def rescore_all_data(X, Y,Model):
    encode = encode_features(X)
    validbatch = Batch(encode)
    validLabels = np.maximum(Y, 0)
    validDataset = TensorDataset(validbatch.src, validbatch.src_mask, torch.LongTensor(validLabels))
    valid_iter = DataLoader(
        dataset=validDataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    DF = []
    for i in range(len(Model)):
        y_hat = run_model_on_data(valid_iter, Model[i], device)
        DF.append(y_hat.cpu().numpy())
    
    DF1 = np.mean(DF,axis=0)
    return pd.DataFrame({'score': DF1, 'Label':Y}), DF

def oversample(X, Y):
    data = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
    minority_data = data[data[:, -1] == -1]
    majority_data = data[data[:, -1] == 1] 

    minority_data_oversampled = resample(minority_data, replace=True, n_samples=len(majority_data))

    data_oversampled = np.concatenate((majority_data, minority_data_oversampled), axis=0)

    X_oversampled = data_oversampled[:, :-1]
    Y_oversampled = data_oversampled[:, -1]
    return X_oversampled, Y_oversampled
   

def multi_train(X,Y,featureNames, n, epochs, LR, q_threshold):
    scores = init_score(X, Y, q_threshold, featureNames)
    Index = np.arange(0,len(X), 1)
    taq, daq, _ = calcQ(scores, Y, q_threshold, True)
    td = [Index[i] for i in taq]
    gd = [i for i in Index if Y[i] != 1]
    

    logging.info('Iteration 1')
    logging.info(" |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d\n" % ( len(Index) - len(gd), len(gd), len(taq), len(daq)))
    
    Scores = []
    Taq = []
    
    Scores.append(scores)
    Taq.append(taq)
    
    i = 1
    
    while i<= n:
        
        trainSids0 = gd + td

        X_train = X[trainSids0,:]
        Y_train = Y[trainSids0]
        
        # X_train,Y_train = oversample(X_train0, Y_train0)

        Model = train(X_train, Y_train, featureNames, epochs, LR)
    
        TransValid, DF = rescore_all_data(X, Y, Model)
        
        scores = TransValid['score'].values
        Index = np.arange(0,len(X), 1)
        taq, daq, _ = calcQ(scores, Y, q_threshold, True)
        
        if 1:
            Scores.append(TransValid['score'].values)
            Taq.append(taq)
            
            td = [Index[i] for i in taq]
            gd = [i for i in Index if Y[i] != 1]
            if (i!= n):
                logging.info('Iteration '+ str(i+1)+'\n')
           
            logging.info(" |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d\n" % ( len(Index) - len(gd), len(gd), len(taq), len(daq)))
            
            i += 1
           
    return Scores, Taq

def find_longest_list_index(collection):
    longest_index = max(range(len(collection)), key=lambda i: len(collection[i]))
    return longest_index

def merge(Scores):
    m = len(Scores)
    n = len(Scores[0])
    merged_scores = []

    l = len(Scores[0][0])

    for i in range(m):
        for j in range(1,n):
            merged_scores.append(Scores[i][j])

    merged_score = []
    for k in range(l):
        temp=0
        for i in range(len(merged_scores)):
            temp += merged_scores[i][k]
        merged_score.append(temp/(len(merged_scores)+1))
    
    return merged_score

def Output(AttnPep, p ,qs,outpath, inpath):
    
    file_name = os.path.splitext(os.path.basename(inpath))[0]
    file_path = outpath + 'output_'+file_name+'.csv'
    
    PSMId = []
    Peptide = []
    Protein = []
    for i in range(len(p)):
        PSMId.append(p[i][0])
        Peptide.append(p[i][1])
        Protein.append(p[i][2])

    AttnPep['PSMId'] = PSMId
    AttnPep['Peptide'] = Peptide
    AttnPep['Protein'] = Protein
    
    AttnPep.sort_values(by="score" , inplace=True, ascending=False) 
    AttnPep['q-values'] = qs

    AttnPep = pd.DataFrame.reindex(AttnPep, columns=["PSMId", "score","q-values", "Label", "Peptide","Protein"])
        
    AttnPep.to_csv(file_path,index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type = float, action= 'store', default = 0.01)
    parser.add_argument('-training_q', type = float, action= 'store', default = 0.01)
    parser.add_argument('-maxIters', type = int, action= 'store', default = 2, help='number of iterations; runs on multiple splits per iterations.')
    parser.add_argument('-epochs', type = int, action= 'store', default = 5, help='number of epochs.')
    parser.add_argument('-n', type = int, action= 'store', default = 2, help='number of separate training.')
    parser.add_argument('-i', type = str, action= 'store', help='input file in PIN format')
    parser.add_argument('-o', type = str, action= 'store', default='.', help='Output path')
    parser.add_argument('-lr', type = float, action= 'store', default = 0.001, help='learning rate for training the model.')

    args = parser.parse_args()
    params = vars(args)
    
    logging.basicConfig(
        filename=params['o']+'/AttnPep.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    
    pep, X, Y, featureNames, _, _  = load_pin_return_featureMatrix(params['i'])
    
    Scores = []
    logging.info('Start Training...\n')
    
    for i in range(params['n']):
        Scoresi, Taqi = multi_train(X, Y, featureNames,params['maxIters'],params['epochs'], params['lr'], params['training_q'])
        Scores.append(Scoresi)
    
    logging.info('Merging scores...\n')
    merged_score = merge(Scores)
    
    AttnPep = pd.DataFrame({'score': merged_score, 'Label':Y})
    taq, daq, qs = calcQ(sigmoid(AttnPep['score'].values), AttnPep['Label'].values, params['q'], False)
    
    logging.info("Could final separate %d PSMs in PSM-level q-values < %f"%(len(taq)+len(daq), params['q']))
    
    logging.info('Output...')
    Output(AttnPep, pep, qs,params['o'],params['i'])
    logging.info('All jobs done!')
    


    
if __name__ == '__main__':
    main()