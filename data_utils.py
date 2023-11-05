# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:45:13 2022

@author: liyulin
"""
import numpy as np
import os
from os.path import splitext
import gzip
import csv
from sklearn import preprocessing
from calc_qvalues import calcQ

_identOutput=False
_debug=True
# _verb=0
_mergescore=True
_scoreInd=0

# Take max wrt sid (TODO: change key from sid to sid+exp_mass)
_topPsm=False
# Check if training has converged over past two iterations
_convergeCheck=False
_reqIncOver2Iters=0.01


def getFileName(path,suffix):
    input_template_All=[]
    input_template_All_Path=[]
    for root, dirs, files in os.walk(path, topdown=False):
         for name in files:
             #print(os.path.join(root, name))
             if os.path.splitext(name)[1] == suffix:
                 input_template_All.append(name)
                 input_template_All_Path.append(os.path.join(root, name))
        
    return input_template_All,input_template_All_Path

def checkGzip_openfile(filename, mode = 'r'):
    if splitext(filename)[1] == '.gz':
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)
    
def load_pin_return_featureMatrix(filename, _standardNorm=True):
    """ Load all PSMs and features from a percolator input (PIN) file
        
        For n input features and m total file fields, the file format is:
        header field 1: SpecId, or other PSM id
        header field 2: Label, denoting whether the PSM is a target or decoy
        header field 3: ScanNr, the scan number.  Note this string must be exactly stated
        header field 4 (optional): ExpMass, PSM experimental mass.  Not used as a feature
        header field 4 + 1 : Input feature 1
        header field 4 + 2 : Input feature 2
        ...
        header field 4 + n : Input feature n
        header field 4 + n + 1 : Peptide, the peptide string
        header field 4 + n + 2 : Protein id 1
        header field 4 + n + 3 : Protein id 2
        ...
        header field m : Protein id m - n - 4
    """
    f = checkGzip_openfile(filename, 'rt')
    r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
    headerInOrder = r.fieldnames
    l = headerInOrder
    sids = [] # keep track of spectrum IDs and exp masses for FDR post-processing
    expMasses = [] 
    # Check that header fields follow pin schema
    # spectrum identification key for PIN files
    # Note: this string must be stated exactly as the third header field
    sidKey = "ScanNr"
    if sidKey not in l:
        raise ValueError("No %s field, exitting" % (sidKey))
    expMassKey = "ExpMass"
    if expMassKey not in l:
        raise ValueError("No %s field, exitting" % (expMassKey))
    constKeys = [l[0]]

    # denote charge keys
    maxCharge = 1
    chargeKeys = set([])
    for i in l:
        m = i.lower()
        if m[:-1]=='charge':
            try:
                c = int(m[-1])
            except ValueError:
                print("Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1))

            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    # Check label
    m = l[1]
    if m.lower() == 'label':
        constKeys.append(l[1])
    # Exclude calcmass and expmass as features
    constKeys += [sidKey, "CalcMass", "ExpMass"]
    # Find peptide and protein ID fields
    psmStrings = [l[0]]
    isConstKey = False
    for key in headerInOrder:
        m = key.lower()
        if m=="peptide":
            isConstKey = True
        if isConstKey:
            constKeys.append(key)
            psmStrings.append(key)    
    constKeys = set(constKeys) # exclude these when reserializing data
    keys = []
    for h in headerInOrder: # keep order of keys intact
        if h not in constKeys:
            keys.append(h)
    
    featureNames = []
    for k in keys:
        featureNames.append(k)  
    targets = {}  # mapping between sids and indices in the feature matrix
    decoys = {}
    X = [] # Feature matrix
    Y = [] # labels
    pepstrings = []
    scoreIndex = _scoreInd # column index of the ranking score used by the search algorithm 
    numRows = 0
    # for i, l in enumerate(reader):
    for i, l in enumerate(r):
        try:
            sid = int(l[sidKey])
        except ValueError:
            print("Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1))
        # try:
        #     expMass = float(l[expMassKey])
        # except ValueError:
        #     print("Could not convert exp mass %s on line %d to float, exitting" % (l[expMassKey], i+1))
        expMass = l[expMassKey]
        try:
            y = int(l["Label"])
        except ValueError:
            print("Could not convert label %s on line %d to int, exitting" % (l["Label"], i+1))
        if y != 1 and y != -1:
            print("Error: encountered label value %d on line %d, can only be -1 or 1, exitting" % (y, i+1))
            exit(-1)
        el = []
        for k in keys:
            try:
                el.append(float(l[k]))
            except ValueError:
                print("Could not convert feature %s with value %s to float, exitting" % (k, l[k]))
        el_strings = [l[k] for k in psmStrings]
        if not _topPsm:
            X.append(el)
            Y.append(y)
            pepstrings.append(el_strings)
            sids.append(sid)
            expMasses.append(expMass)
            numRows += 1
        else:
            if y == 1:
                if sid in targets:
                    featScore = X[targets[sid]][scoreIndex]
                    if el[scoreIndex] > featScore:
                        X[targets[sid]] = el
                        pepstrings[targets[sid]] = el_strings
                else:
                    targets[sid] = numRows
                    X.append(el)
                    Y.append(1)
                    pepstrings.append(el_strings)
                    sids.append(sid)
                    expMasses.append(expMass)
                    numRows += 1
            elif y == -1:
                if sid in decoys:
                    featScore = X[decoys[sid]][scoreIndex]
                    if el[scoreIndex] > featScore:
                        X[decoys[sid]] = el
                        pepstrings[decoys[sid]] = el_strings
                else:
                    decoys[sid] = numRows
                    X.append(el)
                    Y.append(-1)
                    pepstrings.append(el_strings)
                    sids.append(sid)
                    expMasses.append(expMass)
                    numRows += 1
    f.close()
    
    if _standardNorm:
        return pepstrings, preprocessing.scale(np.array(X)), np.array(Y), featureNames, sids, expMasses
    else:
        min_max_scaler = preprocessing.MinMaxScaler()
        return pepstrings, min_max_scaler.fit_transform(np.array(X)), np.array(Y), featureNames, sids, expMasses

def findInitDirection(X, Y, thresh, featureNames):
    l = X.shape
    m = l[1] # number of columns/features
    initDirection = -1
    numIdentified = -1
    # TODO: add check verifying best direction idetnfies more than -1 spectra, otherwise something
    # went wrong
    negBest = False
    for i in range(m):
        scores = X[:,i]
        # Check scores multiplied by both 1 and positive -1
        for checkNegBest in range(2):
            if checkNegBest==1:
                taq, _, _ = calcQ(-1. * scores, Y, thresh, True)
            else:
                taq, _, _ = calcQ(scores, Y, thresh, True)
            if len(taq) > numIdentified:
                initDirection = i
                numIdentified = len(taq)
                negBest = checkNegBest==1
    return initDirection, numIdentified, negBest


