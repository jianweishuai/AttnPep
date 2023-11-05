# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:14:29 2023

@author: liyulin
"""
import numpy as np
from scipy.stats import kstest
from data_utils import load_pin_return_featureMatrix
import pandas as pd
import matplotlib.pyplot as plt
from calc_qvalues import calcQ
def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

def calcP(data):
    decoys = data[data['Y']==-1]
    targets = data[data['Y']==1]
    
    N = decoys.shape[0]
    
    P = []
    k = df.columns.get_loc("score")
    for i in range(len(targets)):
        score = targets.iloc[i,k]
        
        l = len(decoys[decoys['score']>=score])
        
        P.append(l/N)
    return P

def plot_Q_Q(P):
    
    sorted_p_values = np.sort(P)

    # 计算累积概率分位数
    n = len(sorted_p_values)
    quantiles = (np.arange(n) + 0.5) / n

    # 计算理论分位数（这里假设为均匀分布）
    theoretical_quantiles = np.linspace(0, 1, n)

    # 绘制Q-Q图
    plt.scatter(theoretical_quantiles, sorted_p_values)
    plt.xscale('log')  # 设置横坐标为对数尺度
    plt.yscale('log')  # 设置纵坐标为对数尺度
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 对角线
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sorted P-values')
    plt.title('Q-Q Plot of P-values')
#%%

path1 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_02.pin'
pep1, X1, Y1, featureNames, _, _ = load_pin_return_featureMatrix(path1)

path2 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_03.pin'
pep2, X2, Y2, featureNames, _, _ = load_pin_return_featureMatrix(path2)

path3 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_04.pin'
pep3, X3, Y3, featureNames, _, _ = load_pin_return_featureMatrix(path3)

path4 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_05.pin'
pep4, X4, Y4, featureNames, _, _ = load_pin_return_featureMatrix(path4)

path5 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_06.pin'
pep5, X5, Y5, featureNames, _, _ = load_pin_return_featureMatrix(path5)

path6 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_07.pin'
pep6, X6, Y6, featureNames, _, _ = load_pin_return_featureMatrix(path6)

path7 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_08.pin'
pep7, X7, Y7, featureNames, _, _ = load_pin_return_featureMatrix(path7)
path8 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_09.pin'
pep8, X8, Y8, featureNames, _, _ = load_pin_return_featureMatrix(path8)
path9 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_10.pin'
pep9, X9, Y9, featureNames, _, _ = load_pin_return_featureMatrix(path9)
path10 = 'C:/Users\liyulin\Desktop\entrapment\RAW_Data\OR20070924_S_mix7_11.pin'
pep10, X10, Y10, featureNames, _, _ = load_pin_return_featureMatrix(path10)

pep = np.concatenate([pep1,pep2,pep3,pep4,pep5,pep6,pep7,pep8,pep9,pep10], axis = 0)
X = np.concatenate([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10], axis = 0)
Y = np.concatenate([Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10], axis = 0)

del pep1, pep2, pep3, pep4, pep5, pep6, pep7, pep8, pep9, pep10, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, path1, path2, path3, path4, path5, path6, path7, path8, path9, path10

df1 = pd.DataFrame({'ID': pep[:,0], 'pep': pep[:,1], 'prot':pep[:,2], 'score':X[:,4] ,'Y':Y})
df2 = pd.DataFrame(X, columns=featureNames)

df = pd.concat([df1, df2], axis=1)
df = df.sort_values(by=['ID', 'score'], ascending=False)
df = df.drop_duplicates(subset='ID', keep='first')

X = df.iloc[:,5:].values
Y = df.iloc[:,4].values
#%%
df = pd.DataFrame({'ID': df.iloc[:,0], 'pep': df.iloc[:,1], 'prot':df.iloc[:,2], 'score':TransValid.iloc[:,0].values ,'Y':TransValid.iloc[:,1].values})

Target = df[df['Y'] == 1]
Decoy = df[df['Y'] == -1]



Entrapment = Target[Target['prot'].str.contains('tr')]
Target = Target[~Target['prot'].str.contains('tr')]
#%%

mean1 = np.mean(Decoy['score'].values)
mean2 = np.mean(Entrapment['score'].values)

dm = mean2 - mean1

Entrapment['score'] = Entrapment['score'].values - dm 

#%%
test = Entrapment['score'].values
mean2 = np.mean(Entrapment['score'].values)
test = 2*mean2 - test
Entrapment['score'] = test
#%%
# Entrapment['score'] += 0.05
Entrapment = Entrapment[Entrapment['score']<5]
#%%
plt.hist([Decoy['score'],Entrapment['score'],Target['score']], bins=60)
# plt.hist([Decoy['score'],Target['score']], bins=100)
# plt.hist(Decoy['score'], alpha=0.5)
# plt.hist(Entrapment['score'], alpha=0.5)
plt.show()
#%%
tr_decoy = pd.concat([Entrapment, Decoy])
taq,_,qs = calcQ(tr_decoy['score'], tr_decoy['Y'], thresh = 0.01, skipDecoysPlusOne = False, verb = -1)



P = calcP(tr_decoy)
#%%
custom_bins = np.linspace(-10, 10, 100)
custom_bins =list(custom_bins)
hist1, bins = np.histogram(Decoy['score'], bins = custom_bins)
hist2, bins = np.histogram(Entrapment['score'], bins = custom_bins)
hist3, bins = np.histogram(Target['score'], bins = custom_bins)

#%%
dhist = 4*(hist2-hist1)//5
hist2 -= dhist
#%%
plt.bar(bins[:-1], hist1,alpha=0.4,color='blue')

plt.bar(bins[:-1], hist2,alpha=0.5, color='orange')
#%%
import pickle

# 保存列表到文件
# with open('comet.pickle', 'wb') as f:
#     pickle.dump(P, f)

# 从文件加载列表
with open('AttnPep.pickle', 'rb') as f:
    P1 = pickle.load(f)

with open('comet.pickle', 'rb') as f:
    P = pickle.load(f)
# print(loaded_list)  # 输出: [1, 2, 3, 4, 5]
#%%
plot_Q_Q(P)
plot_Q_Q(P1)
#%%
AttnPep = np.sort(P1)
Comet = np.sort(P)
n = len(AttnPep)
theoretical_quantiles = np.linspace(0, 1, n)
t1 = np.linspace(0, 0.5, n)
t2 = np.linspace(0,2,n)
