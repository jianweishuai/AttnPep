#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:54:55 2022

@author: sherlocked
"""

"""
input: DataFrame['protein','peptide']
"""

import re
from operator import mul
from functools import reduce
import pandas as pd

def Graph(ppm):
    prot_list, prot_dict, pep_dict, ppm = graph_PPMs(ppm)
    graph, collapse_list = merge(prot_dict, pep_dict)
    group_list = subgraph(prot_dict, pep_dict)
    minimal_protein_list = reduce_group(group_list,prot_dict, pep_dict)
    final_prot_list = final_report(minimal_protein_list, prot_list,collapse_list)
    final_prot = calc_prob(ppm,final_prot_list)
    return final_prot

def graph_PPMs(ppm):
    ppm = ppm.explode("protein")
    ppm = ppm.sort_values(by="peptideprophet_probability" ,ascending=False) 
    ppm = ppm.drop_duplicates(subset=['peptide','protein'],keep='first').reset_index(drop=True)
    
    prot_list = list(set(ppm['protein'].tolist()))
    pep_list = list(set(ppm['peptide'].tolist()))
    
    prot_dict = {}
    pep_dict = {}
    for i,x in enumerate(prot_list):
        prot_dict['P'+str(i)] = sorted(['p'+str(pep_list.index(xx['peptide'])) for i,xx in ppm.iterrows() if xx['protein']==x])

    for i,x in enumerate(pep_list):
        pep_dict['p'+str(i)] = sorted(['P'+str(prot_list.index(xx['protein'])) for i,xx in ppm.iterrows() if xx['peptide']==x])
    
    # graph=dict(pep_dict, **prot_dict)
    
    return prot_list, prot_dict, pep_dict, ppm

def merge(prot_dict, pep_dict):
    graph=dict(pep_dict, **prot_dict)
    del_list = []
    collapse_list = {}
    for p, ls in graph.items():
        if len(ls) > 1:
            collapse_list, del_list = collapse_and_del(ls, graph,del_list, collapse_list)
    
    for key in list(graph.keys()):
        if key in del_list:
            graph.pop(key)
        else:
            graph[key] = list(set(graph[key]).difference(set(del_list)))
    return graph, collapse_list

def subgraph(prot_dict, pep_dict):
    graph=dict(pep_dict, **prot_dict)
    explored_list = []
    group_list = []
    for key in list(graph.keys()):    
        if key not in explored_list:
            group = []
            DFS(graph,key,group)
            group_list.append(group)
            explored_list.extend(group)
    return group_list

def reduce_group(group_list,prot_dict, pep_dict):
    minimal_protein_list = []
    for ls in group_list:
        unflag = list(set(ls) & set(list(pep_dict.keys())))
        prot_ls = list(set(ls) & set(list(prot_dict.keys())))
        min_prot_list = []
        # most = 0
        while len(unflag):
            most = 0
            for prot in  prot_ls:                
                if prot not in min_prot_list and len(prot_dict.get(prot)) > most:
                    most = len(prot_dict.get(prot))
                    most_prot_now = prot
            min_prot_list.append(most_prot_now)
            unflag = list(set(unflag) - set(prot_dict.get(most_prot_now)))
        minimal_protein_list.extend(min_prot_list)
    return minimal_protein_list

def final_report(minimal_protein_list, prot_list,collapse_list):
    final_prot_list = []
    for prot in minimal_protein_list:
        final_prot_list.append(prot)
        if prot in list(collapse_list.keys()):
            final_prot_list.extend(collapse_list.get(prot))#%%

    final_prot_list = [int(re.findall("\d+",x)[0]) for x in final_prot_list]
    final_prot_list = [prot_list[i] for i in final_prot_list]
    
    return final_prot_list

def calc_prob(ppm,final_prot_list):
    final_prob=[]
    for i in range(len(final_prot_list)):
        test = ppm[ppm['protein'] == final_prot_list[i]]
        test1 = (1-test['peptideprophet_probability']).tolist()
        final_prob.append(1 - reduce(mul, test1))
    final_prot = pd.concat([pd.DataFrame(final_prot_list,columns=['prot']), pd.DataFrame(final_prob,columns=['prob'])],axis=1).sort_values(by="prob" ,ignore_index = True,ascending=False)

    return final_prot

def DFS(graph,x,list):   
    i = 0   
    for y in graph[x]:   
        i += 1
        if y not in list:  
            list.append(y)
            DFS(graph,y,list)   
        else:
            if i == len(graph[x]):
                return
    return

def collapse_and_del(ls, dct,del_list,collapse_list):
    for i in range(len(ls)):
        value = dct[ls[i]]
        for j in range(i+1,len(ls)):
            value1 = dct[ls[j]]
            if value == value1:
                if min(ls[i],ls[j]) not in collapse_list and min(ls[i],ls[j]) not in del_list:
                    # explored_list.extend([ls[i],ls[j]])
                    if max(ls[i],ls[j]) not in del_list:
                        del_list.append(max(ls[i],ls[j]))
                    collapse_list[min(ls[i],ls[j])] = [max(ls[i],ls[j])]
                elif min(ls[i],ls[j]) in collapse_list:
                    if max(ls[i],ls[j]) not in del_list:
                        del_list.append(max(ls[i],ls[j]))
                    
                        collapse_list[min(ls[i],ls[j])] += [max(ls[i],ls[j])]
    return collapse_list, del_list

#%%
# """test"""
# import pandas as pd


# if __name__ == '__main__':
#     data = [['pro1','pep3'],['pro1','pep4'],['pro1','pep7'],['pro1','pep9'],['pro1','pep8'],['pro2','pep8'],
#             ['pro8','pep8'],['pro3','pep6'],['pro4','pep2'],['pro4','pep10'],['pro9','pep10'],['pro9','pep2'],
#             ['pro5','pep4'],['pro5','pep8'],['pro6','pep6'],['pro6','pep2'],['pro7','pep1'],['pro7','pep5']]
#     data = pd.DataFrame(data,columns=['protein','peptide'],dtype=str)
    
#     prot_list = Graph(data)