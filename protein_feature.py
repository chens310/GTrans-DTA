import csv
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import esm
from rdkit import Chem


def generate_protein_pretraining_representation(dataset, prots,key):
    prots_tuple = [(str(i), prots[i][:1000]) for i in range(len(prots))]
    i = 0
    batch = 1
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    while (batch * i) < len(prots):
        print('converting protein batch: ' + str(i))
        if (i + batch) < len(prots):
            pt = prots_tuple[batch * i:batch * (i + 1)]
            prot_key = key[batch * i:batch * (i + 1)]
        else:
            pt = prots_tuple[batch * i:]
            prot_key=key[batch * i:]
        batch_labels, batch_strs, batch_tokens = batch_converter(pt)
        print(prot_key[0])  #AAK1
        print(len(batch_strs[0]))  # 961
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].numpy()
        output=token_representations[0,1: len(batch_strs[0]) + 1]
        i += 1
        np.save('data/'+dataset+'/esm-1/'+prot_key[0]+'.npy',output)
        a=np.load('data/' + dataset + '/esm-1/' + prot_key[0] + '.npy')
        a = torch.Tensor(a)
        print(a.size())

datasets = ['davis', 'kiba']
for dataset in datasets:
    fpath = 'data/' + dataset + '/'
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)

    prots = []
    key = []
    for t in proteins.keys():
        prots.append(proteins[t])
        key.append(t)
    print(prots)
    generate_protein_pretraining_representation(dataset,prots,key)


def generate_protein_PDB(dataset, prots, key):
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    with torch.no_grad():
        output = model.infer_pdb(prots)
    with open('data/' + dataset + '/' + key + '.pdb', "a") as f:
        f.write(output)


datasets = ['davis', 'kiba']
for dataset in datasets:
    fpath = 'data/' + dataset + '/'
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    for t in proteins.keys():
        print(t)
        prots = proteins[t][0:1000]
        generate_protein_PDB(dataset, prots, t)
#


## protein spatial relative position
def protein_xyz(pdb_file):
    sum=0
    pos=[]
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if(lines[i][11:16].strip()=='CA'):
            x = float(lines[i][28:38].strip())
            y=float(lines[i][38:46].strip())
            z=float(lines[i][46:54].strip())
            pos.append([x,y,z])
    np_pos = np.array(pos)
    return np_pos
def Proteintograph(coord,k=30):
    nodes=coord.shape[0]
    dist=torch.cdist(coord,coord,2)  # 欧式距离   n*n
    knn=dist.argsort(1)[:,:k] # 距离上最近的，10个原子 的下标保存起来。n*30
    for i in range(nodes):
        for j in range(nodes):
            if j in knn[i]:
                continue
            else:
                dist[i][j]=0
    return dist

def protein_embedding(prot_key):

    pdb_file = 'data/davis/davis_struct_txt/'+prot_key+'.txt'
    np_pos = protein_xyz(pdb_file)  # 计算每个氨基酸的Ca原子坐标
    pos = torch.Tensor(np_pos)     # 转换成tensor
    bias = Proteintograph(pos)
    print(prot_key)
    bias = np.array(bias)
    np.save('data/davis/mask/'+prot_key+'.npy',bias)

    return bias
datasets = ['kiba']
for dataset in datasets:
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    prots = []
    for t in proteins.keys():

        bias=protein_embedding(t)
        prots.append(proteins[t])
