from __future__ import print_function,division
import numpy as np
import torch
import torch.utils.data
from math import sqrt
from scipy import stats
import pandas as pd
import os
from rdkit import Chem
import algos
from rdkit.Chem import AllChem

class DrugTargetDataset(torch.utils.data.Dataset):

    def __init__(self, X0, X1, Y, pid,drug_id):
        self.X0 = X0 #drug  SMILES
        self.X1 = X1 #protein
        self.Y = Y  # affinity
        self.pid = pid
        self.drug_id = drug_id



    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        prot = self.X1[i]
        pro_key=self.pid[i]
        drug_key=self.drug_id[i]
        protein,p_bias = protein_embedding(pro_key)
        node ,attn_bias,spatial_pos,Geo_dist,in_degree,out_degree = drug_embedding(self.X0[i],drug_key)
        return [protein,p_bias,node ,attn_bias,spatial_pos,Geo_dist,in_degree,out_degree, self.Y[i]]



allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
}

def atom_to_feature_vector(atom):
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature
def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

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
    dist=torch.cdist(coord,coord,2)
    # knn=dist.argsort(1)[:,1:k+1]
    knn=dist.argsort(1)[:,:k]
    for i in range(nodes):
        for j in range(nodes):
            if j in knn[i]:
                continue
            else:
                dist[i][j]=0
    return dist




def protein_embedding(prot_key):

    # pdb_file = 'data/davis/davis_struct_txt/'+prot_key+'.txt'
    # np_pos = protein_xyz(pdb_file)
    # pos = torch.Tensor(np_pos)
    # bias = Proteintograph(pos)

    bias =np.load('data/davis/mask/'+prot_key+'.npy')
    # bias =np.load('data/davis/mask/'+prot_key+'.npy')
    bias=torch.Tensor(bias)


    node_dir = 'data/davis/esm-1/'
    target_file = os.path.join(node_dir, prot_key + '.npy')
    target_feature =np.load(target_file)

    protein =torch.tensor(target_feature)

    return protein,bias

def drug_xyz(smile):
   mol = Chem.MolFromSmiles(smile)
   m = Chem.AddHs(mol)
   AllChem.EmbedMolecule(m)
   AllChem.MMFFOptimizeMolecule(m)
   m = Chem.RemoveHs(m)
   m_con = m.GetConformer(id=0)
   pos = []
   for j in range(m.GetNumAtoms()):
      pos.append(list(m_con.GetAtomPosition(j)))

   np_pos = np.array(pos)

   return np_pos

def Smilestograph(coord,k=10):
    nodes=coord.shape[0]
    dist=torch.cdist(coord,coord,2)
    knn=dist.argsort(1)[:,:k]
    for i in range(nodes):
        for j in range(nodes):
            if j in knn[i]:
                continue
            else:
                dist[i][j]=0
    return dist


def drug_embedding(SMILES,drug_key):
    mol = Chem.MolFromSmiles(SMILES)
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)
    x = torch.tensor(x, dtype=torch.long)
    N = x.size(0)
    x = mol_to_single_emb(x)

    np_pos = drug_xyz(SMILES)   # n*3
    np_pos  = torch.tensor(np_pos)
    d_dist  = Smilestograph(np_pos)

    Geo_dist  = torch.tensor(d_dist,dtype=torch.long)
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges_list.append((i, j))
            edges_list.append((j, i))


        edge_index = np.array(edges_list, dtype=np.int64).T

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)


    x = torch.tensor(x, dtype=torch.long)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    node = x
    attn_bias = attn_bias
    spatial_pos = spatial_pos
    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)


    return node ,attn_bias,spatial_pos,Geo_dist,in_degree,out_degree

def mol_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def collate(args):
    x0 = [a[0] for a in args]
    x0_bias = [a[1] for a in args]
    d1_node, d1_attn_bias, d1_spatial_pos,d1_geo_dist, d1_in_degree, d1_out_degree = [], [], [], [], [],[]
    Y = []

    max_d_node = 100
    spatial_pos_max = 10
    for p,p_bias, d_node, d_attn_bias, d_spatial_pos,Geo_dist, d_in_degree, d_out_degree, y in args:
        if d_node.size(0) > max_d_node:
            d_node =d_node[:100,:]
            d_attn_bias =d_attn_bias[:101,:101]
            d_spatial_pos =d_spatial_pos[:100,:100]
            Geo_dist = Geo_dist[:100,:100]
            d_in_degree =d_in_degree[:100]
            d_out_degree =d_out_degree[:100]

        d1_node.append(d_node)
        d_attn_bias[1:, 1:][d_spatial_pos >= spatial_pos_max] = float('-inf')
        d1_attn_bias.append(d_attn_bias)
        d1_spatial_pos.append(d_spatial_pos)
        d1_geo_dist.append(Geo_dist)
        d1_in_degree.append(d_in_degree)
        d1_out_degree.append(d_out_degree)
        Y.append(y)

    d1_node = torch.cat([pad_2d_unsqueeze(i, max_d_node) for i in d1_node])
    d1_attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_d_node + 1) for i in d1_attn_bias])
    d1_spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_d_node) for i in d1_spatial_pos])
    d1_geo_dist = torch.cat([pad_spatial_pos_unsqueeze(i, max_d_node) for i in d1_geo_dist])
    d1_in_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node) for i in d1_in_degree])
    d1_out_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node) for i in d1_out_degree])


    y = [a[8] for a in args]
    return x0,x0_bias, d1_node, d1_attn_bias, d1_spatial_pos,d1_geo_dist, d1_in_degree, d1_out_degree, torch.stack(y, 0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)
def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)




def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


# def r_squared_error(y_obs,y_pred):
#     y_obs = np.array(y_obs)
#     y_pred = np.array(y_pred)
#     y_obs_mean = [np.mean(y_obs) for y in y_obs]
#     y_pred_mean = [np.mean(y_pred) for y in y_pred]
#
#     mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
#     mult = mult * mult
#
#     y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
#     y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )
#
#     return mult / float(y_obs_sq * y_pred_sq)

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0000000001
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def adj_mask(adj, maxsize):
    #adj should be list   [torch(N,N)] *batch
    b = len(adj)
    out = torch.zeros(b, maxsize, maxsize) #(b, N, N)
    for i in range(b):
        a = adj[i]
        out[i,:a.shape[0],:a.shape[1]] = a
    return out.cuda()

def graph_pad2(x, maxsize):
    #x should be list   [torch(N,features)] *batch
    b = len(x)
    out = torch.zeros(b, maxsize,dtype=x[0].dtype)
    bias = torch.zeros(b, maxsize+1,maxsize+1,dtype=x[0].dtype)
    for i in range(b):
        a = x[i]

        out[i,:a.shape[0]] = a
        bias[i,a.shape[0]:,a.shape[0]:] =-10000.0
    return out.cuda(device=a.device),bias.cuda(device=a.device)

def graph_pad1(x, maxsize):
    #x should be list   [torch(N,features)] *batch
    b = len(x)
    out = torch.zeros(b, maxsize, maxsize)
    for i in range(b):
        a = x[i]
        out[i,:a.shape[0],:a.shape[1]] = a
    return out.cuda()




def graph_pad(x, maxsize):
    #x should be list   [torch(N,features)] *batch
    b = len(x)
    features = x[0].shape[1]
    out = torch.zeros(b, maxsize, features)
    bias = torch.zeros(b, maxsize+1,maxsize+1,dtype=x[0].dtype)
    for i in range(b):
        a = x[i]
        out[i,:a.shape[0]] = a
        bias[i,a.shape[0]:,a.shape[0]:] =-10000.0
    return out.cuda(device=a.device),bias.cuda(device=a.device)

def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        #print(summ)
        #print(pair)
        return summ/pair
    else:
        return 0



def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]




def getdata_from_csv(fname):
    df = pd.read_csv(fname)
    smiles = list(df['compound_iso_smiles'])
    drug_id = list(df['drug_id'])
    protein = list(df['target_sequence'])
    affinity = list(df['affinity'])
    pid = list(df['target_key'])
    # pid = list(df['prot_key'])
    return smiles, protein, affinity, pid,drug_id


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))






