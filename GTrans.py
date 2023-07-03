# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:50:04 2020

@author: a
"""

import torch
import torch.nn as nn
import numpy as np
from utils import graph_pad,graph_pad1,graph_pad2
from math import sqrt
from torch.nn import Parameter
from torch.nn import init

class GTrans1(nn.Module):
    def __init__(self):
        super(GTrans1, self).__init__()
        
        # drugs

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()


        drugSeq_vocabSize = 62
        targetSeq_vocabSize = 16693

        self.num_layers = 1
        self.num_heads = 4
        self.hidden_dim = 256
        self.inter_dim = 512
        # self.flatten_dim = 512
        self.flatten_dim = 512
        self.multi_hop_max_dist = 20
        self.dff = 512
        self.gelu=nn.GELU
        # dropout
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.e_lap =nn.Linear(128,128)
        self.input_dropout = nn.Dropout(0.1)
        self.p_pos_emb = PositionalEncoding(self.hidden_dim,max_len=1001)
        # self.d_pos_emb = PositionalEncoding(self.hidden_dim,max_len=101)
        # Embeddings
        self.p_node_encoder = nn.Embedding(targetSeq_vocabSize, self.hidden_dim, padding_idx=0)
        # self.d_node_encoder = nn.Embedding(drugSeq_vocabSize, self.hidden_dim, padding_idx=0)

        self.d_node_encoder = nn.Embedding(512 * 9 + 1, self.hidden_dim, padding_idx=0)

        self.d_spatial_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.d_in_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
        self.d_out_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, self.num_heads)
        self.d_encoder1 = Encoder1(hidden_dim=self.hidden_dim, inter_dim=self.inter_dim,
                                  n_layers=self.num_layers, n_heads=self.num_heads)

        self.d_encoder2 = Encoder2(hidden_dim=self.hidden_dim, inter_dim=self.inter_dim,
                                  n_layers=self.num_layers, n_heads=self.num_heads)

        self.ffn = FeedForwardNetwork(hidden_size=self.hidden_dim, ffn_size=self.inter_dim)
        self.p_att = Parameter(torch.tensor([1 / 1, 1 / 1]))
        self.d_att = Parameter(torch.tensor([1 / 1, 1 / 1]))




        self.icnn = nn.Conv1d(self.hidden_dim, 16, 3)

        self.d_graph_token = nn.Embedding(1, self.hidden_dim)
        self.p_graph_token = nn.Embedding(1, self.hidden_dim)
        self.gcn = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.smiles_embed = nn.Linear(78, self.hidden_dim)
        self.protein_embed = nn.Linear(1280, self.hidden_dim)
        # self.protein_embed = nn.Embedding(25, self.hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 1)

        )

    def forward(self, protein,p_bias,d_node, d_attn_bias, d_spatial_pos,d_geo_dist, d_in_degree, d_out_degree):
        drug_n_graph, drug_n_node = d_node.size()[:2]
        drug_graph_attn_bias = d_attn_bias.clone()
        drug_graph_attn_bias = drug_graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        drug_spatial_pos_bias = self.d_spatial_pos_encoder(d_spatial_pos).permute(0, 3, 1, 2)
        drug_d_geo_dist_bias = self.d_spatial_pos_encoder(d_geo_dist).permute(0, 3, 1, 2)
        # drug_graph_attn_bias[:, :, 1:, 1:] = drug_graph_attn_bias[:, :, 1:, 1:] + drug_spatial_pos_bias+drug_d_geo_dist_bias
        drug_graph_attn_bias[:, :, 1:, 1:] = drug_graph_attn_bias[:, :, 1:, 1:] +drug_d_geo_dist_bias
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1) #torch.Size([1, 4, 1])
        drug_graph_attn_bias[:, :, 1:, 0] = drug_graph_attn_bias[:, :, 1:, 0] + t
        drug_graph_attn_bias[:, :, 0, :] = drug_graph_attn_bias[:, :, 0, :] + t
        drug_graph_attn_bias = drug_graph_attn_bias + d_attn_bias.unsqueeze(1)
        drug_node_feature = self.d_node_encoder(d_node).sum(dim=-2)
        drug_node_feature = drug_node_feature + self.d_in_degree_encoder(d_in_degree) + self.d_out_degree_encoder(d_out_degree)
        drug_graph_token_feature = self.d_graph_token.weight.unsqueeze(0).repeat(drug_n_graph, 1, 1)
        drug_graph_node_feature = torch.cat([drug_graph_token_feature, drug_node_feature], dim=1)




        protein, p_bias = graph_pad4(protein,p_bias, 1000) # 32 1000 1280

        p_bias = p_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        protein = self.protein_embed(protein)  # 32 1000 256

        p_bias[:, :, 1:, 0] = p_bias[:, :, 1:, 0] + t
        p_bias[:, :, 0, :] = p_bias[:, :, 0, :] + t

        protein_graph_token_feature = self.p_graph_token.weight.unsqueeze(0).repeat(protein.size(0), 1, 1)
        protein_graph_node_feature = torch.cat([protein_graph_token_feature, protein], dim=1)
        # protein_graph_node_feature = self.p_pos_emb(protein_graph_node_feature)
        # 第一层 Transformer
        drug_out,attn = self.d_encoder1(drug_graph_node_feature,drug_graph_attn_bias)
        protein_out,attn = self.d_encoder1(protein_graph_node_feature,p_bias)


        drug_cls = drug_out[:, 0:1, :]
        protein_cls = protein_out[:, 0:1, :]

        protein = torch.concat((drug_cls,protein_out[:, 1:, :]),dim=1) # torch.Size([32, 1001, 256])
        drug = torch.concat((protein_cls,drug_out[:, 1:, :]),dim=1) # torch.Size([32, 101, 256])


        # 第二层 cross_attention

        drug_output1 = self.d_encoder2(drug,drug_graph_attn_bias)
        drug_output2,d_attn1 = self.d_encoder1(drug_output1,drug_graph_attn_bias)

        protein_output1 = self.d_encoder2(protein,p_bias)
        protein_output2,p_attn1 = self.d_encoder1(protein_output1,p_bias)




        drug_output3 = torch.max(drug_output2, dim=1)  # 32 256
        protein_out3 = torch.max(protein_output2, dim=1) # 32 256
        i = torch.cat((drug_output3.values, protein_out3.values), dim=1)  # 32 512
        # i = torch.cat((drug_output2, protein_output2), dim=1).permute(0, 2, 1)
        # i = self.dropout(i)
        # i = self.icnn(i)
        # f = i.view(protein.size(0), -1)

        score = self.decoder(i)
        score = score.squeeze(-1)


        return score,d_attn1,p_attn1







def graph_pad4(x,p_bias,maxsize):
    #x should be list   [torch(N,features)] *batch
    b = len(x)
    features = x[0].shape[1]
    out = torch.zeros(b, maxsize, features)

    bias = torch.zeros(b, maxsize+1,maxsize+1,dtype=x[0].dtype)
    for i in range(b):
        a = x[i]
        p =p_bias[i]

        out[i,:a.shape[0],:] = a
        bias[i,1:p.shape[0]+1,1:p.shape[0]+1] =p
        bias[i,p.shape[0]+1:,p.shape[0]+1:] =-10000.0
    return out.cuda(device=a.device),bias.cuda(device=a.device)

# def graph_pad4(x,p_bias,maxsize):
#     #x should be list   [torch(N,features)] *batch
#     b = len(x)
#     out = torch.zeros(b, maxsize,dtype=x[0].dtype)
#
#     bias = torch.zeros(b, maxsize+1,maxsize+1,dtype=x[0].dtype)
#     for i in range(b):
#         a = x[i]
#         p =p_bias[i]
#
#         out[i,:a.shape[0]] = a
#         bias[i,1:p.shape[0]+1,1:p.shape[0]+1] =p
#         bias[i,p.shape[0]+1:,p.shape[0]+1:] =-10000.0
#     return out.cuda(device=a.device),bias.cuda(device=a.device)







class AttentionLayer1(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AttentionLayer1, self).__init__()

        key_dim = hidden_dim // n_heads
        value_dim = hidden_dim // n_heads
        self.inner_attention = FullAttention(output_attention=True)
        self.query_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.key_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.value_projection = nn.Linear(hidden_dim, value_dim * n_heads)
        self.out_projection = nn.Linear(value_dim * n_heads, hidden_dim)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)  #torch.Size([batchsize, 151, 256])

        return self.out_projection(out), attn
class AttentionLayer2(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AttentionLayer2, self).__init__()

        key_dim = hidden_dim // n_heads
        value_dim = hidden_dim // n_heads
        # self.inner_attention1 = ProbAttention(factor=20, attention_dropout=0.1)

        self.inner_attention1 = FullAttention(output_attention=True)
        self.query_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.key_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.value_projection = nn.Linear(hidden_dim, value_dim * n_heads)
        self.out_projection = nn.Linear(value_dim * n_heads, hidden_dim)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention1(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)  #torch.Size([batchsize, 151, 256])

        return self.out_projection(out), attn




class FullAttention(nn.Module):
    def __init__(self, output_attention=True):
        super(FullAttention, self).__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(0.1)
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # batch_size, seq_len, head_num, dim_feature
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)  # 默认是没有1/sqrt(d)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # score的维度应该是[batch_size, head_num, seq_len, seq_len]
        A = torch.softmax(scale * scores+attn_mask, dim=-1)  # 为什么要进行dropout?
        # A = torch.softmax(scale * scores, dim=-1)  # 为什么要进行dropout?
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)  # torch.Size([32, 151, 8, 32])
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class Encoder1(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_layers, n_heads, dropout=0.1):
        super(Encoder1, self).__init__()
        self.attn_layers = nn.ModuleList(Encoder_layer1(hidden_dim, inter_dim, n_heads, dropout) for l in range(n_layers))
        self.norm = torch.nn.LayerNorm(hidden_dim)
    def forward(self, x, attn_mask=None):
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
        return x,attn

class Encoder_layer1(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_heads, dropout):
        super(Encoder_layer1, self).__init__()
        self.attention = AttentionLayer1(hidden_dim=hidden_dim, n_heads=n_heads)
        self.dropout =nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn =FeedForwardNetwork(hidden_size=hidden_dim,ffn_size=inter_dim)

    def forward(self, x, attn_mask=None):
        attn_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        attn_x = self.dropout(attn_x)
        x =self.norm1(x+attn_x)
        y = x
        x =self.ffn(x)

        return self.norm2.cuda()(x + y), attn

class Encoder2(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_layers, n_heads, dropout=0.1):
        super(Encoder2, self).__init__()
        self.attn_layers = nn.ModuleList(Encoder_layer2(hidden_dim, inter_dim, n_heads, dropout) for l in range(n_layers))
        self.norm = torch.nn.LayerNorm(hidden_dim)
    def forward(self, x, attn_mask=None):
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
        return x

class Encoder_layer2(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_heads, dropout):
        super(Encoder_layer2, self).__init__()
        self.attention = AttentionLayer2(hidden_dim=hidden_dim, n_heads=n_heads)
        self.dropout =nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
    def forward(self, x, attn_mask=None):
        attn_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        attn_x = self.dropout(attn_x)
        attn_x = self.norm1.cuda()(x+attn_x)

        return attn_x, attn

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        self.hidden_size = hidden_size
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.dropout=nn.Dropout(0.1)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x= self.dropout(x)
        x = self.layer2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=10, attention_dropout=0.1, output_attention=True):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # 该函数返回经过筛选后的query和key内积后的结果，以及筛选出的u-top个query的index
    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)   选择sparsity measurement最高的u个query

        B, H, L_K, E = K.shape  # batchsize * heads * 序列长度 *特征维度
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 仅计算u个query和所有key的点积，从而得到attention score
        # 其他的query对应的score是将self-attention层的输入取均值(mean(V))，来保证输入和输出序列长度都是
        # 32 8 96 96 64
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # 对k进行采样，  应该k和所有的q进行计算 ，但是 我们随机采样 原文中 96个中选取25
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # 采用sample_k 个 k     sample_k为25
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # 32 8 96 25 64
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()  # 32 8 96 25    每一个q和25个k做点积

        # 最大的减去均值， 选择前n个最大的q  。  选择重要的q
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # 32 8 96
        M_top = M.topk(n_top, sorted=False)[1]  # 32 8 25

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k  # 32 8 25 96

        return Q_K, M_top

    # 该函数的主要作用就是将V按照倒数第二维度进行均值计算，并扩展复制到多个head的维度
    def _get_initial_context(self, V, L_Q):  # 算平均值
        B, H, L_V, D = V.shape
        if not self.mask_flag:

            V_sum = V.mean(dim=-2)  # 32 8 64
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # 32 8 96 64
        else:
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    # 据论文中的公式计算ProbSparse self-attention，其中 Q表示选择出的top-u个query所组成的新Q矩阵
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)
        ### sofmax 在这，那么之前的地方就是mask部分，把这个部分用distance embedding 替换

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # torch.matmul(attn, V) 要更新的 32 9 25 64   身下96-25个直接给均值
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # q和k长度一样，那岂不是随机采样的个数和我们选的个数是一样的
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)  # v：32 8 96 64
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1001):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe
        self.register_buffer('pe', pe)

    def forward(self, x):

        x=self.pe.expand(x.size(0),1001,256)+x
        return x
