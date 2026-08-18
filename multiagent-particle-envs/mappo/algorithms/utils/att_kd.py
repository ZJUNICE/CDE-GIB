import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .util import init, get_clones
from mappo.utils.util import get_shape_from_obs_space
from .trans import MultiHeadedAttention

class qkv(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hid)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = th.cat([x1, x2], dim=1)
        h = self.layer_norm(self.act(self.fc1(x)))
        return self.fc2(h)

class qkv_net(nn.Module):
    def __init__(self, d_model=128, agent_num=5, head=2):
        super(qkv_net, self).__init__()
        self.d_model = d_model
        self.e_dim = int(d_model / 4)
        self.emb = DisEncode(expand_dim=self.e_dim)

        self.query_dim = self.d_model
        self.key_dim = (self.d_model + self.e_dim)
        self.merge_layer = MergeLayer(in1=self.query_dim,in2=self.query_dim,hid=d_model,out=d_model)

        self.m_a = nn.MultiheadAttention(embed_dim=self.query_dim,
                                         kdim=self.key_dim,
                                         vdim=self.key_dim,
                                         num_heads=head,
                                         dropout=0.1,
                                         batch_first=True)

    def forward(self, src, hidden, dis):

        dis_embed = self.emb(dis)
        input_src = th.cat([hidden, dis_embed], dim=2)

        attn_output, attn_output_weights = self.m_a(src, input_src, input_src)
        attn_output = attn_output.squeeze()
        out = self.merge_layer(src.squeeze(), attn_output)
        return out, attn_output

class MergeLayer(nn.Module):
    def __init__(self, in1, in2, hid, out):
        super().__init__()

        self.fc1 = nn.Linear(in1 + in2, hid)
        self.fc2 = nn.Linear(hid, out)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hid)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = th.cat([x1, x2], dim=1)

        h = self.layer_norm(self.act(self.fc1(x)))
        return self.fc2(h)

class DisEncode(nn.Module):
    def __init__(self, expand_dim):
        super(DisEncode, self).__init__()
        d_model = expand_dim

        self.w = nn.Linear(1, d_model)
        self.w.weight = nn.Parameter((th.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float().reshape(d_model, -1))
        self.w.bias = nn.Parameter(th.zeros(d_model).float())

    def forward(self, dis):

        batch_size = dis.size(0)
        seq_len = dis.size(1)
        dis = dis.view(batch_size, seq_len, 1)
        harmonic = th.cos(self.w(dis))
        return harmonic

class cons_kd(nn.Module):
    def __init__(self, obs_dim, d_model, agent_num):
        super(cons_kd, self).__init__()
        self.merge_layer = MergeLayer(in1=obs_dim, in2=d_model, hid=d_model, out=d_model)
        self.att = qkv_net(d_model=d_model, agent_num=agent_num, head=4)

    def forward(self, obs, last_hid, dis):

        b, n, d = last_hid.size()
        index = th.argmin(dis, dim=1)
        temp = th.zeros([b, d])

        for i in range(b):
            temp[i] = last_hid[i, index[i], :]
        m_curr = self.merge_layer(obs, temp)

        m_curr = m_curr.unsqueeze(dim=1)
        m_new, _ = self.att(src=m_curr, hidden=last_hid, dis=dis)
        return m_new
