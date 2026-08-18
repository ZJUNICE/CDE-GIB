
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import namedtuple
import math, copy, time
from torch.autograd import Variable
import numpy as np

_iscomplex = True

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, dense):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.channel_dim = 16
        self.num_hidden = 128
        self.from_channel_emb = nn.Sequential(nn.Linear(self.channel_dim, self.num_hidden * 2), nn.ReLU(),
                                              nn.Linear(self.num_hidden * 2, self.num_hidden))
        self.dense = dense

    def encode(self, src, src_mask):
        return self.encoder.forward(self.src_embed.forward(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder.forward(self.tgt_embed.forward(tgt), memory, src_mask, tgt_mask)

class Denoiser(nn.Module):

    def __init__(self, denoise1, denoise2, denoise3, denoise4):
        super(Denoiser, self).__init__()
        self.denoise1 = denoise1
        self.denoise2 = denoise2
        self.denoise3 = denoise3
        self.denoise4 = denoise4

    def denoise11(self, memory, snr):
        return self.denoise1(memory, snr)

    def denoise12(self, memory, snr):
        return self.denoise2(memory, snr)

    def denoise13(self, memory, snr):
        return self.denoise3(memory, snr)

    def denoise14(self, memory, snr):
        return self.denoise4(memory, snr)

class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):

    def __init__(self, layer, N, hidden_size, act=False):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.norm1 = LayerNorm(16)
        self.act = act
        self.num_layers = N
        self.hidden_size = hidden_size
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                                                  nn.Linear(256, 16))
        self.positionalEncoding = PositionalEncoding(128, 0)

        if (self.act):
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, x, mask):
        if self.act == False:
            for cishu in range(0, 2):
                for layer in self.layers:
                    x = layer(x, mask)

            x = self.to_chanenl_embedding(x)
            return self.norm1(x)
        else:
            x, (remainders, n_updates) = self.act_fn(x, x, mask, None, self.layers, self.num_layers)
            x = self.to_chanenl_embedding(x)

            return self.norm1(x)

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder2(nn.Module):

    def __init__(self, decoder, tgt_embed, generator, N1):
        super(Decoder2, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(N1, 256), nn.ReLU(),
                                                  nn.Linear(256, 128))

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder.forward(self.tgt_embed.forward(tgt), memory, src_mask, tgt_mask)

class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))\
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:

            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value =\
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous()\
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Dense1(nn.Module):

    def __init__(self, n1, L):
        super(Dense1, self).__init__()
        self.layer1 = nn.Linear(n1, 1)
        self.layer2 = nn.Linear(L + 1, 1)
        self.layer2.bias.data.fill_(9)
        self.device = torch.device("cuda:0")

    def forward(self, x, _snr):
        x = self.layer1(x)
        snr = torch.tensor(_snr)
        snr1 = torch.zeros(x.shape[0], 1, 1)
        snr = snr + snr1
        snr = snr.to(self.device)
        x = torch.cat((x, snr), 1)
        x = torch.squeeze(x)
        x = self.layer2(x)

        return x

class Denoise1(nn.Module):

    def __init__(self, n1, L):
        super(Denoise1, self).__init__()
        self.layer1 = nn.Linear(n1, 1)
        self.layer2 = nn.Linear(L, L - 1)
        self.device = torch.device("cuda:0")

    def forward(self, x, _snr):
        x = self.layer1(x)
        snr = torch.tensor(_snr).to(self.device)
        snr1 = torch.zeros(x.shape[0], 1, 1).to(self.device)
        snr = snr + snr1
        snr = snr.to(self.device)
        x = torch.cat((x, snr), 1)
        x = torch.squeeze(x)
        x = self.layer2(x)
        x = torch.unsqueeze(x, -1)

        return x

def make_denoiser(N1=16, N2=32):
    model = Denoiser(Denoise1(N1, N2), Denoise1(N1, N2), Denoise1(N1, N2), Denoise1(N1, N2))
    return model

class Denoiser1(nn.Module):

    def __init__(self, layer, N, hidden_size):
        super(Denoiser1, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.num_layers = N
        self.hidden_size = hidden_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, None)
        return self.norm(x)

def make_denoiser1(N=3, N1=32,
                   d_model=256, d_ff=1024, h=8, dropout=0.1, act1=False,
                   act2=False):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = Denoiser1(EncoderLayer(d_model, c(attn), c(ff), dropout), N, d_model)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
