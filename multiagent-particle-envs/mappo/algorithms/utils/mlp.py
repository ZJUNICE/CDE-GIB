import torch.nn as nn
from .util import init, get_clones
import torch as th

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x

class MLPSimple(nn.Module):
    def __init__(self,  obs_dim, hidden_size, layer):
        super(MLPSimple, self).__init__()
        self.feature_norm = nn.LayerNorm(obs_dim)
        self.mlp = MLPLayer(obs_dim, hidden_size, layer, True, True)
    def forward(self, x):
        x = self.feature_norm(x)
        x = self.mlp(x)
        return x

class GRUAll(nn.Module):
    def __init__(self, in1, in2, hid, out):
        super().__init__()

        self.fc1 = nn.Linear(in1 + in2, 512)
        self.fc2 = nn.Linear(hid, out)
        self.fc3 = nn.Linear(512,hid)
        self.act = nn.ReLU()
        self.act1 = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hid)
        self.layer_norm_2 = nn.LayerNorm(out)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = th.cat([x1, x2], dim=1)

        h = self.act(self.fc1(x))

        h1 = nn.Tanh()(self.fc3(h))
        h = nn.Tanh()(self.fc2(h1))

        return h
