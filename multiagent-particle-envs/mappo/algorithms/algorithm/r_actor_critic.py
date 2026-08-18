
import torch
import torch.nn as nn
from mappo.algorithms.utils.util import init, check
from mappo.algorithms.utils.cnn import CNNBase
from mappo.algorithms.utils.mlp import MLPBase
from mappo.algorithms.utils.rnn import RNNLayer
from mappo.algorithms.utils.act import ACTLayer
from mappo.algorithms.utils.att import new_cons, Res
from mappo.algorithms.utils.popart import PopArt
from mappo.utils.util import get_shape_from_obs_space
from copy import deepcopy

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_attn = args.use_attn
        self._att_hidden = args.att_hidden
        self._n_agent = args.num_agents
        self._n_roll = args.n_rollout_threads
        self.count = 0
        self.data_size = 0
        self.event_trigger_mask = None
        self.event_trigger_score = None
        self.event_trigger_state = None
        self.train_event_trigger_mask = None
        self.trigger_loss = None
        self.ib_mean = None
        self.ib_logvar = None
        self.ib_loss = None
        obs_shape = get_shape_from_obs_space(obs_space)

        if self._use_attn:
            print('use attn')
            obs_dim = obs_shape[0]

            self.base = new_cons(obs_dim=obs_dim, d_model=self._att_hidden,
                                 agent_num=self._n_agent, out_d=self.hidden_size)

            self.resnet = Res(obs_dim=obs_dim, d_model=self._att_hidden, out_d=self.hidden_size)

            self.resnet_mlp = MLPBase(args, obs_shape)
            self.dis = torch.zeros([self.step, self._n_roll, self._n_agent, self._n_agent])
            self.last_hid = torch.zeros([self.step + 1, self._n_roll, self._n_agent, self.hidden_size])

            # for train
            self.last_hid_tr = torch.zeros([self.step, self._n_roll, self._n_agent, self.hidden_size])
            self.curr_hid_tr = torch.zeros([self.step, self._n_roll, self._n_agent, self.hidden_size])
            self.est_temp_tr = torch.zeros([self.step, self._n_roll, self._n_agent, obs_dim])
     

        else:

            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def init_att(self):
        self.count = 0
        self.event_trigger_state = None

    def _event_trigger(self, actor_features, training=False):
        flat_mask = actor_features.new_zeros((actor_features.shape[0], 1))
        self.event_trigger_score = None
        self.event_trigger_state = None
        self.trigger_loss = None
        if training:
            self.train_event_trigger_mask = flat_mask
        else:
            self.event_trigger_mask = flat_mask

        return flat_mask.detach().cpu()

    def _information_bottleneck(self, features):
        self.ib_mean = None
        self.ib_logvar = None
        self.ib_loss = None
        return self.ib_loss

    def forward(self, step, obs, rnn_states, masks,
                available_actions=None, deterministic=False, hid_state=None):

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        self.count += 1
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self._use_attn:

            hid_state = check(hid_state).to(**self.tpdv)
            obs_dis = obs[:, -self._n_agent:]
            obs_in = obs[:, :-self._n_agent]
            dis_in = obs_dis.sort(dim=1)[0]
            attn_features, _ = self.base(obs=obs_in, last_hid=hid_state, dis=dis_in)
            resd = self.resnet_mlp(obs_in)
            wei = self.resnet(obs=obs_in)
            actor_features = wei * attn_features.detach() + resd
            self.data_size = sum(sum(attn_features))/140

            hid_s_ori = actor_features.reshape(self._n_roll, self._n_agent, -1)
            hid_s_t = hid_s_ori.repeat(1, self._n_agent, 1)\
                .reshape(self._n_roll, self._n_agent, self._n_agent, self.hidden_size)  
            index_dis = obs_dis.argsort(dim=1).reshape(self._n_roll, self._n_agent, -1)
            att_dis = obs_dis.reshape(self._n_roll, self._n_agent, -1)
            
            
            for i in range(self._n_roll):
                hid_temp = deepcopy(hid_s_ori[i])
                for agent in range(self._n_agent):
                    inx_temp = index_dis[i, agent, :]
                    for a in range(self._n_agent):
                        #print(att_dis[i, agent, inx_temp[a]])
                        if att_dis[i, agent, inx_temp[a]] < self.cl or a < 2: #< self._n_agent/2:
                            #hid_s_t[i, agent, inx_temp[a], :] = hid_temp[inx_temp[a]]
                            hid_s_t[i, agent, a, :] = hid_temp[inx_temp[a]]
                        else:
                            #hid_s_t[i, agent, inx_temp[a], :] = torch.zeros(self._att_hidden)
                            hid_s_t[i, agent, a, :] = torch.zeros(self._att_hidden)
            hid_state = hid_s_t.reshape(self._n_roll * self._n_agent, self._n_agent, self.hidden_size)


        else:

            obs = obs[:, :-self._n_agent]

            actor_features = self.base(obs)

            hid_s_ori = actor_features.reshape(self._n_roll, self._n_agent, -1)
            hid_s_t = hid_s_ori.repeat(1, self._n_agent, 1)\
                .reshape(self._n_roll, self._n_agent, self._n_agent, self.hidden_size)
            hid_state = hid_s_t.reshape(self._n_roll * self._n_agent, self._n_agent, self.hidden_size)

        trigger_door = self._event_trigger(actor_features, training=False)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        trigger_door_count = trigger_door

        return actions, action_log_probs, rnn_states, hid_state, trigger_door_count

    def evaluate_actions(self, cent_obs, obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None, hid_state=None):
        cent_obs = check(cent_obs).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        if self._use_attn:

            hid_state = check(hid_state).to(**self.tpdv)
            obs_dis = obs[:, -self._n_agent:]
            obs_in = obs[:, :-self._n_agent]
            dis_in = obs_dis.sort(dim=1)[0]

            attn_features, self.est_temp = self.base(obs=obs_in, last_hid=hid_state, dis=dis_in)

            resd = self.resnet_mlp(obs_in)
            wei = self.resnet(obs=obs_in)
            actor_features = wei * attn_features.detach() + resd
            self.wei = wei.detach().cpu().numpy().mean()
            self.hid_fea = actor_features
            self._event_trigger(actor_features, training=True)
            self._information_bottleneck(attn_features)
        else:
            obs = obs[:, :-self._n_agent]
            actor_features = self.base(obs)


        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        return action_log_probs, dist_entropy

class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
