import gym
import numpy as np
from copy import deepcopy
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import spaces

def m_env(scenario_name, num=7, cl=10, benchmark=False,
          discrete_action=False, discrete_action_input=False
               ):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(num=num, cl=cl)
    if benchmark:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            info_callback=scenario.benchmark_data,
                            )
    else:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            )
    return env

class EnvCore(object):

    def __init__(self, args):
        self.agent_num = args.num_agents
        self.gui = True
        self.world =  m_env('simple_spread',num=args.num_agents,cl=args.comm_dis,
                             benchmark=False, discrete_action=False,
                             discrete_action_input=False)
        self.action_dim = self.world.action_space[0].shape[0]

        self.obs_dim = 4*self.agent_num
        self.single_dim = 4
        self.cl = args.comm_dis

    def obs_re(self, obs):
        share_state = np.zeros([self.agent_num, self.single_dim])

        for i in range(self.agent_num):
            share_state[i, :] = obs[i][:4]

        share_state = share_state.reshape(-1)

        return obs, share_state

    def reset(self):
        obs = self.world.reset()

        real_state, share_obs = self.obs_re(obs)
        sub_agent_obs = []
        share_obs = np.concatenate([share_obs, np.zeros(self.world.observation_space[0].shape[0] - self.obs_dim)])

        sub_agent_obs.append(share_obs)
        for i in range(self.agent_num):
            sub_obs = real_state[i]
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        input_act = deepcopy(actions)
        obs_, reward_, _, _ = self.world.step(input_act)

        real_state, share_obs = self.obs_re(obs_)

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        share_obs = np.concatenate([share_obs, np.zeros(self.world.observation_space[0].shape[0] - self.obs_dim)])
        sub_agent_obs.append(share_obs)

        for i in range(self.agent_num):
            sub_agent_obs.append(real_state[i])
            sub_agent_reward.append([reward_[i]])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def close(self):
        self.world.close()

class PybulletEnv(object):
    def __init__(self, args):
        self.env = EnvCore(args)
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        self.discrete_action_input = False

        self.movable = True

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0

        for agent in range(self.num_agent):
            total_action_space = []
            if self.discrete_action_input:

                u_action_space = spaces.Discrete(self.signal_action_dim)
            else:
                u_action_space = spaces.Box(low=-1., high=1., shape=(self.signal_action_dim,),
                                                     dtype=np.float32)

            if self.movable:
                total_action_space.append(u_action_space)

            if len(total_action_space) > 1:

                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):

                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:

                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            self.observation_space.append(spaces.Box(low=-10., high=10., shape=(self.signal_obs_dim,),
                                                     dtype=np.float32))
        share_obs_dim = self.signal_obs_dim
        self.share_observation_space = [spaces.Box(low=-10., high=10., shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        self.env.close()

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass

class MultiDiscrete(gym.Space):

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):

        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)

if __name__ == "__main__":
    PybulletEnv().step(actions=None)
