import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents

        self.n = len(world.policy_agents)

        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.discrete_action_space = False

        self.discrete_action_input = False

        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False

        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        self.action_space = []
        self.observation_space = []
        for  i, agent in enumerate(self.agents):
            total_action_space = []

            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)

            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)

            if len(total_action_space) > 1:

                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            obs_dim = len(observation_callback(agent, i, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents

        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        self.world.step()

        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent, i))
            reward_n.append(self._get_reward(agent, i))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        return obs_n, reward_n, done_n, info_n

    def reset(self):

        self.reset_callback(self.world)

        self._reset_render()

        obs_n = []
        self.agents = self.world.policy_agents
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent, i))
        return obs_n

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def _get_obs(self, agent, i):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, i, self.world)

    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    def _get_reward(self, agent, i):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, i, self.world)

    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:

            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)

                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:

            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]

        assert len(action) == 0

    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')

        for i in range(len(self.viewers)):

            if self.viewers[i] is None:

                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

                self.viewers[i].set_bounds(-20, 20, -20, 20)

        if self.render_geoms is None:

            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color)
                else:
                    geom.set_color(*entity.color, alpha=0.2)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        mean_pos = np.mean([self.agents[i].state.p_pos for i in range(self.n)],axis=0)

        for i in range(len(self.viewers)):
            from multiagent import rendering

            cam_range = 3

            if self.shared_viewer:
                pos = mean_pos
            else:
                pos = self.agents[i].state.p_pos

            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)

            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)

            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []

        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))

            dx.append(np.array([0.0, 0.0]))

        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs

            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
