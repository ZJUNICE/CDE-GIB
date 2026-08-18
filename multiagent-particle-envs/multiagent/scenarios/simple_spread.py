import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy.spatial.distance import directed_hausdorff
import train
class Scenario(BaseScenario):
    def make_world(self, num=7, cl=10):
        world = World()

        world.dim_c = 2
        self.num_agents = num
        self.cl = cl
        self.dis_p = 4
        world.collaborative = False

        world.agents = [Agent() for i in range(self.num_agents)]

        if num == 9:
            self.dis_p = 6
            gt = np.array([[0., 1.],
                    [-1., 0.], [0., 0.], [1., 0.],
                    [-2., -1.], [-1., -1.], [0., -1.], [1., -1.], [2., -1.]])

            gt = np.array([[0., 2.],
                    [-1., 1.], [1., 1.],
                    [-2., 0.], [0., 0.], [2., 0.],
                    [-2., -1.], [0., -1.], [2., -1.]])
        elif self.num_agents == 5:
            gt = np.array([[-1., 1., 1.], [1., 1., 1.],
                           [0., 0., 1.],
                           [-1., -1., 1.], [1., -1., 1.]])

        elif self.num_agents == 6:
            gt = np.array([[0., 1., 1.],
                           [-1., 0., 1.],[0., 0., 1.],[1., 0., 1.],
                           [-2., -1., 1.],            [2., -1., 1.]])

        elif self.num_agents == 7:
            gt = np.array([[0., 1., 1.],
                           [-1., 0., 1.], [0., 0., 1.], [1., 0., 1.],
                           [-2., -1., 1.], [0., -1., 1.], [2., -1., 1.]])

        elif self.num_agents == 8:

            gt = np.array([[0., 1.5, 1.],
                           [-1., 0.5, 1.],[0., 0.5, 1.],[1., 0.5, 1.],
                           [-1., -0.5, 1.],[0., -0.5, 1.], [1., -0.5, 1.], [0., -1.5, 1.]])

        else:
            raise RuntimeError('Num Error')
        self.ideal_shape = gt[:,0:2]
        self.target_pos = np.array([0, 10])
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
            agent.u_range = 1

        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
           landmark.name = 'landmark %d' % i
           landmark.collide = False
           landmark.movable = False
           landmark.size = 0.01

        self.reset_world(world)
        return world

    def reset_world(self, world):

        self.l = 0
        for i, agent in enumerate(world.agents):

            agent.color = np.array([0.85, 0.95, 0.25])

        for i, landmark in enumerate(world.landmarks):
           landmark.color = np.array([0.25, 0.25, 0.25])

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([0, 10])

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        return dist < (agent1.size + agent2.size)/2

    def reward(self, agent, i, world):
        rew = 0
        rew += self.rew

        if agent.collide:
            for a in world.agents:
                if agent!=a and self.is_collision(a, agent):
                    rew -= 6
        return rew

    def observation(self, agent, i, world):

        if i == 0:
            self.caculate_dis(world)
            positions = [agent.state.p_pos for agent in world.agents]

        other_pos = np.zeros([self.num_agents, 4])
        p_t = np.zeros(self.num_agents)
        p_in = np.zeros([self.num_agents, 4])

        index = self.pos[i].argsort()

        for j in range(self.num_agents):

            other_pos[j, 0:2] = world.agents[index[j]].state.p_pos
            other_pos[j, 2:4] = world.agents[index[j]].state.p_vel

            if self.pos[i][index[j]] < self.cl or j < 2:
                p_t[index[j]] = self.pos[i][index[j]]
                p_in[j, 0:2] = world.agents[index[j]].state.p_pos
                p_in[j, 2:4] = world.agents[index[j]].state.p_vel

            else:
                p_t[index[j]] = 10

        return np.concatenate((other_pos, p_in, p_t),axis=None)

    def caculate_dis(self, world):
        self.pos = np.zeros([self.num_agents, self.num_agents])
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                self.pos[i][j] = np.linalg.norm(world.agents[j].state.p_pos - world.agents[i].state.p_pos)
        agent_shape = [a.state.p_pos for a in world.agents]

        c_mean = np.mean(agent_shape, 0)
        print('position',c_mean)
        agent_shape = agent_shape - c_mean

        hd = max(directed_hausdorff(agent_shape, self.ideal_shape)[0],
                 directed_hausdorff(self.ideal_shape, agent_shape)[0])

        rew_forma = 20*(0.5-hd)
        rew_path = np.linalg.norm(self.target_pos) - np.linalg.norm(c_mean - self.target_pos)
        if self.l == 0:
            delta_form, delta_path, self.r_form, self.pr_last = 0., 0., 0., 0.
            self.l = 1
        else:
            delta_form = 1 * (rew_forma - self.r_form)
            delta_path = 5 * (rew_path - self.pr_last)

        self.r_form = rew_forma
        self.pr_last = rew_path
        if rew_forma<0:
            rew_path=0.5*rew_path
            delta_path=0.5*delta_path
        else:
            rew_path=2*rew_path
            delta_path=2*delta_path

        lim = 0
        for a in world.agents:
            dt = np.linalg.norm(a.state.p_pos-c_mean)
            if dt > self.dis_p:
                lim += 3 * np.power((dt - self.dis_p), 2)
        self.rew = rew_forma + delta_form - lim\
                          + 7 * rew_path + delta_path
