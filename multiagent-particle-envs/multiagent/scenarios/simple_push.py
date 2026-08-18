import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        self.reset_world(world)
        return world

    def reset_world(self, world):

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[i + 1] += 0.8
            landmark.index = i

        goal = np.random.choice(world.landmarks)
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                j = goal.index
                agent.color[j + 1] += 0.5

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):

        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):

        return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

    def adversary_reward(self, agent, world):

        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
        pos_rew = min(agent_dist)

        neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))

        return pos_rew - neg_rew

    def observation(self, agent, world):

        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)

        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] + [agent.goal_a.state.p_pos - agent.state.p_pos] + [agent.color] + entity_pos + entity_color + other_pos)
        else:

            return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
