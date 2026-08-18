import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True

        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075

        world.agents[0].movable = False

        world.agents[1].silent = True

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04

        self.reset_world(world)
        return world

    def reset_world(self, world):

        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None

        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])

        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])

        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):

        return self.reward(agent, reward)

    def reward(self, agent, world):

        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):

        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)

        if not agent.movable:
            return np.concatenate([goal_color])

        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
