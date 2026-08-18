import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        world.dim_c = 10
        world.collaborative = True

        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False

        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        self.reset_world(world)
        return world

    def reset_world(self, world):

        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None

        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])

        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        world.landmarks[1].color = np.array([0.25,0.75,0.25])
        world.landmarks[2].color = np.array([0.25,0.25,0.75])

        world.agents[0].goal_a.color = world.agents[0].goal_b.color
        world.agents[1].goal_a.color = world.agents[1].goal_b.color

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):

        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color

        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)

        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
