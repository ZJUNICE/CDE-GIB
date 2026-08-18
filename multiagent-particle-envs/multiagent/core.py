import numpy as np

class EntityState(object):
    def __init__(self):

        self.p_pos = None

        self.p_vel = None

class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()

        self.c = None

class Action(object):
    def __init__(self):

        self.u = None

        self.c = None

class Entity(object):
    def __init__(self):

        self.name = ''

        self.size = 0.050

        self.movable = False

        self.collide = True

        self.density = 25.0

        self.color = None

        self.max_speed = None
        self.accel = None

        self.state = EntityState()

        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()

        self.movable = True

        self.silent = False

        self.blind = False

        self.u_noise = None

        self.c_noise = None

        self.u_range = 1.0

        self.state = AgentState()

        self.action = Action()

        self.action_callback = None

class World(object):
    def __init__(self):

        self.agents = []
        self.landmarks = []

        self.dim_c = 0

        self.dim_p = 2

        self.dim_color = 3

        self.dt = 0.1

        self.damping = 0.25

        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    @property
    def entities(self):
        return self.agents + self.landmarks

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):

        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        p_force = [None] * len(self.entities)

        p_force = self.apply_action_force(p_force)

        p_force = self.apply_environment_force(p_force)

        self.integrate_state(p_force)

        for agent in self.agents:
            self.update_agent_state(agent)

    def apply_action_force(self, p_force):

        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    def apply_environment_force(self, p_force):

        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):

        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]
        if (entity_a is entity_b):
            return [None, None]

        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        dist_min = entity_a.size + entity_b.size

        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
