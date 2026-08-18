import numpy as np

class BaseScenario(object):

    def make_world(self):
        raise NotImplementedError()

    def reset_world(self, world):
        raise NotImplementedError()
