from abc import *
import model


class Integrator(metaclass=ABCMeta):

    def __init__(self):
        self.eps = 0
        self.geps = 0
        v = 1
        while 1 + v > 1:
            self.u = v
            v = v / 2

    @abstractmethod
    def run(self, tm):
        pass
