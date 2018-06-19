import numpy as np
from abc import *


def getsize(a):
    size = 0
    for i in range(len(a)):
        size += 1
    return size


# noinspection PyCompatibility
class Model(metaclass=ABCMeta):
    # __metaclass__ = ABCMeta
    sampling_increment = 0
    t1 = 0
    t0 = 0
    result = np.empty((0, 0))
    for_print = np.empty((2, 0))

    def __init__(self, at0, at1, ah, n):
        self.my_list = ['Title', 'XLabel', 'YLabel']
        Model.t0 = at0
        Model.t1 = at1
        Model.sampling_increment = ah
        self.x0 = np.empty(n)

    def add_result(self, a, t):
        row = Model.result.shape
        rows = row[0]
        Model.result = np.resize(Model.result, (rows + 1, getsize(self.x0) + 1))
        Model.result[rows][0] = t
        for i in range(1, getsize(self.x0)+1):
            Model.result[rows][i] = a[i-1]

    def get_order(self):
        return getsize(self.x0)

    @abstractmethod
    def get_right(self, tv, t):
        pass
