import integr
import model
import numpy as np
import math


class TDP(integr.Integrator):

    def __init__(self):
        self.k = np.empty((1, 4))
        self.h = 0.00001
        super().__init__()
        self.a = np.array([[0, 0, 0, 0, 0, 0],
                           [1 / 5, 0, 0, 0, 0, 0],
                           [3 / 40, 9 / 40, 0, 0, 0, 0],
                           [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
                           [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
                           [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
                           [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]])
        self.b1 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
        self.b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
        self.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
        self.geps = 0

    def run(self, tm: model.Model):
        self.h = tm.sampling_increment
        tk = tm.t1
        t = tm.t0
        y0 = tm.x0
        tn = 0
        y4 = np.empty(tm.get_order())
        y5 = np.empty(tm.get_order())
        y6 = np.empty(tm.get_order())
        bet = np.empty(6)
        while t <= tk:
            self.k = np.resize(self.k, (1, tm.get_order()))
            hs = self.h
            temp = np.empty(tm.get_order())
            for i in range(len(y0)):
                temp[i] = y0[i]

            self.k = np.row_stack((self.k, tm.get_right(temp, t)))
            self.k = np.delete(self.k, 0, 0)
            for i in range(tm.get_order()):
                temp[i] = y0[i] + self.h * self.a[1, 0] * self.k[0][i]

            self.k = np.row_stack((self.k, tm.get_right(temp, t)))
            for i in range(tm.get_order()):
                temp[i] = y0[i] + self.h * (self.a[2, 0] * self.k[0][i] + self.a[2, 1] * self.k[1][i])

            self.k = np.row_stack((self.k, tm.get_right(temp, t)))
            for i in range(tm.get_order()):
                temp[i] = y0[i] + self.h * (
                    self.a[3, 0] * self.k[0][i] + self.a[3, 1] * self.k[1][i] + self.a[3, 2] * self.k[2, i])

            self.k = np.row_stack((self.k, tm.get_right(temp, t)))
            for i in range(tm.get_order()):
                temp[i] = y0[i] + self.h * (
                    self.a[4, 0] * self.k[0][i] + self.a[4, 1] * self.k[1][i] + self.a[4, 2] * self.k[2][i] + self.a[
                        4, 3] * self.k[3][i])

            self.k = np.row_stack((self.k, tm.get_right(temp, t)))
            for i in range(tm.get_order()):
                temp[i] = y0[i] + self.h * (
                    self.a[5, 0] * self.k[0][i] + self.a[5, 1] * self.k[1][i] + self.a[5, 2] * self.k[2][i] + self.a[
                        5, 3] * self.k[3][i] + self.a[5, 4] * self.k[4][i])

            self.k = np.row_stack((self.k, tm.get_right(temp, t)))
            for i in range(tm.get_order()):
                temp[i] = y0[i] + self.h * (
                    self.a[6, 0] * self.k[0][i] + self.a[6, 1] * self.k[1][i] + self.a[6, 2] * self.k[2][i] + self.a[
                        6, 3] * self.k[3][i] + self.a[6, 4] * self.k[4][i] +
                    self.a[6, 5] * self.k[5][i])

            self.k = np.row_stack((self.k, tm.get_right(temp, t)))
            for i in range(tm.get_order()):
                y4[i] = y0[i] + self.h * (self.b[0] * self.k[0][i] +
                                          self.b[1] * self.k[1][i] + self.b[2] * self.k[2][i] + self.b[3] * self.k[3][
                                              i] +
                                          self.b[4] * self.k[4][i] + self.b[5] * self.k[5][i] + self.b[6] * self.k[6][
                                              i])
                y5[i] = y0[i] + self.h * (self.b1[0] * self.k[0][i] +
                                          self.b1[1] * self.k[1][i] + self.b1[2] * self.k[2][i] + self.b1[3] *
                                          self.k[3][i] +
                                          self.b1[4] * self.k[4][i] + self.b1[5] * self.k[5][i] + self.b1[6] *
                                          self.k[6][i])
            at = model.getsize(y0)
            sum = 0
            for i in range(at - 1):
                nevepi = math.pow((self.h * (y4[i] - y5[i]) / max(max(1e-5, abs(y4[i])),
                                                                  max(abs(y0[i]), 2 * self.u / self.geps))), 2)
                sum += nevepi
            eps = math.sqrt(sum / at)
            self.h = self.h / max(0.1, min(5.0, (math.pow((eps / self.geps), 0.2) / 0.9)))
            if self.geps < eps:
                continue
            while (tn <= t + hs) and (tn <= tm.t1):
                teta = (tn - t) / hs
                bet[0] = teta * (1 + teta * (-1337 / 480 + teta * (1039 / 360 + teta * (-1163 / 1152))))
                bet[1] = 0
                bet[2] = 100 * math.pow(teta, 2) * (1054 / 9275 + teta * (-4682 / 27825 + teta * (379 / 5565))) / 3
                bet[3] = -5 * math.pow(teta, 2) * (27 / 40 + teta * (-9 / 5 + teta * (83 / 96))) / 2
                bet[4] = 18225 * math.pow(teta, 2) * (-3 / 250 + teta * (22 / 375 + teta * (-37 / 600))) / 848
                bet[5] = -22 * math.pow(teta, 2) * (-3 / 10 + teta * (29 / 30 + teta * (-17 / 24))) / 7
                for i in range(tm.get_order()):
                    y6[i] = y0[i]
                    for j in range(5):
                        y6[i] = y6[i] + hs * bet[j] * self.k[j][i]
                tm.add_result(y6, tn)
                tn = tn + tm.sampling_increment
            t = t + hs
            y0 = np.empty(len(y4))
            for i in range(len(y4)):
                y0[i] = y4[i]
