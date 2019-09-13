import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def hot(u_vec, u):
    return (u_vec > u).argmax() - 1


def basis_function(u, u_vec, i, k):
    if k == 0:
        if u_vec[i - 1] == u_vec[i]:
            return 0
        elif u_vec[i-1] <= u < u_vec[i]:
            return 1
        else:
            return 0
    else:
        u1 = u - u_vec[i - 1]
        u2 = u_vec[i + k - 1] - u_vec[i - 1]
        u3 = u_vec[i + k] - u
        u4 = u_vec[i + k] - u_vec[i]
        if u1 == 0 and u2 == 0 and u3 == 0 and u4 == 0:
            return np.dot(0, basis_function(u, u_vec, i, k - 1)) \
                   + np.dot(0, basis_function(u, u_vec, i +1, k - 1))
        elif u1 == 0 and u2 == 0:
            return np.dot(0, basis_function(u, u_vec, i, k - 1)) \
                   + np.dot(u3 / u4, basis_function(u, u_vec, i + 1, k - 1))
        elif u3 == 0 and u4 == 0:
            return np.dot(u1 / u2, basis_function(u, u_vec, i, k - 1)) \
                   + np.dot(0, basis_function(u, u_vec, i + 1, k - 1))

        return np.dot(u1 / u2, basis_function(u, u_vec, i, k - 1)) + np.dot(u3 / u4, basis_function(u, u_vec, i+1, k - 1))


class CubicSpline:
    def __init__(self, u, u_vec, d):
        self.d = d
        self.u = u
        self.u_vec = u_vec
        self.hot_interval = hot(u, u_vec)
        self.alpha = (u_vec[-1] - u) / (u_vec[-1] - u_vec[0])

    def __call__(self, *args, **kwargs):
        print("Hot interval is ", self.hot_interval)
        print("Alpha is", self.alpha)

    def su(self, rm, lm):
        if rm - lm == 3:
            alpha = (self.u_vec[rm] - self.u) / (self.u_vec[rm] - self.u_vec[lm])
            return alpha * self.d[lm, :] + (1 - alpha) * self.d[lm + 1, :]
        elif self.u_vec[rm] - self.u_vec[lm] == 0:
            alpha = 0
            return alpha * self.su(rm, lm - 1) + (1 - alpha) * self.su(rm + 1, lm)
        else:
            alpha = (self.u_vec[rm] - self.u) / (self.u_vec[rm] - self.u_vec[lm])
            return alpha * self.su(rm, lm - 1) + (1 - alpha) * self.su(rm + 1, lm)

    def plot(self, SU):
        plt.plot(SU[0, :], SU[1, :], label='Spline')
        plt.plot(self.d[:, 0], self.d[:, 1], color='r', marker='o', ls='-.', label='Control Polygon')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # u_vec = np.array([0., 0., 0.,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 1.0, 1.0, 1.0])
    u = np.linspace(0.001, 1.0, 1000)
    u_vec = np.linspace(0, 1, 26)
    u_vec[1] = u_vec[2] = u_vec[0]
    u_vec[-3] = u_vec[-2] = u_vec[-1]
    # print(u_vec)
    # d = np.array([[1., 2., 6., 7., 9., 6., 2., 6., 2., 7., 4., 0.], [8., 3., 4., 0., 10., 8., 1., 7., 4., 0., 1., 5.]])
    d = np.array([(-12.73564, 9.03455),
                  (-26.77725, 15.89208),
                  (-42.12487, 20.57261),
                  (-15.34799, 4.57169),
                  (-31.72987, 6.85753),
                  (-49.14568, 6.85754),
                  (-38.09753, -1e-05),
                  (-67.92234, -11.10268),
                  (-89.47453, -33.30804),
                  (-21.44344, -22.31416),
                  (-32.16513, -53.33632),
                  (-32.16511, -93.06657),
                  (-2e-05, -39.83887),
                  (10.72167, -70.86103),
                  (32.16511, -93.06658),
                  (21.55219, -22.31397),
                  (51.377, -33.47106),
                  (89.47453, -33.47131),
                  (15.89191, 0.00025),
                  (30.9676, 1.95954),
                  (45.22709, 5.87789),
                  (14.36797, 3.91883),
                  (27.59321, 9.68786),
                  (39.67575, 17.30712)])
    SU = np.zeros((2, len(u)))
    N = np.zeros(len(u))
    for k in range(len(u)):
        spline = CubicSpline(u[k], u_vec, d)
        i = hot(u_vec, u[k])
        SU[:, k] = spline.su(i + 1, i)
        N[k] = basis_function(u[k], u_vec, 5, 3)

    plt.plot(u, N)
    plt.show()
    # spline.plot(SU)
