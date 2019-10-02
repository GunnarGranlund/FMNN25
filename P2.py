import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
import scipy.optimize as op
from chebyquad_problem import *
from BaseMethods import *


class OptimizationProblem(object):
    def __init__(self, function, n):
        self.f = function
        self.h = 0.00001
        self.n = n
        self.Ginv = np.eye(n)

    def __call__(self, number):
        return

    def g(self, x):  # Calculates the gradient of the objective function f, for given
        g = np.zeros(n)  # values of x1,..,xn. The vector e represents the indexes where the
        for i in range(n):  # stepsize, h should be added depending on what partial derivative we want
            e = np.zeros(n)  # to caclulate
            e[i] = self.h
            g[i] = (self.f(x + e) - self.f(x)) / self.h
        return g

    def G(self, x):
        G = np.zeros((n, n))  # Calculates the hessian matrix of the objective function f for given
        for i in range(n):  # values of x1,...,xn. The vectors e1 and e2 represents the indexes where
            for j in range(n):  # the stepsize, h should be added depending on what partial derivative
                h1 = np.zeros(n)  # we want to calculate
                h2 = np.zeros(n)
                h1[i] = self.h
                h2[j] = self.h
                G[i, j] = (self.f(x + h1 + h2) - self.f(x + h1) - self.f(x + h2) + self.f(x)) / self.h ** 2
                if i == j:
                    if G[i, j] == 0:
                        G[i, j] = 0.00001
                if i != j:  # Symmetrizing step
                    G[i, j] = G[j, i]
        return G

    def invG(self, x):
        G = self.G(x)
        L = np.linalg.cholesky(G)
        Linv = np.linalg.inv(L)
        self.Ginv = np.dot(Linv, Linv.T)

    def posDefCheck(self, x):
        try:
            np.linalg.cholesky(self.G(x))
        except np.linalg.LinAlgError:
            print("Ej positivt definit matris, testa annan initial guess")


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def contour_plot(base_method, type, x, function):
    minimum, minix, miniy = base_method(type, x)
    X, Y = np.meshgrid(np.linspace(-0.5, 2, 1000), np.linspace(-0.5, 4, 1000))
    Z = function([X, Y])
    """plt.figure(1)
    plt.contour(X, Y, Z, [0, 0.1, 0.5, 1, 2, 3, 5, 10, 15, 20, 50, 100, 200, 300, 400,
                          500, 600, 700, 800], colors='black')
    plt.title('Rosenbrock function f(x,y) = 100(y-x^2)^2+(1-x)^2')"""
    plt.figure(2)
    plt.contour(X, Y, Z, [1, 3.831, 14.678, 56.234, 215.443, 825.404], colors='black')
    plt.plot(minix, miniy, color='k', marker='o', ls='-.')
    plt.plot(minimum[0], minimum[1], color='r', marker='o', ls='-.')
    plt.show()


if __name__ == '__main__':
    x1 = 1.0
    x2 = 1.1
    x = np.append(x1, x2)
    n = len(x)
    opt = OptimizationProblem(rosenbrock, n)
    bm = BaseMethods(opt)
    print(bm('exact', x)[0])
    print(bm('inexact', x)[0])
    print(bm('broyden', x)[0])
    print(bm('dfp', x)[0])
    #contour_plot(bm, 'exact', x, rosenbrock)
    #contour_plot(bm, 'inexact', x, rosenbrock)
    opt2 = OptimizationProblem(chebyquad, n)
    bm2 = BaseMethods(opt2)
    #print(bm2('exact', x)[0])
    #print(bm2('inexact', x)[0])
    #print(bm2('broyden', x)[0])
    #contour_plot(bm2, 'broyden', x, chebyquad)
