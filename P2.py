import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
# ===============================================================================
# The optimization problem class. The consctructor takes an objective function f,
# a grid x, the dimension n, and the opportunity to calculate the gradient as inputs
# ===============================================================================


def check(list1, val):
    # traverse in the list
    for x in list1:
        # compare with all the values
        # with val
        if x < val:
            return False
    return True


class OptimizationProblem(object):
    def __init__(self, f, n):
        self.f = f
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
            g[i] = (f(x + e) - f(x)) / self.h
        return g

    def G(self, x):
        G = np.zeros((n, n))  # Calculates the hessian matrix of the objective function f for given
        for i in range(n):  # values of x1,...,xn. The vectors e1 and e2 represents the indexes where
            for j in range(n):  # the stepsize, h should be added depending on what partial derivative
                h1 = np.zeros(n)  # we want to calculate
                h2 = np.zeros(n)
                h1[i] = self.h
                h2[j] = self.h
                G[i, j] = (f(x + h1 + h2) - f(x + h1) - f(x + h2) + f(x)) / self.h ** 2
                if i != j:  # Symmetrizing step
                    #print(G[i, j])
                    if G[i, j] == 0:
                        G[i, j] = 0.00001
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


class BaseMethods:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, initial_guess, alpha, type):
        if type == 'newton':
            return self.newton(initial_guess, alpha)
        else:
            print("Can't find given type.")

    def newton(self, x_prev, alpha):
        while 1:
            print(x_prev)
            
            if check(opt.g(x_prev), 10**(-5)):
                return x_prev
            print(opt.Ginv)
            opt.posDefCheck(x_prev)
            opt.invG(x_prev)
            sK = -np.dot(opt.Ginv, opt.g(x_prev))
            alphaK = opt.f(x_prev +alpha*sK)
            x_next = x_prev + alphaK * sK
            x_prev = x_next
            

    def exact_line_search(self):
        pass

    def inexact_line_search(self):
        pass

# Testsaker

def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


if __name__ == '__main__':

    x1 = 1.1
    x2 = 1.0
    x = np.append(x1, x2)
    n = len(x)
    opt = OptimizationProblem(f, n)
    opt(4)
    #print(opt.g(x), '\n',  opt.G(x))
    bm = BaseMethods(opt)
    print(bm(x, 0.7, 'newton'))

    



