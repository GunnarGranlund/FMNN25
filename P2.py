import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
import scipy.optimize as op


# ===============================================================================
# The optimization problem class. The consctructor takes an objective function f,
# a grid x, the dimension n, and the opportunity to calculate the gradient as inputs
# ===============================================================================


def check(list1, val):
    # traverse in the list
    for x in list1:
        # compare with all the values
        # with val
        if abs(x) > val:
            return False
    return True


def compare(bigger_list, smaller_list):
    print("Bigger list: ", bigger_list)
    print("Smaller list: ", smaller_list)
    if len(bigger_list) != len(smaller_list):
        print("Not comppareable lists!")
        return False
    for i in range(len(bigger_list)):
        if bigger_list[i] < smaller_list[i]:
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
                    # print(G[i, j])
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

    def __call__(self, type, initial_guess=None):
        if type == 'newton':
            return self.newton(initial_guess)
        elif type == 'inexact':
            return self.newton_inexact(initial_guess)
        else:
            print("Can't find given type.")

    def newton(self, x_prev):
        xPx = np.array(())
        xPy = np.array(())
        while 1:
            if check(opt.g(x_prev), 0.05):  # Check is in wrong way????
                return x_prev, xPx, xPy
            opt.posDefCheck(x_prev)
            opt.invG(x_prev)
            s_k = -np.dot(opt.Ginv, opt.g(x_prev))
            alpha_k = op.fmin(self.f_alpha, 1, (x_prev, s_k), disp=0)
            x_next = x_prev + alpha_k * s_k
            xPx = np.append(xPx, x_prev[0])
            xPy = np.append(xPy, x_prev[1])
            x_prev = x_next

    def exact_line_search(self):
        pass

    def f_alpha(self, x_prev, alpha, s_k):
        return f(x_prev + alpha * s_k)

    def f_prim_alpha(self, x, alpha, s_k):
        return 200 * (x[1] + alpha * s_k[1] - (x[0] ** 2 + 2 * x[0] * alpha * s_k[0] + (alpha * s_k[0]) ** 2)) * \
               (s_k[1] - 2 * (x[0] * s_k[0] + alpha * (s_k[0] ** 2))) - 2 * (
                           s_k[0] + x[0] * s_k[0] + alpha * (s_k[1] ** 2))

    def extrapolation(self, alpha_zero, alpha_lower, x, s_k):
        return (alpha_zero - alpha_lower) * (self.f_prim_alpha(x, alpha_zero, s_k) /
                                             (self.f_prim_alpha(x, alpha_lower, s_k) - self.f_prim_alpha(x, alpha_zero,
                                                                                                         s_k)))

    def interpolation(self, alpha_zero, alpha_lower, x, s_k):
        return (alpha_zero - alpha_lower) ** 2 * self.f_prim_alpha(x, alpha_lower, s_k) / \
               (2 * (self.f_alpha(x, alpha_lower, s_k) - self.f_alpha(x, alpha_zero, s_k)) +
                (alpha_zero - alpha_lower) * self.f_alpha(x, alpha_lower, s_k))

    def left_con(self, alpha_zero, alpha_lower, x_prev, s_k):
        sigma = 0.7
        return self.f_prim_alpha(x_prev, alpha_zero, s_k) >= sigma * self.f_prim_alpha(x_prev, alpha_lower, s_k)

    def right_con(self, alpha_zero, alpha_lower, x_prev, s_k):
        rho = 0.1
        return self.f_alpha(x_prev, alpha_lower, s_k) + rho * (alpha_zero - alpha_lower) * \
               self.f_prim_alpha(x_prev, alpha_lower, s_k) >= self.f_alpha(x_prev, alpha_zero, s_k)

    def inexact_line_search(self, alpha_zero, alpha_lower, alpha_upper, x_prev, s_k):
        tau = 0.1
        xi = 9
        while not self.left_con(alpha_zero, alpha_lower, x_prev, s_k) and \
                self.right_con(alpha_zero, alpha_lower, x_prev, s_k):
            if not self.left_con(alpha_zero, alpha_lower, x_prev, s_k):
                delta_a = self.extrapolation(alpha_zero, alpha_lower, x_prev, s_k)
                delta_a = max(delta_a, tau * (alpha_zero - alpha_lower))
                delta_a = min(delta_a, xi * (alpha_zero - alpha_lower))
                alpha_lower = alpha_zero
                alpha_zero = alpha_zero + delta_a
            else:
                alpha_upper = min(alpha_zero, alpha_upper)
                alpha_bar = self.interpolation(alpha_zero, alpha_lower, x_prev, s_k)
                alpha_bar = max(alpha_bar, alpha_lower + tau * (alpha_upper - alpha_lower))
                alpha_bar = min(alpha_bar, alpha_upper - tau * (alpha_upper - alpha_lower))
                alpha_zero = alpha_bar
        return alpha_zero

    def newton_inexact(self, x_prev):
        alpha_lower = 0.
        alpha_upper = 10 ** 99
        alpha_zero = 1.  # ???????????????????????????
        while 1:
            if check(opt.g(x_prev), 0.05):
                return x_prev
            opt.posDefCheck(x_prev)
            opt.invG(x_prev)
            s_k = -np.dot(opt.Ginv, opt.g(x_prev))
            alpha_zero = self.inexact_line_search(alpha_zero, alpha_lower, alpha_upper, x_prev, s_k)
            x_next = x_prev + alpha_zero * s_k
            x_prev = x_next


def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def contour_plot(bm):
    minimum, minix, miniy = bm('newton', x)
    X, Y = np.meshgrid(np.linspace(-0.5, 2, 1000), np.linspace(-0.5, 4, 1000))
    Z = f([X, Y])
    plt.figure(1)
    plt.contour(X, Y, Z, [0, 0.1, 0.5, 1, 2, 3, 5, 10, 15, 20, 50, 100, 200, 300, 400,
                          500, 600, 700, 800], colors='black')
    plt.title('Rosenbrock function f(x,y) = 100(y-x^2)^2+(1-x)^2')
    plt.figure(2)
    plt.contour(X, Y, Z, [1, 3.831, 14.678, 56.234, 215.443, 825.404], colors='black')
    print(minix, miniy)
    print(minimum)
    plt.plot(minix, miniy, color='k', marker='o', ls='-.')
    plt.plot(minimum[0], minimum[1], color='r', marker='o', ls='-.')
    plt.show()


if __name__ == '__main__':
    x1 = 1.1
    x2 = 1.0
    x = np.append(x1, x2)
    n = len(x)
    opt = OptimizationProblem(f, n)
    bm = BaseMethods(opt)
    print(bm('newton', x)[0])
    #print(bm('inexact', x))
    contour_plot(bm)
