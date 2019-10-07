import numpy as np
import scipy.optimize as op


def check(list1, val):
    # traverse in the list
    for x in list1:
        # compare with all the values
        # with val
        if abs(x) >= val:
            return False
    return True


class BaseMethods:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self, initial_guess, alpha_type=None, hessian_type=None):
            return self.newton(initial_guess, alpha_type, hessian_type)

    def f_alpha(self, x_prev, alpha, s_k):
        return self.optimizer.f(x_prev + alpha * s_k)

    def f_prim_alpha(self, x, alpha, s_k):
        e = self.optimizer.h
        return (self.f_alpha(x, alpha + e, s_k) - self.f_alpha(x, alpha, s_k)) / self.optimizer.h

    def extrapolation(self, alpha_zero, alpha_lower, x, s_k):
        return (alpha_zero - alpha_lower) * (self.f_prim_alpha(x, alpha_zero, s_k) /
                (self.f_prim_alpha(x, alpha_lower, s_k) - self.f_prim_alpha(x, alpha_zero, s_k)))

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

    def newton(self, x_prev, alpha_type, hessian_type):
        alpha_lower = 0.
        alpha_upper = 10 ** 99
        alpha_zero = 1.
        x_px = np.array(())
        x_py = np.array(())
        self.optimizer.invG(x_prev)
        while 1:
            s_k = - np.dot(self.optimizer.Ginv, self.optimizer.g(x_prev))
            if check(self.optimizer.g(x_prev), 0.05) :
                return x_prev, x_px, x_py
            if alpha_type == 'exact':
                alpha_zero = op.fmin(self.f_alpha, 1, (x_prev, s_k), disp=0)
            elif alpha_type == 'inexact':
                alpha_zero = self.inexact_line_search(alpha_zero, alpha_lower, alpha_upper, x_prev, s_k)
            else:
                print("No known alpha type.")
                return
            x_next = x_prev + alpha_zero * s_k
            if hessian_type == 'bfgs':
                self.optimizer.Ginv = self.bfgs(x_prev, x_next)
            if hessian_type == 'broyden':
                self.optimizer.Ginv = self.broyden(x_prev, x_next)
            if hessian_type == 'dfp':
                self.optimizer.Ginv = self.dfp(x_prev, x_next)
            else:
                self.optimizer.posDefCheck(x_prev)
                self.optimizer.invG(x_prev)
            
            x_px = np.append(x_px, x_prev[0])
            x_py = np.append(x_py, x_prev[1])
            print(x_prev)
            x_prev = x_next

    def broyden(self, x_prev, x_next):
        delta_k = x_next - x_prev
        gamma_k = self.optimizer.g(x_next) - self.optimizer.g(x_prev)
        u = delta_k - np.dot(self.optimizer.Ginv, gamma_k)
        a = 1 / (np.dot(u.T, gamma_k))
        return self.optimizer.Ginv + np.dot(a, np.dot(u, u.T))

    def dfp(self, x_prev, x_next):
        delta_k = x_next - x_prev
        gamma_k = self.optimizer.g(x_next) - self.optimizer.g(x_prev)
        return self.optimizer.Ginv + (np.dot(delta_k, delta_k.T)) / (np.dot(delta_k.T, gamma_k)) - \
                (np.dot(self.optimizer.Ginv, np.dot(gamma_k, np.dot(gamma_k.T, self.optimizer.Ginv)))) / \
                (np.dot(gamma_k.T, np.dot(self.optimizer.Ginv, gamma_k)))

    def bfgs(self, x_prev, x_next):
        delta_k = x_next - x_prev
        gamma_k = self.optimizer.g(x_next) - self.optimizer.g(x_prev)
        divide = np.dot(delta_k.T, gamma_k)
        first = np.dot(np.dot(gamma_k.T, self.optimizer.Ginv), gamma_k)/divide
        second = np.dot(delta_k, delta_k.T)/divide
        third = np.dot(np.dot(delta_k, gamma_k.T), self.optimizer.Ginv) + np.dot(np.dot(self.optimizer.Ginv, gamma_k), delta_k.T)
        return  self.optimizer.Ginv + np.dot((1 + first), second) - third/divide