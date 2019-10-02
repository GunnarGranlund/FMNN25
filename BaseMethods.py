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

    def __call__(self, type, initial_guess):
        if type == 'broyden':
            return self.broyden(initial_guess)
        if type == 'dfp':
            return self.dfp(initial_guess)
        if type == 'bfgs':
            return self.bfgs(initial_guess)
        else:
            return self.newton(type, initial_guess)

    def f_alpha(self, x_prev, alpha, s_k):
        return self.optimizer.f(x_prev + alpha * s_k)

    def f_prim_alpha(self, x, alpha, s_k):
        e = self.optimizer.h
        return (self.f_alpha(x, alpha + e, s_k) - self.f_alpha(x, alpha, s_k)) / self.optimizer.h

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

    def newton(self, type, x_prev):
        alpha_lower = 0.
        alpha_upper = 10 ** 99
        alpha_zero = 1.
        x_px = np.array(())
        x_py = np.array(())
        while 1:
            if check(self.optimizer.g(x_prev), 0.05):
                return x_prev, x_px, x_py
            self.optimizer.posDefCheck(x_prev)
            self.optimizer.invG(x_prev)
            s_k = -np.dot(self.optimizer.Ginv, self.optimizer.g(x_prev))
            if type == 'exact':
                alpha_zero = op.fmin(self.f_alpha, 1, (x_prev, s_k), disp=False)
            elif type == 'inexact':
                alpha_zero = self.inexact_line_search(alpha_zero, alpha_lower, alpha_upper, x_prev, s_k)
            else:
                print("No known type.")
                return
            x_next = x_prev + alpha_zero * s_k
            x_px = np.append(x_px, x_prev[0])
            x_py = np.append(x_py, x_prev[1])
            x_prev = x_next

    def broyden(self, x_k):
        self.optimizer.invG(x_k)
        H_k_minus1 = self.optimizer.Ginv
        x_px = np.array(())
        x_py = np.array(())
        x_k_minus1 = x_k -1.
        while 1:
            if check(self.optimizer.g(x_k), 0.05):
                return x_k, x_px, x_py
            delta_k = x_k - x_k_minus1
            gamma_k = self.optimizer.g(x_k) - self.optimizer.g(x_k_minus1)
            u = delta_k - np.dot(H_k_minus1, gamma_k)
            a = 1 / (np.dot(u.T, gamma_k))
            H_k = H_k_minus1 + np.dot(a, np.dot(u, u.T))
            s_k = - np.dot(H_k, self.optimizer.g(x_k))
            x_next = x_k + s_k
            H_k_minus1 = H_k
            x_px = np.append(x_px, x_next[0])
            x_py = np.append(x_py, x_next[1])
            x_k_minus1 = x_k
            x_k = x_next

    def dfp(self, x_k):
        self.optimizer.invG(x_k)
        H_k = self.optimizer.Ginv
        x_px = np.array(())
        x_py = np.array(())
        x_k_minus1 = x_k -1.
        while 1:
            if check(self.optimizer.g(x_k), 0.05):
                return x_k, x_px, x_py
            s_k = - np.dot(H_k, self.optimizer.g(x_k))
            x_next = x_k + s_k
            delta_k = x_k - x_k_minus1
            gamma_k = self.optimizer.g(x_k) - self.optimizer.g(x_k_minus1)
            H_next = H_k + (np.dot(delta_k, delta_k.T)) / (np.dot(delta_k.T, gamma_k)) - \
                  (np.dot(H_k, np.dot(gamma_k, np.dot(gamma_k.T, H_k)))) / \
                  (np.dot(gamma_k.T, np.dot(H_k, gamma_k)))
            H_k = H_next
            x_px = np.append(x_px, x_next[0])
            x_py = np.append(x_py, x_next[1])
            x_k_minus1 = x_k
            x_k = x_next

    def bfgs(self, x_prev):
        self.optimizer.invG(x_prev)
        H_prev = self.optimizer.Ginv
        x_px = np.array(())
        x_py = np.array(())
        while 1:
            if check(self.optimizer.g(x_prev), 0.05):
                return x_prev, x_px, x_py
            s_k = np.dot(H_prev, self.optimizer.g(x_prev))
            x_next = x_prev - s_k
            delta_k = x_next - x_prev
            gamma_k = self.optimizer.g(x_next) - self.optimizer.g(x_prev)
            H_k = H_prev + (1 + (np.dot(gamma_k.T, np.dot(H_prev, gamma_k)) / np.dot(delta_k.T, gamma_k))) * \
                  (np.dot(delta_k, delta_k.T)) / np.dot(delta_k.T, gamma_k) - \
                  (np.dot(delta_k, np.dot(gamma_k.T, H_prev)) + np.dot(H_prev, np.dot(gamma_k, delta_k.T))) / \
                  (np.dot(delta_k.T, gamma_k))
            H_prev = H_k
            x_px = np.append(x_px, x_prev[0])
            x_py = np.append(x_py, x_prev[1])
            x_prev = x_next
