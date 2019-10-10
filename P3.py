import numpy as np
from scipy import linalg


class Solver:
    def __init__(self, heating_problem):
        self.A1 = heating_problem.A1
        self.b1 = heating_problem.b1
        self.A2 = heating_problem.A2
        self.b2 = heating_problem.b2
        self.A3 = heating_problem.A3
        self.b3 = heating_problem.b3

    def solve(self, A, b):
        return linalg.solve(A, b)

    def problem_solve_omega2(self, u1, u2, u3):
        return u2

    def problem_solve_other(self, u1, u2, u3):
        return u1, u2

    def relaxation(self, old_u1, old_u2, old_u3, u1, u2, u3):
        u1 = omega*u1 + (1-omega)*old_u1
        u2 = omega*u2 + (1-omega)*old_u2
        u3 = omega*u3 + (1-omega)*old_u3
        return u1, u2, u3

    def iterate(self):
        old_u1 = self.solve(self.A1, self.b1)
        old_u2 = self.solve(self.A2, self.b2)
        old_u3 = self.solve(self.A3, self.b3)
        k = 1
        while k < 10:
            u2 = self.problem_solve_omega2(old_u1, old_u2, old_u3)
            u1, u3 = self.problem_solve_other(old_u1, u2, old_u3)
            u1, u2, u3 = self.relaxation(old_u1, old_u2, old_u3, u1, u2, u3)
            k += 1
            old_u1 = u1
            old_u2 = u2
            old_u3 = u3


if __name__ == '__main__':
    room1 = np.zeros((10, 10))
    room2 = np.zeros((10, 20))
    room3 = np.zeros((10, 10))
    omega = 0.8
    delta_x = 1/20

    normal_wal = 15
    wall_heater = 40
    wall_window = 5

