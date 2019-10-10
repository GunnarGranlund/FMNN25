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

    def iterate(self):
        u1 = self.solve(self.A1, self.b1)
        u2 = self.solve(self.A2, self.b2)
        u3 = self.solve(self.A3, self.b3)
        k = 1
        while k < 10:
            u2 = problem_solve_omega2(u1, u2, u3)
            u1, u3 = problem_solve_other(u1, u2, u3)
            relaxation()
            k += 1

        


if __name__ == '__main__':
    room1 = np.zeros((10, 10))
    room2 = np.zeros((10, 20))
    room3 = np.zeros((10, 10))
    omega = 0.8
    delta_x = 1/20

    normal_wal = 15
    wall_heater = 40
    wall_window = 5

