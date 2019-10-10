import numpy as np
from scipy.linalg import toeplitz
from scipy import linalg


class RoomHeatingProblem:
    def __init__(self, dx, type=None):
        self.temp1 = 40
        self.temp2 = 15
        self.temp3 = 5
        self.dx = dx
        self.type = type
        if type == 'first':
            X = 1
            Y = 1
            self.grid1 = np.zeros((round(X / self.dx) + 1, round(Y / self.dx) + 1))
            self.grid1[0] = self.grid1[-1] = self.grid1[:, 0] = np.ones(len(self.grid1[0]))
            self.grid1[0][0] = self.grid1[-1][0] = 1

    def A_matrix(self):
        if self.type == 'first':
            a = np.zeros(len(self.grid1[0]) ** 2)
            a[0] = -4
            a[1] = a[len(a) - 1] = a[len(self.grid1[0])] = a[len(self.grid1[0] * (len(self.grid1[0] - 1)))] = 1
            A1 = toeplitz(a)
            dX = (1 / (self.dx ** 2)) * np.eye(len(A1[0]))
            k = 0
            bc = np.array([])
            for i in range(len(self.grid1[0])):
                for j in range(len(self.grid1[0])):
                    if (self.grid1[i][j]) == 0:
                        bc = np.append(bc, k)
                    k += 1
        for i in range(len(a)):
            if i not in bc:
                A1[i] = A1[i] - A1[i]
                A1[i][i] = 1
        return A1


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
        u1 = omega * u1 + (1 - omega) * old_u1
        u2 = omega * u2 + (1 - omega) * old_u2
        u3 = omega * u3 + (1 - omega) * old_u3
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
    delta_x = 1 / 20

    normal_wal = 15
    wall_heater = 40
    wall_window = 5

    Room1 = RoomHeatingProblem(1 / 3, type='first')
    print(Room1.A_matrix())
