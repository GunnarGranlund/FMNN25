import numpy as np
from scipy.linalg import toeplitz
from scipy import linalg
import matplotlib.pyplot as plt


class RoomHeatingProblem:
    def __init__(self, dx, type=None):
        self.heating_wall = 40
        self.normal_wall = 15
        self.window_wall = 5
        self.dx = dx
        self.type = type
        if type == 'first':
            X = 1
            Y = 1
            self.grid1 = np.zeros((round(X / self.dx) + 1, round(Y / self.dx) + 1))
            self.grid1[0] = self.grid1[-1] = self.grid1[:, 0] = np.ones(len(self.grid1[0]))
            self.grid1[0][0] = self.grid1[-1][0] = 1
        if type == 'second':
            X = 2
            Y = 1
            self.grid2 = np.zeros((round(X / self.dx) + 1, round(Y / self.dx) + 1))
            self.grid2[0] = self.grid2[-1] = np.ones(len(self.grid2[0]))
            self.grid2[0:len(self.grid2[0]), 0] = self.grid2[len(self.grid2[0]) \
                                                             - 1:2 * len(self.grid2[0]), len(self.grid2[0]) - 1] = 1
            self.grid2[-1][-1] = self.grid2[0][-1] = 1
        if type == 'third':
            X = 1
            Y = 1
            self.grid3 = np.zeros((round(X / self.dx) + 1, round(Y / self.dx) + 1))
            self.grid3[0] = self.grid3[-1] = self.grid3[:, len(self.grid3[0]) - 1] = np.ones(len(self.grid3[0]))
            self.grid3[-1][-1] = self.grid3[0][-1] = 1

    def __call__(self):
        A, b = self.A_matrix()
        if self.type == 'first':
            grid = self.grid1
        elif self.type == 'second':
            grid = self.grid2
        elif self.type == 'third':
            grid = self.grid3
        return A, b, grid

    def A_matrix(self):
        if self.type == 'first':
            a = np.zeros(len(self.grid1[0]) ** 2)
            a[0] = -4
            a[1] = a[len(a) - 1] = a[len(self.grid1[0])] = a[len(self.grid1[0] * (len(self.grid1[0] - 1)))] = 1
            A = toeplitz(a)
            dX = (1 / (self.dx ** 2)) * np.eye(len(A[0]))
            k = 0
            bc = np.array([])
            b = np.zeros(len(a))
            for i in range(len(self.grid1[0])):
                for j in range(len(self.grid1[0])):
                    if (self.grid1[i][j]) == 1:
                        bc = np.append(bc, k)
                    k += 1

        if self.type == 'second':
            a = np.zeros(len(self.grid2[0]) * len(self.grid2[:, 0]))
            a[0] = -4
            a[1] = a[len(a) - 1] = a[len(self.grid2[0])] = a[len(self.grid2[0] * (len(self.grid2[0] - 1)))] = 1
            A = toeplitz(a)
            dX = (1 / (self.dx ** 2)) * np.eye(len(A[0]))
            k = 0
            bc = np.array([])
            b = np.zeros(len(a))
            for i in range(len(self.grid2[:, 0])):
                for j in range(len(self.grid2[0])):
                    if (self.grid2[i][j]) == 1:
                        bc = np.append(bc, k)
                    k += 1

        if self.type == 'third':
            a = np.zeros(len(self.grid3[0]) ** 2)
            a[0] = -4
            a[1] = a[len(a) - 1] = a[len(self.grid3[0])] = a[len(self.grid3[0] * (len(self.grid3[0] - 1)))] = 1
            A = toeplitz(a)
            dX = (1 / (self.dx ** 2)) * np.eye(len(A[0]))
            k = 0
            bc = np.array([])
            b = np.zeros(len(a))
            for i in range(len(self.grid3[0])):
                for j in range(len(self.grid3[0])):
                    if (self.grid3[i][j]) == 1:
                        bc = np.append(bc, k)
                    k += 1
        for i in range(len(a)):
            if i in bc:
                A[i] = A[i] - A[i]
                A[i][i] = self.dx ** 2

        if self.type == 'first':
            b[1:len(self.grid1[0])] = b[1 + len(self.grid1[0]) * 3:len(b)] = self.normal_wall
            for i in range(len(self.grid1[0, :])):
                b[i * len(self.grid1[0])] = self.heating_wall

        if self.type == 'second':
            for i in range(len(self.grid2[0])):
                b[i * len(self.grid2[0])] = self.normal_wall
            for i in range(len(self.grid2[0])):
                b[i * len(self.grid2[0]) + len(self.grid2[0]) ** 2 - 1] = self.normal_wall
            b[0:len(self.grid2[0])] = self.heating_wall
            b[len(b) - len(self.grid2[0]):len(b)] = self.window_wall

        if self.type == 'third':
            b[0:len(self.grid3[0])] = b[len(self.grid3[0]) * 3:len(b)] = self.normal_wall
            for i in range(len(self.grid3[0])):
                b[i * len(self.grid3[0]) + len(self.grid3[0]) - 1] = self.heating_wall
        return np.dot(dX, A), b


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


def create_temp_room(grid, u):
    idx = 0
    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            grid[i][k] = u[idx]
            idx += 1
    return grid


def plot_apart(grid1, grid2, grid3):
    egrid = np.zeros((len(grid1[0]) - 1, len(grid1[0])))
    G1 = np.append(egrid, grid1, axis=0)
    G2 = np.append(grid3, egrid, axis=0)
    G3 = np.append(G1, grid2, axis=1)
    G4 = np.append(G3, G2, axis=1)
    plt.imshow(G4)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    room1 = np.zeros((10, 10))
    room2 = np.zeros((10, 20))
    room3 = np.zeros((10, 10))
    omega = 0.8
    delta_x = 1 / 3

    normal_wal = 15
    wall_heater = 40
    wall_window = 5

    Room1 = RoomHeatingProblem(1 / 3, type='first')
    Room2 = RoomHeatingProblem(1 / 3, type='second')
    Room3 = RoomHeatingProblem(1 / 3, type='third')
    A1, b1, grid1 = Room1()
    A2, b2, grid2 = Room2()
    A3, b3, grid3 = Room3()
    u1 = linalg.solve(A1, b1)
    u2 = linalg.solve(A2, b2)
    u3 = linalg.solve(A3, b3)
    grid1 = create_temp_room(grid1, u1)

    grid2 = create_temp_room(grid2, u2)
    grid3 = create_temp_room(grid3, u3)
    plot_apart(grid1, grid2, grid3)

