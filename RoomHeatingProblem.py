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
            self.grid1[1:len(self.grid1[0]) - 1, len(self.grid1[0]) - 1 ] = 2
            self.u1_r1 = np.array([])
        if type == 'second':
            X = 2
            Y = 1
            self.grid2 = np.zeros((round(X / self.dx) + 1, round(Y / self.dx) + 1))
            self.grid2[0] = self.grid2[-1] = np.ones(len(self.grid2[0]))
            self.grid2[0:len(self.grid2[0]), 0] = self.grid2[len(self.grid2[0]) \
                         - 1:2 * len(self.grid2[0]), len(self.grid2[0]) - 1] = 1
            self.grid2[-1][-1] = self.grid2[0][-1] = 1
            self.grid2[len(self.grid2[0]):len(self.grid2[:,0]) - 1, 0] = 2
            self.grid2[1:len(self.grid2[0]) - 1, len(self.grid2[0]) - 1] = 3
            self.u2_r1 = np.array([])
            self.u2_r2 = np.array([])
        if type == 'third':
            X = 1
            Y = 1
            self.grid3 = np.zeros((round(X / self.dx) + 1, round(Y / self.dx) + 1))
            self.grid3[0] = self.grid3[-1] = self.grid3[:, len(self.grid3[0]) - 1] = np.ones(len(self.grid3[0]))
            self.grid3[-1][-1] = self.grid3[0][-1] = 1
            self.grid3[1:len(self.grid3[0])-1, 0] = 2
            self.u3_r2 = np.array([])

        self.b = self.create_initial_b()
        self.A = self.A_matrix(self.b)
    def __call__(self):
        
        A, b = self.A_matrix()
        if self.type == 'first':
            grid = self.grid1
        elif self.type == 'second':
            grid = self.grid2
        elif self.type == 'third':
            grid = self.grid3
        return A, b, grid

    def A_matrix(self, b):
        if self.type == 'first':
            a = np.zeros(len(self.grid1[0]) ** 2)
            a[0] = -4
            a[1] = a[-1] = a[len(self.grid1[0])] = a[-len(self.grid1[0])] = 1
            a_neu = np.zeros(len(self.grid1[0]) ** 2)
            a_neu[0] = -3
            a_neu[-len(self.grid1[0])] = a_neu[len(self.grid1[0])] = 1
            a_neu2 = a_neu.copy()
            a_neu2[1] = 1
            A_neu = toeplitz(a_neu2, a_neu)
            A = toeplitz(a)
            dX = (1 / (self.dx ** 2)) * np.eye(len(A[0]))
            k = 0
            bc = np.array([])
            for i in range(len(self.grid1[0])):
                for j in range(len(self.grid1[0])):
                    if (self.grid1[i][j]) == 1:
                        bc = np.append(bc, k)
                    elif self.grid1[i][j] == 2:
                        self.u1_r1 = np.append(self.u1_r1, k)
                    k += 1

        if self.type == 'second':
            a = np.zeros(len(self.grid2[0]) * len(self.grid2[:, 0]))
            a[0] = -4
            a[1] = a[-1] = a[len(self.grid2[0])] = a[-len(self.grid2[0])] = 1
            A = toeplitz(a)
            dX = (1 / (self.dx ** 2)) * np.eye(len(A[0]))
            k = 0
            bc = np.array([])
            for i in range(len(self.grid2[:, 0])):
                for j in range(len(self.grid2[0])):
                    if (self.grid2[i][j]) == 1:
                        bc = np.append(bc, k)
                    elif self.grid2[i][j] == 2:
                        self.u2_r1 = np.append(self.u2_r1, k)
                    elif self.grid2[i][j] == 3:
                        self.u2_r2 = np.append(self.u2_r2, k)
                    k += 1

        if self.type == 'third':
            a = np.zeros(len(self.grid3[0]) ** 2)
            a[0] = -4
            a[1] = a[-1] = a[len(self.grid3[0])] = a[-len(self.grid3[0])] = 1
            a_neu = np.zeros(len(self.grid3[0]) ** 2)
            a_neu[0] = -3
            a_neu[-len(self.grid3[0])] = a_neu[len(self.grid3[0])] = 1
            a_neu2 = a_neu.copy()
            a_neu2[1] = 1
            A_neu = toeplitz(a_neu, a_neu2)
            A = toeplitz(a)
            dX = (1 / (self.dx ** 2)) * np.eye(len(A[0]))
            k = 0
            bc = np.array([])
            for i in range(len(self.grid3[0])):
                for j in range(len(self.grid3[0])):
                    if (self.grid3[i][j]) == 1:
                        bc = np.append(bc, k)
                    if self.grid3[i][j] == 2:
                        self.u3_r2 = np.append(self.u3_r2, k)
                    k += 1
        for i in range(len(b)):
            if b[i] != 0:
                A[i] = A[i] - A[i]
                A[i][i] = self.dx ** 2
            if self.type == 'first':
                if i in self.u1_r1:
                    A[i] = A_neu[i]
            if self.type == 'third':
                if i in self.u3_r2:
                    A[i] = A_neu[i]
        return np.dot(dX, A)
                
    def create_initial_b(self):
        if self.type == 'first':
            b = np.zeros(len(self.grid1[0])**2)
            b[1:len(self.grid1[0])] = b[1 + len(self.grid1[0]) * (len(self.grid1[0]) - 1):len(b)] = self.normal_wall
            for i in range(len(self.grid1[0, :])):
                b[i * len(self.grid1[0])] = self.heating_wall

        if self.type == 'second':
            b = np.zeros(len(self.grid2[0])*len(self.grid2[:,0]))
            for i in range(len(self.grid2[0])):
                b[i * len(self.grid2[0])] = self.normal_wall
                b[i * len(self.grid2[0]) + len(self.grid2[0]) ** 2 - 1] = self.normal_wall
            b[0:len(self.grid2[0])] = self.heating_wall
            b[len(b) - len(self.grid2[0]):len(b)] = self.window_wall

        if self.type == 'third':
            b = np.zeros(len(self.grid3[0])**2)
            b[0:len(self.grid3[0])] = b[len(self.grid3[0]) * (len(self.grid3[0]) - 1):len(b)] = self.normal_wall
            for i in range(len(self.grid3[0])):
                b[i * len(self.grid3[0]) + len(self.grid3[0]) - 1] = self.heating_wall
        return b
                
        

