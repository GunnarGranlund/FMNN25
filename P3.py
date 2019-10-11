from scipy import linalg
import matplotlib.pyplot as plt
from RoomHeatingProblem import *


class Solver:
    def __init__(self, room1, room2, room3):
        self.A1 = room1.A
        self.b1 = room1.b
        self.A2 = room2.A
        self.b2 = room2.b
        self.A3 = room3.A
        self.b3 = room3.b
        self.u1_r1 = room1.u1_r1
        self.u2_r1 = room2.u2_r1
        self.u2_r2 = room2.u2_r2
        self.u3_r2 = room3.u3_r2
        self.grid1 = room1.grid1
        self.grid2 = room2.grid2
        self.grid3 = room3.grid3

    def __call__(self):
        return self.iterate()

    def create_temp_room(self, grid, u):
        idx = 0
        for i in range(grid.shape[0]):
            for k in range(grid.shape[1]):
                grid[i][k] = u[idx]
                idx += 1
        return grid

    def plot_apart(self):
        u1, u2, u3 = self.iterate()
        grid1 = self.create_temp_room(self.grid1, u1)
        grid2 = self.create_temp_room(self.grid2, u2)
        grid3 = self.create_temp_room(self.grid3, u3)
        egrid = np.zeros((len(grid1[0]) - 1, len(grid1[0])))
        G1 = np.append(egrid, grid1, axis=0)
        G2 = np.append(grid3, egrid, axis=0)
        G3 = np.append(G1, grid2, axis=1)
        G4 = np.append(G3, G2, axis=1)
        plt.imshow(G4)
        plt.colorbar()
        plt.show()

    def solve(self, A, b):
        return linalg.solve(A, b)

    def problem_solve_omega2(self, u1, u3):
        for i in range(4):
            self.b2[int(self.u2_r1[i])] = u1[int(self.u1_r1[i])]
        for k in range(4):
            self.b2[int(self.u2_r2[k])] = u3[int(self.u3_r2[k])]
        u2 = self.solve(self.A2, self.b2)
        return u2

    def problem_solve_other(self, u1, u2, u3):
        for i in range(4):
            self.b1[int(self.u1_r1[i])] = u2[int(self.u2_r1[i])] - u1[int(self.u1_r1[i])]
        for k in range(4):
            self.b3[int(self.u3_r2[k])] = u2[int(self.u2_r2[i])] - u3[int(self.u3_r2[k])]
        u1 = self.solve(self.A1, self.b1)
        u3 = self.solve(self.A3, self.b3)
        return u1, u3

    def relaxation(self, old_u1, old_u2, old_u3, u1, u2, u3):
        omega = 0.8
        u1 = omega * u1 + (1 - omega) * old_u1
        u2 = omega * u2 + (1 - omega) * old_u2
        u3 = omega * u3 + (1 - omega) * old_u3
        return u1, u2, u3

    def iterate(self):
        old_u1 = self.solve(self.A1, self.b1)
        old_u2 = self.solve(self.A2, self.b2)
        old_u3 = self.solve(self.A3, self.b3)
        k = 10
        while k < 10:
            u2 = self.problem_solve_omega2(old_u1, old_u3)
            u1, u3 = self.problem_solve_other(old_u1, u2, old_u3)
            u1, u2, u3 = self.relaxation(old_u1, old_u2, old_u3, u1, u2, u3)
            k += 1
            old_u1 = u1
            old_u2 = u2
            old_u3 = u3
        return old_u1, old_u2, old_u3


if __name__ == '__main__':
    delta_x = 1 / 3

    Room1 = RoomHeatingProblem(delta_x, type='first')
    Room2 = RoomHeatingProblem(delta_x, type='second')
    Room3 = RoomHeatingProblem(delta_x, type='third')

    solve = Solver(Room1, Room2, Room3)
    solve.plot_apart()
