from scipy import linalg
import matplotlib.pyplot as plt
from RoomHeatingProblem import *
import numpy as np
from mpi4py import MPI


def solve(A, b):
    return linalg.solve(A, b), linalg.solve(A, b)


def relax(u, old_u):
    omega = 0.8
    return omega * u + (1 - omega) * old_u


def solve_u2(u1, u3):
    for i in range(len(Room1.u1_r1)):
        Room2.b[int(Room2.u2_r1[i])] = u1[int(Room1.u1_r1[i])]
        Room2.b[int(Room2.u2_r2[i])] = u3[int(Room3.u3_r2[i])]
    Room2.A = Room2.A_matrix(Room2.b)
    return linalg.solve(Room2.A, Room2.b)


def solve_u1(u1, u2):
    for i in range(len(Room1.u1_r1)):
        Room1.b[int(Room1.u1_r1[i])] = -(u2[int(Room2.u2_r1[i]) + 1] - u1[int(Room1.u1_r1[i])])/(Room1.dx**2)
    return linalg.solve(Room1.A, Room1.b)


def solve_u3(u3, u2):
    for i in range(len(Room3.u3_r2)):
        Room3.b[int(Room3.u3_r2[i])] = -(u2[int(Room2.u2_r2[i]) - 1] - u3[int(Room3.u3_r2[i])]) / (Room3.dx ** 2)
    return linalg.solve(Room3.A, Room3.b)


def create_temp_room(grid, u):
    new_grid = np.zeros((len(grid[:, 0]) - 1, len(grid[0]) - 1))
    idx = 0
    for i in range(grid.shape[0] - 1):
        for k in range(grid.shape[1] - 1):
            new_grid[i][k] = (u[i + idx] + u[idx + i + 1] + u[idx + i + len(grid[0])] + u[idx + i + 1 + len(grid[0])])/4
            grid[i][k] = u[idx]
            idx += 1
    return new_grid


def plot_apart(u1, u2, u3):
    grid1 = create_temp_room(Room1.grid1, u1)
    grid2 = create_temp_room(Room2.grid2, u2)
    grid3 = create_temp_room(Room3.grid3, u3)
    egrid = np.zeros((len(grid1[0]), len(grid1[0])))
    G1 = np.append(egrid, grid1[:, 0:len(grid1[0])], axis=0)
    G2 = np.append(grid3[:, 0:len(grid3[0])], egrid, axis=0)
    G3 = np.append(G1, grid2, axis=1)
    G4 = np.append(G3, G2, axis=1)
    plt.imshow(G4)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    delta_x = 1 / 3

    k = 0

    while k < 11:
        if rank == 0:
            if not k:
                Room1 = RoomHeatingProblem(delta_x, type='first')
                Room2 = RoomHeatingProblem(delta_x, type='second')
                u1, old_u1 = solve(Room1.A, Room1.b)
                u2, old_u2 = solve(Room2.A, Room2.b)
                u3 = comm.recv(source=1, tag=3)
                Room3 = comm.recv(source=1, tag=333)
            else:
                u2 = solve_u2(u1, u3)
                u2 = relax(u2, old_u2)
                comm.send(u2, dest=1, tag=2)
                comm.send(Room2, dest=1, tag=22)
                u3 = comm.recv(source=1, tag=33)
                u1 = solve_u1(u1, u2)
                u1 = relax(u1, old_u1)
                old_u1 = u1
                old_u2 = u2
            if k == 10:
                comm.send(u1, dest=1, tag=1)
                comm.send(Room1, dest=1, tag=11)

        if rank == 1:
            if not k:
                Room3 = RoomHeatingProblem(delta_x, type='third')
                u3, old_u3 = solve(Room3.A, Room3.b)
                comm.send(u3, dest=0, tag=3)
                comm.send(Room3, dest=0, tag=333)
            else:
                u2 = comm.recv(source=0, tag=2)
                Room2 = comm.recv(source=0, tag=22)
                u3 = solve_u3(u3, u2)
                u3 = relax(u3, old_u3)
                comm.send(u3, dest=0, tag=33)
                old_u3 = u3
            if k == 10:
                u1 = comm.recv(source=0, tag=1)
                Room1 = comm.recv(source=0, tag=11)
                plot_apart(u1, u2, u3)
        k += 1



