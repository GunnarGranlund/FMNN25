import numpy as np


def second_order_diff(delta_x, u, i,  j):
    u_ij = (u[i][j+1] + u[i][j-1] - 4*u[i][j], u[i+1][j] + u[i-1][j]) / (delta_x**2)
    return u_ij


if __name__ == '__main__':
    room1 = np.zeros((10, 10))
    room2 = np.zeros((10, 20))
    room3 = np.zeros((10, 10))
    omega = 0.8
    delta_x = 1/20

    normal_wal = 15
    wall_heater = 40
    wall_window = 5

