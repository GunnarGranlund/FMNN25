#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:05:00 2019

@author: hannanilsson
"""

from python.cubicSpline import basis_function
import unittest
import numpy as np
import numpy.testing as npt
import os
import matplotlib.pyplot as plt

class TestIdentity(unittest.Testcase):
    def test_CheckSum(self):
        for k in range(len(u)):
            result = sum(basis_function(u[k], u_vec, 5, 3))
        assert sum(result) == 1, "Should be 1"
        
    def test_CheckPositive(self):
        for k in range(len(u)):
            N[k] = basis_function(u[k], u_vec, 5, 3)
        assert N[:] > 0, "Should be positive"
        
        
    #Are all basis functions N_i^3 positive inside [u_0,u_K]?
    #Does the basis functions sum up to 1?
    #Does s(u) = sum(d(j)),N(j)^3(u)) hold?
    #If we give the interval in the wrong order, what happens?
    #What happens if we start in 0?
    #If you give a cubic polynomial, you get a cubic pol. back
    #What if u is not sorted?
    #What if u is a list, not an array?
    #Does our code work in 1D aswell as in 2D?
    
if __name__ == '__main__':
    u = np.linspace(0.001, 1.0, 1000)
    u_vec = np.linspace(0, 1, 26)
    u_vec[1] = u_vec[2] = u_vec[0]
    u_vec[-3] = u_vec[-2] = u_vec[-1]
    d = np.array([(-12.73564, 9.03455),
                  (-26.77725, 15.89208),
                  (-42.12487, 20.57261),
                  (-15.34799, 4.57169),
                  (-31.72987, 6.85753),
                  (-49.14568, 6.85754),
                  (-38.09753, -1e-05),
                  (-67.92234, -11.10268),
                  (-89.47453, -33.30804),
                  (-21.44344, -22.31416),
                  (-32.16513, -53.33632),
                  (-32.16511, -93.06657),
                  (-2e-05, -39.83887),
                  (10.72167, -70.86103),
                  (32.16511, -93.06658),
                  (21.55219, -22.31397),
                  (51.377, -33.47106),
                  (89.47453, -33.47131),
                  (15.89191, 0.00025),
                  (30.9676, 1.95954),
                  (45.22709, 5.87789),
                  (14.36797, 3.91883),
                  (27.59321, 9.68786),
                  (39.67575, 17.30712)])
    N = np.zeros(len(u))
    unittest.main()