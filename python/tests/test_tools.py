#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:22:35 2022

@author: nvilla
"""


import unittest

import numpy as np
from random import randint

import mpc_interface.tools as use

# the Agent helps us run tests
from testAgent import *

class ToolsTestCase(unittest.TestCase):
    # def test_extend_matrices(self):

    #     n = randint(1, 15)  # number of states
    #     m = randint(1, 15)  # number of inputs
    #     N = randint(1, 100)  # horizon length

    #     A = np.eye(n)
    #     B = np.ones([n, m])

    #     S, U = use.extend_matrices(N, A, B)

    #     self.assertTrue(S.shape == (N, n, n))
    #     self.assertIsInstance(U, list)
    #     self.assertEqual(len(U), m)
    #     self.assertTrue(U[0].shape == (N, N, n))

    def test_extend_matrices_body(self):
        A = np.array([[1., 0.1, 0.005], [0., 1., 0.1], [0., 0., 1.]])
        a0, a1 = A.shape
        B = np.array([[0.00016667], [0.005], [0.1]])
        b0, b1 = B.shape

        write_output("output/test_extend_matrices_body_A.op", get_output_2d(A))
        write_output("output/test_extend_matrices_body_B.op", get_output_2d(B))

        N = 9  # horizon length
        S, U = use.extend_matrices(N, A, B)

        write_output("output/test_extend_matrices_body_A_after_extend_matrices.op", get_output_2d(A))
        write_output("output/test_extend_matrices_body_B_after_extend_matrices.op", get_output_2d(B))

        self.assertTrue(S.shape == (N, a0, a1))
        self.assertIsInstance(U, list)
        self.assertEqual(len(U), b1)
        self.assertTrue(U[0].shape == (N, N, b0))

        write_output("output/test_extend_matrices_body_S.op", get_output_3d(S))
        write_output("output/test_extend_matrices_body_U.op", get_output_4d(U))

if __name__ == "__main__":
    unittest.main()
