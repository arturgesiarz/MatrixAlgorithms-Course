# Tests for the matrix multiplication

import numpy as np
import unittest
from multiply_binet import binet_matrix_multiply


def generate_random_matrix(n):
    return np.random.uniform(low=1e-8, high=1.0, size=(n, n))


class TestMatrixMultiplicationBinet(unittest.TestCase):
    def test_multiplication_with_small_matrices(self):
        n = 4
        A = generate_random_matrix(n)
        B = generate_random_matrix(n)
        
        expected = np.dot(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_allclose(expected, result, rtol=1e-5, atol=1e-9)
        
    def test_multiplication_identity_matrix(self):
        n = 4
        A = generate_random_matrix(n)
        B = np.eye(n)
        
        expected = np.dot(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_allclose(expected, result, rtol=1e-5, atol=1e-9)
    
    def test_multiplication_zero_matrix(self):
        n = 4
        A = generate_random_matrix(n)
        B = np.zeros((n, n))
        
        expected = np.dot(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_allclose(expected, result,rtol=1e-5, atol=1e-9)


if __name__ == '__main__':
    unittest.main()