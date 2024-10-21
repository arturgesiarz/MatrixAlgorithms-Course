# Tests for the matrix multiplication

import numpy as np
import unittest


from multiply_binet import binet_matrix_multiply
from multiply_classic import classic_matrix_multiply


def generate_random_matrix(n):
    return np.random.uniform(low=1e-8, high=1.0, size=(n, n))


class TestMatrixMultiplicationBinet(unittest.TestCase):
    def test_multiplication_with_small_matrices(self):
        n = 3
        A = generate_random_matrix(n)
        B = generate_random_matrix(n)
        
        expected = classic_matrix_multiply(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_array_equal(expected, result)
        
    def test_multiplication_identity_matrix(self):
        n = 3
        A = generate_random_matrix(n)
        B = np.eye(n)
        
        expected = classic_matrix_multiply(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_allclose(expected, result,rtol=1e-5, atol=1e-8)
    
    def test_multiplication_zero_matrix(self):
        n = 3
        A = generate_random_matrix(n)
        B = np.zeros((n, n))
        
        expected = classic_matrix_multiply(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_allclose(expected, result,rtol=1e-5, atol=1e-8)
    
    def test_multiplication_large_matrices(self):
        n = 9
        A = generate_random_matrix(n)
        B = generate_random_matrix(n)
        
        expected = classic_matrix_multiply(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_allclose(expected, result,rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main()