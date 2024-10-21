# Tests for the matrix multiplication

import numpy as np
import unittest
from multiply_ai import ai_matrix_multiply


def generate_random_matrix(n, m):
    return np.random.uniform(low=1e-8, high=1.0, size=(n, m))

class TestMatrixMultiplicationAI(unittest.TestCase):
    def test_multiplication_with_small_matrices(self):
        A = generate_random_matrix(4, 5)
        B = generate_random_matrix(5, 5)
        
        expected = np.dot(A, B)
        result = ai_matrix_multiply(A, B)
        
        np.testing.assert_allclose(expected, result,rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
