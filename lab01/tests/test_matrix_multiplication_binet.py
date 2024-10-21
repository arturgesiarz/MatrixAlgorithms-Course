# Tests for the matrix multiplication

import unittest

from lab01.multiply_binet import binet_matrix_multiply
from lab01.multiply_classic import classic_matrix_multiply

def generate_matrix(n):
    m = 1 << (n - 1).bit_length()
    A = np.random.uniform(low=1e-8, high=1.0, size=(n, n))
    
    if n < m:
        A = np.pad(A, ((0, m - n), (0, m - n)), mode='constant')
    return A



class TestMatrixMultiplicationBinet(unittest.TestCase):
    
    def test_multiplication_with_small_matrices(self):
        A = generate_matrix(2)
        B = generate_matrix(2)
        
        expected = classic_matrix_multiply(A, B)
        result = binet_matrix_multiply(A, B)
        
        np.testing.assert_array_equal(expected, result)
        
    def test_multiplication_identity_matrix(self):
        pass
    
    def test_multiplication_zero_matrix(self):
        pass
    
    def test_multiplication_non_square_matrices(self):
        pass
    
    def test_multiplication_large_matrices(self):
        pass

if __name__ == '__main__':
    unittest.main()