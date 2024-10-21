# Recursive matrix multiplication using  Binét’s method
# O(n^3)

import numpy as np

def binet_matrix_multiply(A, B):
    """Matrix multiplication using Binét’s method

    Args:
        A : matrix
        B : matrix

    Returns:
        C : Matrix product of A and B
    """
    n = A.shape[0]
    
    A = pad_matrix(A)
    B = pad_matrix(B)
    C = binet_matrix_multiply_recursive(A, B)
    
    return C[:n, :n]

def pad_matrix(A):
    """Matrix padding with zeros to make it a square matrix of size 2^k which
    k is the smallest integer such that 2^k >= n

    Args:
        A : matrix

    Returns:
        C : Matrix padded with zeros
    """
    n = A.shape[0]
    m = 1 << (n - 1).bit_length()
    if n < m:
        A = np.pad(A, ((0, m - n), (0, m - n)), mode='constant')
    return A

def binet_matrix_multiply_recursive(A, B):
    """Recursive matrix multiplication using Binét’s method

    Args:
        A : Matrix n x n, where n is a power of 2
        B : Matrix n x n, where n is a power of 2
        
    Returns:
        C : Matrix product of A and B
    """
    n = A.shape[0]

    if n == 1:
        return A * B

    mid = n // 2

    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    C11 = binet_matrix_multiply(A11, B11) + binet_matrix_multiply(A12, B21)
    C12 = binet_matrix_multiply(A11, B12) + binet_matrix_multiply(A12, B22)
    C21 = binet_matrix_multiply(A21, B11) + binet_matrix_multiply(A22, B21)
    C22 = binet_matrix_multiply(A21, B12) + binet_matrix_multiply(A22, B22)

    C = np.vstack([np.hstack([C11, C12]), np.hstack([C21, C22])])
    return C