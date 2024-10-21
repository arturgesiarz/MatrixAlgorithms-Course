import numpy as np


def classic_matrix_multiply(A, B):
    """
    Matrix multiplication using the classic method.
    Classic method needed to compare the results obtained
    
    Args:
        A : matrix 
        B : matrix

    Returns:
        C: matrix given by the multiplication of A and B
    """

    C = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(B.shape[0]):
                C[i, j] += A[i, k] * B[k, j]
    return C
