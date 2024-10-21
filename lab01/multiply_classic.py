
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
    n = A.shape[0]
    C = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C