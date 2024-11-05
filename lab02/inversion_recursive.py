import numpy as np
import math
from multiplication.strassen import strassen, split_matrix

def generate_random_matrix(n):
    return np.random.uniform(0.00000001, 1.0, (n, n))

def invert_matrix(A):
    n = A.shape[0]
    mid = n // 2
    
    if n == 1:
        return np.array([[1 / A[0, 0]]]), 1
    
    A11, A12, A21, A22 = split_matrix(A)
    count = [0] * 16
    
    # A11_inv = inverse(A11)
    A11_inv, count[0] = invert_matrix(A11)
    
    # S22 = A22 - A21 @ A11_inv @ A12
    S22, count[1] = strassen(A21, A11_inv)
    S22, count[2] = strassen(S22, A12)
    S22, count[3] = A22 - S22, mid**2
    
    # S22_inv = inverse(S22)
    S22_inv, count[4] = invert_matrix(S22)
    
    # B11 = A11_inv + A11_inv @ A12 @ S22_inv @ A21 @ A11_inv
    B11, count[5] = strassen(A11_inv, A12)
    B11, count[6] = strassen(B11, S22_inv)
    B11, count[7] = strassen(B11, A21)
    B11, count[8] = strassen(B11, A11_inv)
    B11, count[9] = A11_inv + B11, mid
    
    # B12 = -A11_inv @ A12 @ S22_inv
    B12, count[10] = strassen(A11_inv, A12)
    B12, count[11] = strassen(B12, S22_inv)
    B12, count[12] = -1 * B12, mid**2

    # B21 = -S22_inv @ A21 @ A11_inv
    B21, count[13] = strassen(S22_inv, A21)
    B21, count[14] = strassen(B21, A11_inv)
    B21, count[15] = -1 * B21, mid**2

    # B22 = S22_inv
    B22 = S22_inv
    
    return np.vstack((np.hstack((B11, B12)), np.hstack((B21, B22)))), sum(count)


# Przykład użycia
A = np.array([[4, 7, 2, 5], [2, 6, 3, 3], [5, 8, 9, 3], [1,1,3,3]])
A_inv, flops = invert_matrix(A)
print("Macierz odwrotna:\n", flops)

# Sprawdzenie poprawności wyniku
print("Sprawdzenie: A * A_inv:\n", A @ A_inv)
