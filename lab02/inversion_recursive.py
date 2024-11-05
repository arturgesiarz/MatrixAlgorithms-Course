import numpy as np

def invert_matrix(A):
    n = A.shape[0]
    
    if n == 1:
        return np.array([[1 / A[0, 0]]])
    
    mid = n // 2
    
    if n % 2 != 0:
        A11 = A[:mid+1, :mid+1]
        A12 = A[:mid+1, mid+1:]
        A21 = A[mid+1:, :mid+1]
        A22 = A[mid+1:, mid+1:]
    else:
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]
    
    A11_inv = invert_matrix(A11)
    
    S22 = A22 - A21 @ A11_inv @ A12
    
    S22_inv = invert_matrix(S22)
    
    B11 = A11_inv + A11_inv @ A12 @ S22_inv @ A21 @ A11_inv
    B12 = -A11_inv @ A12 @ S22_inv
    B21 = -S22_inv @ A21 @ A11_inv
    B22 = S22_inv
    
    top = np.hstack((B11, B12))
    bottom = np.hstack((B21, B22))
    return np.vstack((top, bottom))


# # Przykład użycia
# A = np.array([[4, 7, 2, 5], [2, 6, 3, 3], [5, 8, 9, 3], [1,1,3,3]])
# A_inv = invert_matrix_recursive_general(A)
# print("Macierz odwrotna:\n", A_inv)

# # Sprawdzenie poprawności wyniku
# print("Sprawdzenie: A * A_inv:\n", A @ A_inv)
