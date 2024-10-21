# Recursive matrix multiplication using  Binét’s method

import numpy as np

def generate_matrix(n):
    m = 1 << (n - 1).bit_length()
    A = np.random.uniform(low=1e-8, high=1.0, size=(n, n))
    
    if n < m:
        A = np.pad(A, ((0, m - n), (0, m - n)), mode='constant')
    return A
  
def binet_matrix_multiply(A, B):
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


n = 5
A = generate_matrix(5)
B = generate_matrix(5)
C = binet_matrix_multiply(A, B)

C = C[:n, :n]

print(C)  