import numpy as np
from inversion_recursive import invert_matrix
from multiplication.strassen import strassen, split_matrix

def lu_factorization(A):
  n = A.shape[0]
  mid = n // 2
  
  if n == 1:
    return np.array([[1]]), A.copy(), 0
  
  flops = [0] * 10
  A11, A12, A21, A22 = split_matrix(A)
  
  L11, U11, flops[0] = lu_factorization(A11)
  U11_inv, flops[1] = invert_matrix(U11)
  L21, flops[2] = strassen(A21,U11_inv)
  L11_inv, flops[3] = invert_matrix(L11)
  
  U12, flops[4] = strassen(L11_inv, A12)
  
  # S = A22 - A21 @ U11_inv @ L11_inv @ A12
  S, flops[5] = strassen(A21, U11_inv)
  S, flops[6] = strassen(S, L11_inv)
  S, flops[7] = strassen(S, A12)
  S, flops[8] = A22 - S, mid**2
  
  L22, U22, flops[9] = lu_factorization(S)
  

  L = np.block([
      [L11, np.zeros((mid, n - mid))],
      [L21, L22]
  ])
  
  U = np.block([
      [U11, U12],
      [np.zeros((n - mid, mid)), U22]
  ])
  
  return L, U, sum(flops)


A = np.array([[4, 3, 2, 1],
              [6, 3, 4, 2],
              [2, 7, 3, 4],
              [1, 8, 6, 4]])

L, U, flops = lu_factorization(A)
print("Macierz L:\n", L)
print("Macierz U:\n", U)
# print(L @ U)
print(flops)