import numpy as np
from inversion_recursive import invert_matrix

def lu_factorization(A):
  n = A.shape[0]
  
  if n == 1:
    return np.array([[1]]), A.copy()
  
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
  
  L11, U11 = lu_factorization(A11)
  
  U11_inv = invert_matrix(U11)
  
  L21 = A21 @ U11_inv
  
  L11_inv = invert_matrix(L11)
  
  U12 = L11_inv @ A12
  
  S = A22 - A21 @ U11_inv @ L11_inv @ A12
  
  L22, U22 = lu_factorization(S)
  

  L = np.block([
      [L11, np.zeros((mid, n - mid))],
      [L21, L22]
  ])
  
  U = np.block([
      [U11, U12],
      [np.zeros((n - mid, mid)), U22]
  ])
  
  return L, U


A = np.array([[4, 3, 2, 1],
              [6, 3, 4, 2],
              [2, 7, 3, 4],
              [1, 8, 6, 4]])

L, U = lu_factorization(A)
print("Macierz L:\n", L)
print("Macierz U:\n", U)
print(L @ U)