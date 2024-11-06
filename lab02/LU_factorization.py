import numpy as np
import matplotlib.pyplot as plt
import time
from inversion_recursive import invert
from multiplication.strassen import strassen, split_matrix

def generate_random_matrix(n):
    return np.random.uniform(0.00000001, 1.0, (n, n))
  
def LU(matrix):
  n = matrix.shape[0]
  
  def pad_matrix(A):
        n = A.shape[0]
        m = 1 << (n - 1).bit_length()
        if n < m:
            A = np.pad(A, ((0, m - n), (0, m - n)), mode='constant')
        return A
  
  def lu_factorization(A):
    n = A.shape[0]
    mid = n // 2
    
    if n == 1:
      return np.array([[1]]), A.copy(), 0
    
    flops = [0] * 10
    A11, A12, A21, A22 = split_matrix(A)
    
    L11, U11, flops[0] = lu_factorization(A11)
    U11_inv, flops[1] = invert(U11)
    L21, flops[2] = strassen(A21,U11_inv)
    L11_inv, flops[3] = invert(L11)
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
  matrix = pad_matrix(matrix)
  L, U, flops = lu_factorization(matrix)
  return L[:n, :n], U[:n, :n], flops


def generate_plot():
  matrix_sizes = []
  times = []
  flopss = []
  max_size = 30

  for size in range(1, max_size):
      matrix = np.array(generate_random_matrix(size).tolist())
      
      start_time = time.time()
      L, U, flops = LU(matrix)
      end_time = time.time()
      
      elapsed_time = end_time - start_time
      matrix_sizes.append(size)
      times.append(elapsed_time)
      flopss.append(flops)
      
      assert np.allclose(np.tril(L), L), "Macierz L nie jest dolnotrójkątna."
      assert np.allclose(np.triu(U), U), "Macierz U nie jest górnotrójkątna."
      assert np.allclose(L @ U, matrix, atol=1e-4), "Macierz L * U nie odtwarza poprawnie oryginalnej macierzy."
            
          
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  plt.plot(matrix_sizes, times, label='Czas działania (s)')
  plt.xlabel('Rozmiar macierzy')
  plt.ylabel('Czas działania (s)')
  plt.title('Czas działania LU faktoryzacji')


  plt.subplot(1, 2, 2)
  plt.plot(matrix_sizes, flopss, label='Liczba operacji')
  plt.xlabel('Rozmiar macierzy')
  plt.ylabel('Liczba operacji zmiennoprzecinkowych')
  plt.title('Liczba operacji dla LU faktoryzacji')

  plt.tight_layout()
  plt.show()
  
# generate_plot()
