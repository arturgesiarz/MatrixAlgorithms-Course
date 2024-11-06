import numpy as np
import matplotlib.pyplot as plt
import time
from multiplication.strassen import strassen, split_matrix

def generate_random_matrix(n):
    return np.random.uniform(0.00000001, 1.0, (n, n))

def invert(A):
    n = A.shape[0]
    mid = n // 2
    
    if n == 1:
        A[0, 0] += 1e-17
        return np.array([[1 / A[0, 0]]]), 1
    
    A11, A12, A21, A22 = split_matrix(A)
    
    count = [0] * 16
    
    # A11_inv = inverse(A11)
    A11_inv, count[0] = invert(A11)
    
    # S22 = A22 - A21 @ A11_inv @ A12
    S22, count[1] = strassen(A21, A11_inv)
    S22, count[2] = strassen(S22, A12)
    S22, count[3] = A22 - S22, mid**2
    
    # S22_inv = inverse(S22)
    S22_inv, count[4] = invert(S22)
    
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

def invert_with_padding(A):
    n = A.shape[0]
    
    def pad_matrix(A):
            n = A.shape[0]
            m = 1 << (n - 1).bit_length()
            if n < m:
                A = np.pad(A, ((0, m - n), (0, m - n)), mode='constant')
            return A
    
    A = pad_matrix(A)
    A_inv, flops = invert(A)
    
    return A_inv[:n, :n], flops

def generate_plot():
    matrix_sizes = []
    times = []
    flopss = []
    max_size = 50

    for size in range(3, max_size):
        print(size)
        matrix = np.array(generate_random_matrix(size).tolist())
        
        start_time = time.time()
        A_inv, flops = invert_with_padding(matrix)
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        matrix_sizes.append(size)
        times.append(elapsed_time)
        flopss.append(flops)
        print(A_inv @ matrix)
          
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(matrix_sizes, times, label='Czas działania (s)')
    plt.xlabel('Rozmiar macierzy')
    plt.ylabel('Czas działania (s)')
    plt.title('Czas działania odwracania macierzy')


    plt.subplot(1, 2, 2)
    plt.plot(matrix_sizes, flopss, label='Liczba operacji')
    plt.xlabel('Rozmiar macierzy')
    plt.ylabel('Liczba operacji zmiennoprzecinkowych')
    plt.title('Liczba operacji dla odwracania macierzy')

    plt.tight_layout()
    plt.show()


generate_plot()