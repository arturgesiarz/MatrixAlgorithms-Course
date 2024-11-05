# Recursive matrix multiplication using  Binét’s method
# O(n^3)

import numpy as np
import matplotlib.pyplot as plt
import time

def binet_matrix_multiply(A, B):
    n = A.shape[0]
    flops = [0]
    
    def pad_matrix(A):
        n = A.shape[0]
        m = 1 << (n - 1).bit_length()
        if n < m:
            A = np.pad(A, ((0, m - n), (0, m - n)), mode='constant')
        return A

    def binet(A, B):
        nonlocal flops
        n = A.shape[0]

        if n == 1:
            flops[0] += 1
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

        C11 = binet(A11, B11) + binet(A12, B21)
        flops[0] += mid**2
        C12 = binet(A11, B12) + binet(A12, B22)
        flops[0] += mid**2
        C21 = binet(A21, B11) + binet(A22, B21)
        flops[0] += mid**2
        C22 = binet(A21, B12) + binet(A22, B22)
        flops[0] += mid**2

        C = np.vstack([np.hstack([C11, C12]), np.hstack([C21, C22])])
        return C
    
    
    A = pad_matrix(A)
    B = pad_matrix(B)
    C = binet(A, B)
    
    return C[:n, :n], flops


def generate_plots_binet():
    sizes = range(1, 30)
    times = []
    operations = []

    for size in sizes:
        A = generate_random_matrix(size)
        B = generate_random_matrix(size)
        
        start_time = time.time()
        result, flops = binet_matrix_multiply(A, B)
        end_time = time.time()
        
        times.append(end_time - start_time)
        operations.append(flops)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, label='Time (s)')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Time (seconds)')
    plt.title('Binet Multiplication Time vs Matrix Size')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sizes, operations, label='Operations (Flops)', color='orange')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Number of Operations')
    plt.title('Binet Floating Point Operations vs Matrix Size')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    
def generate_random_matrix(n):
    return np.random.uniform(low=1e-8, high=1.0, size=(n, n))

generate_plots_binet()