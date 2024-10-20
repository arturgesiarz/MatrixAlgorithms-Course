import numpy as np
import time
import matplotlib.pyplot as plt

def generate_matrix(n):
    return np.random.uniform(0.00000001, 1.0, (n, n)).tolist()

def add_matrix(A, B):
    global flops
    n = len(A)
    result = [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]
    flops += n * n
    return result

def subtract_matrix(A, B):
    global flops
    n = len(A)
    result = [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]
    flops += n * n
    return result

def split_matrix(matrix):
    n = len(matrix)
    mid = n // 2
    A11 = [row[:mid] for row in matrix[:mid]]
    A12 = [row[mid:] for row in matrix[:mid]]
    A21 = [row[:mid] for row in matrix[mid:]]
    A22 = [row[mid:] for row in matrix[mid:]]
    return A11, A12, A21, A22

def multiply(A, B):
    global flops
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                flops += 2
    return C


def pad_matrix_to_even(matrix):
    n = len(matrix)
    if n % 2 == 0:
        return matrix
    padded_matrix = [row + [0] for row in matrix]
    padded_matrix.append([0] * (n + 1))
    return padded_matrix

def strassen(A, B, n, cutoff=16):
    global flops
    if n <= cutoff:
        return multiply(A, B)

    A_padded = pad_matrix_to_even(A)
    B_padded = pad_matrix_to_even(B)
    padded_n = len(A_padded)

    A11, A12, A21, A22 = split_matrix(A_padded)
    B11, B12, B21, B22 = split_matrix(B_padded)

    M1 = strassen(add_matrix(A11, A22), add_matrix(B11, B22), padded_n // 2, cutoff)
    M2 = strassen(add_matrix(A21, A22), B11, padded_n // 2, cutoff)
    M3 = strassen(A11, subtract_matrix(B12, B22), padded_n // 2, cutoff)
    M4 = strassen(A22, subtract_matrix(B21, B11), padded_n // 2, cutoff)
    M5 = strassen(add_matrix(A11, A12), B22, padded_n // 2, cutoff)
    M6 = strassen(subtract_matrix(A21, A11), add_matrix(B11, B12), padded_n // 2, cutoff)
    M7 = strassen(subtract_matrix(A12, A22), add_matrix(B21, B22), padded_n // 2, cutoff)

    C11 = add_matrix(subtract_matrix(add_matrix(M1, M4), M5), M7)
    C12 = add_matrix(M3, M5)
    C21 = add_matrix(M2, M4)
    C22 = add_matrix(subtract_matrix(add_matrix(M1, M3), M2), M6)

    new_size = len(C11)
    C = [[0 for _ in range(2 * new_size)] for _ in range(2 * new_size)]
    for i in range(new_size):
        for j in range(new_size):
            C[i][j] = C11[i][j]
            C[i][j + new_size] = C12[i][j]
            C[i + new_size][j] = C21[i][j]
            C[i + new_size][j + new_size] = C22[i][j]

    return [row[:n] for row in C[:n]] 

sizes = range(1, 200)
times = []
operations = []

for size in sizes:
    A = generate_matrix(size)
    B = generate_matrix(size)
    
    flops = 0 
    start_time = time.time()
    strassen(A, B, size)
    end_time = time.time()
    
    times.append(end_time - start_time)
    operations.append(flops)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(sizes, times, label='Time (s)')
plt.xlabel('Matrix Size (n x n)')
plt.ylabel('Time (seconds)')
plt.title('Strassen Multiplication Time vs Matrix Size')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(sizes, operations, label='Operations (Flops)', color='orange')
plt.xlabel('Matrix Size (n x n)')
plt.ylabel('Number of Operations')
plt.title('Floating Point Operations vs Matrix Size')
plt.grid(True)

plt.tight_layout()
plt.show()
