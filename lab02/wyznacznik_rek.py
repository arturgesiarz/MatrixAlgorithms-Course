import numpy as np
import time
import matplotlib.pyplot as plt


flops = 0

def generate_random_matrix(n):
    return np.random.uniform(0.00000001, 1.0, (n, n))

def det_rec(matrix):
    global flops
    
    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        flops += 3
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(len(matrix)):
        submatrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = (-1) ** col * matrix[0][col]
        
        flops += 1
        
        det += cofactor * det_rec(submatrix)
        
        flops += 1
    
    return det

matrix_sizes = []
times = []
flopss = []
max_size = 10

for size in range(1, max_size + 1):
    matrix = generate_random_matrix(size)
    flops = 0
    
    start_time = time.time()
    det = det_rec(matrix.tolist())
    print("rozmiar:", size, "det:", round(det, 5), "sprawdzenie:", round(np.linalg.det(matrix.tolist()), 5))
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    matrix_sizes.append(size)
    times.append(elapsed_time)
    flopss.append(flops)
        
       
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(matrix_sizes, times, label='Czas działania (s)')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas działania (s)')
plt.title('Czas działania rekurencyjnego liczenia wyznacznika')



plt.subplot(1, 2, 2)
plt.plot(matrix_sizes, flopss, label='Liczba operacji')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Liczba operacji zmiennoprzecinkowych')
plt.title('Liczba operacji dla rekurencyjnego liczenia wyznacznika')

plt.tight_layout()
plt.show()
