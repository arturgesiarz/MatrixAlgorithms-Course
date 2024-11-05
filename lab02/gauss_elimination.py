import numpy as np
import time
import matplotlib.pyplot as plt

flops = 0

def generate_random_matrix(n):
    return np.random.uniform(0.00000001, 1.0, (n, n))


def gauss_det(matrix):
    global flops
    
    n = len(matrix)

    if n == 1:
        return matrix[0][0]
    
    for i in range(n):
        if matrix[i][i] == 0:
            for k in range(i + 1, n):
                if matrix[k][i] != 0:
                    matrix[i], matrix[k] = matrix[k], matrix[i]
                    flops += n 
                    break
            else:
                return 0  
        
        for j in range(i + 1, n):
            ratio = matrix[j][i] / matrix[i][i]
            flops += 1
            
            for k in range(i, n):
                matrix[j][k] -= ratio * matrix[i][k]
                flops += 2 
    

    det = 1
    for i in range(n):
        det *= matrix[i][i]
        flops += 1 
    
    return det

matrix_sizes = []
times = []
flopss = []
max_size = 150

for size in range(1, max_size + 1):
    matrix = generate_random_matrix(size).tolist()
    flops = 0
    

    start_time = time.time()
    det = gauss_det(matrix)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    matrix_sizes.append(size)
    times.append(elapsed_time)
    flopss.append(flops)
    
    
    if abs(det) < 1:
        assert round(det, 5) == round(np.linalg.det(matrix), 5)
    elif abs(det) < 10000:
        assert round(det, 2) == round(np.linalg.det(matrix), 2)
    elif abs(det) < 10000000000:
        assert round(det) == round(np.linalg.det(matrix))
    elif abs(det) < 100000000000000:
        assert round(det, -4) == round(np.linalg.det(matrix), -4)
    elif abs(det) < 100000000000000000000:
        assert round(det, -10) == round(np.linalg.det(matrix), -10)
   
       
        

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(matrix_sizes, times, label='Czas działania (s)')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas działania (s)')
plt.title('Czas działania eliminacji Gaussa')



plt.subplot(1, 2, 2)
plt.plot(matrix_sizes, flopss, label='Liczba operacji')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Liczba operacji zmiennoprzecinkowych')
plt.title('Liczba operacji dla eliminacji Gaussa')



plt.tight_layout()
plt.show()
