# Recursive matrix multiplication using Strassen's method

import numpy as np
import math

def split_matrix(A):
    mid = A.shape[0] // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
        
    return A11,A12,A21,A22

def strassen(A: np.ndarray, B: np.ndarray):
    if A.shape == (1, 1):
        return A * B, 1

    A11, A12, A21, A22 = split_matrix(A)    
    B11, B12, B21, B22 = split_matrix(B)
    
    flops = [0] * 7

    M1, flops[0] = strassen(A11 + A22, B11 + B22)
    M2, flops[1] = strassen(A21 + A22, B11)
    M3, flops[2] = strassen(A11, B12 - B22)
    M4, flops[3] = strassen(A22, B21 - B11)
    M5, flops[4] = strassen(A11 + A12, B22)
    M6, flops[5] = strassen(A21 - A11, B11 + B12)
    M7, flops[6] = strassen(A12 - A22, B21 + B22)
 
    up = M1 + M4 - M5 + M7, M3 + M5
    down = M2 + M4, M1 - M2 + M3 + M6

    return np.vstack((np.hstack(up),np.hstack(down))), sum(flops) + 18 * math.prod(A.shape)