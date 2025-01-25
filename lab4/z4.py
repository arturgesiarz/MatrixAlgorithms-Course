import numpy as np
import matplotlib.pyplot as plt



def generate_3d_grid_matrix(k):
    size = 2**k
    matrix = np.zeros((size**3, size**3))

    for x in range(size):
        for y in range(size):
            for z in range(size):
                index = x * size**2 + y * size + z

                neighbors = [
                    (x - 1, y, z), (x + 1, y, z),
                    (x, y - 1, z), (x, y + 1, z),
                    (x, y, z - 1), (x, y, z + 1)
                ]

                for nx, ny, nz in neighbors:
                    if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                        neighbor_index = nx * size**2 + ny * size + nz
                        matrix[index, neighbor_index] = np.random.rand()

    return matrix

def compress_matrix(matrix, max_rank, epsilon=1e-10):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    rank = min(max_rank, np.sum(S > epsilon))
    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    V_reduced = Vt[:rank, :]

    compressed_matrix = U_reduced @ np.diag(S_reduced) @ V_reduced
    return compressed_matrix, U_reduced, S_reduced, V_reduced

def draw_matrix(matrix, title="Matrix"):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title(title)
    plt.show()

def draw_vector(vector, title="Vector"):
    plt.figure(figsize=(10, 2))
    plt.plot(vector, marker='o')
    plt.title(title)
    plt.grid(True)
    plt.show()

def multiply_matrix_vector(matrix, vector):
    return matrix @ vector

def multiply_matrix_self(matrix):
    return matrix @ matrix

k_values = [2, 3, 4]
max_rank = 10

for k in k_values:
    print(f"\n--- Processing for k={k} ---")
    original_matrix = generate_3d_grid_matrix(k)

    compressed_matrix, U, S, Vt = compress_matrix(original_matrix, max_rank)

    draw_matrix(compressed_matrix, title=f"Compressed Matrix for k={k}")

    vector = np.random.rand(original_matrix.shape[1])
    result_vector = multiply_matrix_vector(compressed_matrix, vector)
    draw_vector(result_vector, title=f"Result Vector for k={k}")

    result_matrix = multiply_matrix_self(compressed_matrix)
    draw_matrix(result_matrix, title=f"Result Matrix (Matrix x Matrix) for k={k}")
