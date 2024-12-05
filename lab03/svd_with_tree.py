import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def power_iteration(A, num_simulations: int = 100):
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    # Wartość własna
    eigenvalue = np.dot(b_k, np.dot(A, b_k)) / np.dot(b_k, b_k)
    return b_k, eigenvalue

def svd_decomposition(A, k, epsilon=1e-10):
    m, n = A.shape
    U = np.zeros((m, k))
    S = []
    V = np.zeros((k, n))
    B = np.dot(A.T, A)
    num_valid_components = 0
    for i in range(k):
        v, sigma_squared = power_iteration(B)
        if sigma_squared < epsilon * epsilon:
            break
        sigma = np.sqrt(sigma_squared)
        S.append(sigma)
        V[num_valid_components, :] = v
        u = np.dot(A, v) / sigma
        U[:, num_valid_components] = u
        B -= sigma_squared * np.outer(v, v)
        num_valid_components += 1
        if np.allclose(B, 0, atol=epsilon):
            break
    S = np.array(S)
    U = U[:, :num_valid_components]
    V = V[:num_valid_components, :]
    return U, S, V


A = np.random.rand(5, 4)
k = 3 # liczba maksymalnych wymiarów do zachowania
epsilon = 1e-2 # próg odrzucania małych wartości osobliwych
U, S, V = svd_decomposition(A, k, epsilon)
print("Oryginalna macierz:")
print(A)
print("\nMacierz U:")
print(U)
print("\nWartości osobliwe (S):")
print(S)
print("\nMacierz V:")
print(V)
SS = np.zeros((U.shape[1], V.shape[0]))
SS[:S.shape[0], :S.shape[0]] = np.diag(S)
print("\nMacierz:")

print(U @ SS @ V)


class Node:
    def __init__(self, rank, size, singular_values=None, U=None, V=None):
        self.rank = rank
        self.size = size
        self.singular_values = singular_values
        self.U = U
        self.V = V
        self.sons = []

    def draw_compression(self, matrix, x_bounds=None, y_bounds=None):
        if x_bounds is None or y_bounds is None:
            x_bounds=(0, self.size[0])
            y_bounds=(0, self.size[1])
        if len(self.sons) == 0:
            matrix[x_bounds[0] : x_bounds[1], y_bounds[0] : (y_bounds[0] + self.rank)] = 0
            matrix[x_bounds[0] : (x_bounds[0] + self.rank), y_bounds[0] :y_bounds[1]] = 0
            return
        x_mid = (x_bounds[0] + x_bounds[1]) // 2
        y_mid = (y_bounds[0] + y_bounds[1]) // 2
        self.sons[0].draw_compression(matrix, (x_bounds[0], x_mid),(y_bounds[0], y_mid))
        self.sons[1].draw_compression(matrix, (x_bounds[0], x_mid), (y_mid,y_bounds[1]))
        self.sons[2].draw_compression(matrix, (x_mid, x_bounds[1]),(y_bounds[0], y_mid))
        self.sons[3].draw_compression(matrix, (x_mid, x_bounds[1]), (y_mid,y_bounds[1]))

def svd_compress(A, max_rank, epsilon=1e-10):
    U, S, Vt = svd_decomposition(A, max_rank)
    significant_singular_values = S[S > epsilon]
    rank = min(max_rank, len(significant_singular_values))
    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    V_reduced = Vt[:rank, :]
    root = Node(rank=rank, size=A.shape, singular_values=S_reduced,U=U_reduced, V=V_reduced)
    return root

def reconstruct_matrix_from_tree(node):
    if not node.sons:
        return node.U @ np.diag(node.singular_values) @ node.V
    
    top_left = reconstruct_matrix_from_tree(node.sons[0])
    top_right = reconstruct_matrix_from_tree(node.sons[1])
    bottom_left = reconstruct_matrix_from_tree(node.sons[2])
    bottom_right = reconstruct_matrix_from_tree(node.sons[3])
    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    return np.vstack((top, bottom))


def compress_matrix_tree(A, max_rank, epsilon=1e-10, min_size=2):
    if A.shape[0] <= min_size or A.shape[1] <= min_size:
        return svd_compress(A, max_rank, epsilon)
    compressed = svd_compress(A, max_rank, epsilon)
    error = compute_compression_error(A, compressed)
    if error <= epsilon:
        return compressed
    root = Node(0, A.shape)
    mid_row = A.shape[0] // 2
    mid_col = A.shape[1] // 2
    submatrices = [
    A[:mid_row, :mid_col],
    A[:mid_row, mid_col:],
    A[mid_row:, :mid_col],
    A[mid_row:, mid_col:]]

    for submatrix in submatrices:
        if submatrix.size > 0:
            child_node = compress_matrix_tree(submatrix, max_rank, epsilon,min_size)
            root.sons.append(child_node)
    return root

def compute_compression_error(original, compressed):
    reconstruction = reconstruct_matrix_from_tree(compressed)
    return np.linalg.norm(original - reconstruction, ord='fro') / np.linalg.norm(original, ord='fro')


def compress_and_reconstruct_image(img_array, max_rank, epsilon=1e-3,min_size=2):
    red_channel, green_channel, blue_channel = img_array[:, :, 0], img_array[:,:, 1], img_array[:, :, 2]
    red_tree = compress_matrix_tree(red_channel, max_rank, epsilon, min_size)
    green_tree = compress_matrix_tree(green_channel, max_rank, epsilon,min_size)
    blue_tree = compress_matrix_tree(blue_channel, max_rank, epsilon, min_size)
    red_compression_matrix = np.ones(img_array.shape)
    red_tree.draw_compression(red_compression_matrix)
    green_compression_matrix = np.ones(img_array.shape)
    green_tree.draw_compression(green_compression_matrix)
    blue_compression_matrix = np.ones(img_array.shape)
    blue_tree.draw_compression(blue_compression_matrix)
    red_reconstructed = reconstruct_matrix_from_tree(red_tree)
    green_reconstructed = reconstruct_matrix_from_tree(green_tree)
    blue_reconstructed = reconstruct_matrix_from_tree(blue_tree)
    reconstructed_image = np.stack((
        red_reconstructed.clip(0, 1),
        green_reconstructed.clip(0, 1),
        blue_reconstructed.clip(0, 1)
    ), axis=-1)
    return img_array, reconstructed_image, \
    red_reconstructed, green_reconstructed, blue_reconstructed, \
    red_compression_matrix, green_compression_matrix,blue_compression_matrix


def read_image_from_path(path):
    image = Image.open(path)
    image = image.convert('RGB')
    return np.asarray(image) / 255.0

def draw_compression(original_image, max_rank, epsilon, title=""):
    if original_image is None:
        original_image = np.random.rand(64, 64, 3)
    original, compressed_image, \
        red_compressed, green_compressed, blue_compressed, \
        red_compression_matrix, green_compression_matrix,blue_compression_matrix = \
            compress_and_reconstruct_image(original_image, max_rank, epsilon)
    print(f'Mean error: {np.mean(np.absolute(original - compressed_image)):.3f}')
    print(f'MSE: {np.mean(np.square(original - compressed_image)):.3f}')
    plt.figure(figsize=(15, 15))
    plt.suptitle(title, fontsize=16)


    plt.subplot(3, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.title("Compressed Image")
    plt.imshow(compressed_image)
    plt.axis("off")

    plt.subplot(3, 3, 4)
    plt.title("Compressed Red Channel")
    plt.imshow(red_compressed, cmap='Reds')
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.title("Compressed Green Channel")
    plt.imshow(green_compressed, cmap='Greens')
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.title("Compressed Blue Channel")
    plt.imshow(blue_compressed, cmap='Blues')
    plt.axis("off")

    plt.subplot(3, 3, 7)
    plt.title("Red Compression Matrix")
    plt.imshow(red_compression_matrix)
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.title("Green Compression Matrix")
    plt.imshow(green_compression_matrix)
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.title("Blue Compression Matrix")
    plt.imshow(blue_compression_matrix)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


original_image = read_image_from_path('image.jpg')
red_channel, green_channel, blue_channel = original_image[:, :, 0],original_image[:, :, 1], original_image[:, :, 2]
max_rank = original_image.shape[0]
_, red_S, _ = svd_decomposition(red_channel, max_rank)
_, green_S, _ = svd_decomposition(green_channel, max_rank)
_, blue_S, _ = svd_decomposition(blue_channel, max_rank)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Red singular values")
plt.bar(range(1, len(red_S) + 1), red_S, color='red')
plt.subplot(1, 3, 2)
plt.title("Green singular values")
plt.bar(range(1, len(green_S) + 1), green_S, color='green')
plt.subplot(1, 3, 3)
plt.title("Blue singular values")
plt.bar(range(1, len(blue_S) + 1), blue_S, color='blue')
plt.show()
for max_rank in (1, 4):
    for color, channel_eigenvalues in (("red", red_S), ("green", green_S),("blue", blue_S)):
        draw_compression(original_image, max_rank, channel_eigenvalues[1],
        f'Max rank: {max_rank}, epsilon:{channel_eigenvalues[1]:.3f} (first singular value of {color})')
        draw_compression(original_image, max_rank, channel_eigenvalues[-1],
        f'Max rank: {max_rank}, epsilon:{channel_eigenvalues[-1]:.3f} (last singular value of {color})')
        draw_compression(original_image, max_rank,channel_eigenvalues[len(channel_eigenvalues)//2],
        f'Max rank: {max_rank}, epsilon:{channel_eigenvalues[len(channel_eigenvalues)//2]:.3f} (middle singularvalue of {color})')