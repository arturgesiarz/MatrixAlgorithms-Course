import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def power_iteration(A, num_simulations=100):
    b = np.random.rand(A.shape[1])
    
    for _ in range(num_simulations):
        b_next = np.dot(A, b)
        b = b_next / np.linalg.norm(b_next)
    
    eigenvalue = np.dot(b, np.dot(A, b)) / np.dot(b, b)
    
    return b, eigenvalue


def svd_decomposition(A, k, epsilon=1e-10):
    m, n = A.shape
    U = np.zeros((m, k))
    S = []
    V = np.zeros((k, n))

    B = np.dot(A.T, A)
    num_components = 0

    for _ in range(k):
        v, sigma_squared = power_iteration(B)

        if sigma_squared < epsilon ** 2:
            break

        sigma = np.sqrt(sigma_squared)
        S.append(sigma)

        V[num_components, :] = v

        u = np.dot(A, v) / sigma
        U[:, num_components] = u

        B -= sigma_squared * np.outer(v, v)

        num_components += 1

        if np.allclose(B, 0, atol=epsilon):
            break

    S = np.array(S)
    U = U[:, :num_components]
    V = V[:num_components, :]

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
        self.children = []

    def draw_compression(self, matrix, x_bounds=(0, None), y_bounds=(0, None)):
        if x_bounds[1] is None: x_bounds = (x_bounds[0], self.size[0])
        if y_bounds[1] is None: y_bounds = (y_bounds[0], self.size[1])

        if not self.children:
            x_start, x_end = x_bounds
            y_start, y_end = y_bounds
            matrix[x_start:x_end, y_start:y_start + self.rank] = 0
            matrix[x_start:x_start + self.rank, y_start:y_end] = 0
            return

        x_mid = (x_bounds[0] + x_bounds[1]) // 2
        y_mid = (y_bounds[0] + y_bounds[1]) // 2

        self.children[0].draw_compression(matrix, (x_bounds[0], x_mid), (y_bounds[0], y_mid))
        self.children[1].draw_compression(matrix, (x_bounds[0], x_mid), (y_mid, y_bounds[1]))
        self.children[2].draw_compression(matrix, (x_mid, x_bounds[1]), (y_bounds[0], y_mid))
        self.children[3].draw_compression(matrix, (x_mid, x_bounds[1]), (y_mid, y_bounds[1]))


def svd_compress(A, max_rank, epsilon=1e-10):
    U, S, Vt = svd_decomposition(A, max_rank)

    rank = min(max_rank, np.sum(S > epsilon))
    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    V_reduced = Vt[:rank, :]

    return Node(rank=rank, size=A.shape, singular_values=S_reduced, U=U_reduced, V=V_reduced)


def reconstruct_matrix_from_tree(node):
    if not node.children:
        return node.U @ np.diag(node.singular_values) @ node.V

    parts = [reconstruct_matrix_from_tree(child) for child in node.children]
    top = np.hstack((parts[0], parts[1]))
    bottom = np.hstack((parts[2], parts[3]))
    return np.vstack((top, bottom))



def compress_matrix_tree(A, max_rank, epsilon=1e-10, min_size=2):
    if A.shape[0] <= min_size or A.shape[1] <= min_size:
        return svd_compress(A, max_rank, epsilon)

    compressed = svd_compress(A, max_rank, epsilon)

    if compute_compression_error(A, compressed) <= epsilon:
        return compressed

    root = Node(rank=0, size=A.shape)
    mid_row, mid_col = A.shape[0] // 2, A.shape[1] // 2

    submatrices = [
        A[:mid_row, :mid_col], 
        A[:mid_row, mid_col:], 
        A[mid_row:, :mid_col], 
        A[mid_row:, mid_col:]  
    ]

    for submatrix in submatrices:
        if submatrix.size > 0:
            root.children.append(compress_matrix_tree(submatrix, max_rank, epsilon, min_size))

    return root


def compute_compression_error(original, compressed):
    reconstructed = reconstruct_matrix_from_tree(compressed)
    
    return np.linalg.norm(original - reconstructed, 'fro') / np.linalg.norm(original, 'fro')



def compress_and_reconstruct_image(img_array, max_rank, epsilon=1e-3, min_size=2):
    channels = [img_array[:, :, i] for i in range(3)]
    
    trees = [compress_matrix_tree(channel, max_rank, epsilon, min_size) for channel in channels]
    reconstructions = [reconstruct_matrix_from_tree(tree) for tree in trees]
    compression_matrices = [np.ones(img_array.shape) for _ in range(3)]
    
    for tree, compression_matrix in zip(trees, compression_matrices):
        tree.draw_compression(compression_matrix)
    
    reconstructed_image = np.stack([channel.clip(0, 1) for channel in reconstructions], axis=-1)

    return (
        img_array, reconstructed_image, *reconstructions, *compression_matrices
    )



def read_image_from_path(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB')) / 255.0


def draw_compression(original_image=None, max_rank=10, epsilon=1e-3, title=""):
    if original_image is None:
        original_image = np.random.rand(64, 64, 3)
    
    (
        original, compressed_image,
        red_compressed, green_compressed, blue_compressed,
        red_compression_matrix, green_compression_matrix, blue_compression_matrix
    ) = compress_and_reconstruct_image(original_image, max_rank, epsilon)
    
    mean_error = np.mean(np.abs(original - compressed_image))
    mse_error = np.mean((original - compressed_image) ** 2)
    print(f"Mean error: {mean_error:.3f}")
    print(f"MSE: {mse_error:.3f}")
    
    plt.figure(figsize=(15, 15))
    plt.suptitle(title, fontsize=16)

    images_and_titles = [
        (original, "Original Image"),
        (compressed_image, "Compressed Image"),
        (red_compressed, "Compressed Red Channel"),
        (green_compressed, "Compressed Green Channel"),
        (blue_compressed, "Compressed Blue Channel"),
        (red_compression_matrix, "Red Compression Matrix"),
        (green_compression_matrix, "Green Compression Matrix"),
        (blue_compression_matrix, "Blue Compression Matrix"),
    ]

    positions = [1, 2, 4, 5, 6, 7, 8, 9]
    for pos, (image, title) in zip(positions, images_and_titles):
        plt.subplot(3, 3, pos)
        plt.title(title)
        plt.imshow(image, cmap=('Reds' if 'Red' in title else 
                                'Greens' if 'Green' in title else 
                                'Blues' if 'Blue' in title else None))
        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
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