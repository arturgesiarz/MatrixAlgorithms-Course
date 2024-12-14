import numpy as np
import matplotlib.pyplot as plt

def truncated_svd(A, rank):
    U, S, V = np.linalg.svd(A, full_matrices=False)
    S_truncated = np.diag(S[:rank])
    U_truncated = U[:, :rank]
    V_truncated = V[:rank, :]
    return U_truncated, S_truncated, V_truncated

def compress_matrix(A, t_min, t_max, s_min, s_max, rank, delta):
    if t_max - t_min <= 0 or s_max - s_min <= 0:
        return {
            "rank": 1,
            "U": np.array([[1]]),
            "S": np.array([[A[t_min, s_min]]]),
            "V": np.array([[1]]),
            "size": (t_max - t_min + 1, s_max - s_min + 1),
        }

    sub_matrix = A[t_min:t_max+1, s_min:s_max+1]
    U, S, V = truncated_svd(sub_matrix, rank)

    if S[-1, -1] < delta:
        return {
            "rank": rank,
            "U": U,
            "S": S,
            "V": V,
            "size": (t_max - t_min + 1, s_max - s_min + 1),
        }
    else:
        t_mid = (t_min + t_max) // 2
        s_mid = (s_min + s_max) // 2

        return {
            "top_left": compress_matrix(A, t_min, t_mid, s_min, s_mid, rank, delta),
            "top_right": compress_matrix(A, t_min, t_mid, s_mid+1, s_max, rank, delta),
            "bottom_left": compress_matrix(A, t_mid+1, t_max, s_min, s_mid, rank, delta),
            "bottom_right": compress_matrix(A, t_mid+1, t_max, s_mid+1, s_max, rank, delta),
            "size": (t_max - t_min + 1, s_max - s_min + 1),
        }

def reconstruct_matrix(compressed_matrix):
    if isinstance(compressed_matrix, dict) and "rank" in compressed_matrix:
        U = compressed_matrix["U"]
        S = compressed_matrix["S"]
        V = compressed_matrix["V"]
        return U @ S @ V
    else:
        top_left = reconstruct_matrix(compressed_matrix["top_left"])
        top_right = reconstruct_matrix(compressed_matrix["top_right"])
        bottom_left = reconstruct_matrix(compressed_matrix["bottom_left"])
        bottom_right = reconstruct_matrix(compressed_matrix["bottom_right"])
        
        top_left_rows, top_left_cols = top_left.shape
        top_right_rows, top_right_cols = top_right.shape
        bottom_left_rows, bottom_left_cols = bottom_left.shape
        bottom_right_rows, bottom_right_cols = bottom_right.shape

        max_top_rows = max(top_left_rows, top_right_rows)
        max_bottom_rows = max(bottom_left_rows, bottom_right_rows)
        max_left_cols = max(top_left_cols, bottom_left_cols)
        max_right_cols = max(top_right_cols, bottom_right_cols)

        top_left = np.pad(top_left, ((0, max_top_rows - top_left_rows), (0, max_left_cols - top_left_cols)), mode='constant')
        top_right = np.pad(top_right, ((0, max_top_rows - top_right_rows), (0, max_right_cols - top_right_cols)), mode='constant')
        bottom_left = np.pad(bottom_left, ((0, max_bottom_rows - bottom_left_rows), (0, max_left_cols - bottom_left_cols)), mode='constant')
        bottom_right = np.pad(bottom_right, ((0, max_bottom_rows - bottom_right_rows), (0, max_right_cols - bottom_right_cols)), mode='constant')

        top = np.hstack((top_left, top_right))
        bottom = np.hstack((bottom_left, bottom_right))
        return np.vstack((top, bottom))


def create_rgb_matrices(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return R, G, B

def compress_image(image, rank, delta):
    R, G, B = create_rgb_matrices(image)
    compressed_R = compress_matrix(R, 0, R.shape[0]-1, 0, R.shape[1]-1, rank, delta)
    compressed_G = compress_matrix(G, 0, G.shape[0]-1, 0, G.shape[1]-1, rank, delta)
    compressed_B = compress_matrix(B, 0, B.shape[0]-1, 0, B.shape[1]-1, rank, delta)
    return compressed_R, compressed_G, compressed_B

def visualize_compression(R, G, B, compressed_R, compressed_G, compressed_B):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.title("Original R")
    plt.imshow(R, cmap='Reds')
    plt.subplot(2, 3, 2)
    plt.title("Original G")
    plt.imshow(G, cmap='Greens')
    plt.subplot(2, 3, 3)
    plt.title("Original B")
    plt.imshow(B, cmap='Blues')

    plt.subplot(2, 3, 4)
    plt.title("Compressed R")
    plt.imshow(reconstruct_matrix(compressed_R), cmap='Reds')
    plt.subplot(2, 3, 5)
    plt.title("Compressed G")
    plt.imshow(reconstruct_matrix(compressed_G), cmap='Greens')
    plt.subplot(2, 3, 6)
    plt.title("Compressed B")
    plt.imshow(reconstruct_matrix(compressed_B), cmap='Blues')

    plt.tight_layout()
    plt.show()




np.random.seed(42)
image = np.random.randint(0, 256, (500, 500, 3))
rank = 10
delta = 1e-3

compressed_R, compressed_G, compressed_B = compress_image(image, rank, delta)
visualize_compression(image[:, :, 0], image[:, :, 1], image[:, :, 2], compressed_R, compressed_G, compressed_B)
