import numpy as np
import matplotlib.pyplot as plt

def truncated_svd(A, r):
    n, m = len(A), len(A[0])
    r = min(r, n, m)
    U = [[0 for _ in range(r)] for _ in range(n)]
    D = [0 for _ in range(r)]
    V = [[0 for _ in range(m)] for _ in range(r)]

    for i in range(r):
        sigma = 0
        for j in range(n):
            for k in range(m):
                sigma += A[j][k] ** 2
        sigma = sigma ** 0.5
        D[i] = sigma

        for j in range(n):
            U[j][i] = A[j][0] / sigma if sigma != 0 else 0
        for k in range(m):
            V[i][k] = A[0][k] / sigma if sigma != 0 else 0

        for j in range(n):
            for k in range(m):
                A[j][k] -= sigma * U[j][i] * V[i][k]

    return U, D, V



def compress_matrix(tmin, tmax, smin, smax, A, r, epsilon):
    if tmin > tmax or smin > smax:
        return {
            "rank": 0,
            "singular_values": [],
            "U": [],
            "V": [],
            "size": (0, 0),
            "sons": None
        }

    block = [[A[i][j] for j in range(smin, smax + 1)] for i in range(tmin, tmax + 1)]
    n, m = len(block), len(block[0]) if block else (0, 0)

    if n == 0 or m == 0:
        return {
            "rank": 0,
            "singular_values": [],
            "U": [],
            "V": [],
            "size": (n, m),
            "sons": None
        }

    U, D, V = truncated_svd(block, r + 1)

    if len(D) <= r or D[r] < epsilon:
        return {
            "rank": r,
            "singular_values": D[:r],
            "U": [row[:r] for row in U],
            "V": [col[:r] for col in V],
            "size": (n, m),
            "sons": None
        }

    mid_t = (tmin + tmax) // 2
    mid_s = (smin + smax) // 2

    top_left = compress_matrix(tmin, mid_t, smin, mid_s, A, r, epsilon)
    top_right = compress_matrix(tmin, mid_t, mid_s + 1, smax, A, r, epsilon)
    bottom_left = compress_matrix(mid_t + 1, tmax, smin, mid_s, A, r, epsilon)
    bottom_right = compress_matrix(mid_t + 1, tmax, mid_s + 1, smax, A, r, epsilon)

    return {
        "rank": r,
        "size": (n, m),
        "sons": [top_left, top_right, bottom_left, bottom_right],
    }



def generate_random_bitmap(size):
    import random
    return [[[random.randint(0, 255) for _ in range(3)] for _ in range(size)] for _ in range(size)]


def process_rgb_bitmap(bitmap, r, epsilon):

    height, width, _ = len(bitmap), len(bitmap[0]), len(bitmap[0][0])

    red_channel = [[bitmap[i][j][0] for j in range(width)] for i in range(height)]
    green_channel = [[bitmap[i][j][1] for j in range(width)] for i in range(height)]
    blue_channel = [[bitmap[i][j][2] for j in range(width)] for i in range(height)]

    compressed_red = compress_matrix(0, height - 1, 0, width - 1, red_channel, r, epsilon)
    compressed_green = compress_matrix(0, height - 1, 0, width - 1, green_channel, r, epsilon)
    compressed_blue = compress_matrix(0, height - 1, 0, width - 1, blue_channel, r, epsilon)

    return {
        "red": compressed_red,
        "green": compressed_green,
        "blue": compressed_blue
}


def reconstruct_matrix(compressed, size):
    if compressed["sons"] is None:
        if not compressed["singular_values"]:
            return [[0 for _ in range(size[1])] for _ in range(size[0])]
        U = np.array(compressed["U"])
        S = np.diag(compressed["singular_values"])
        V = np.array(compressed["V"])
        return (U @ S @ V).tolist()
    else:
        h, w = size
        mid_h, mid_w = h // 2, w // 2

        top_left = reconstruct_matrix(compressed["sons"][0], (mid_h, mid_w))
        top_right = reconstruct_matrix(compressed["sons"][1], (mid_h, w - mid_w))
        bottom_left = reconstruct_matrix(compressed["sons"][2], (h - mid_h, mid_w))
        bottom_right = reconstruct_matrix(compressed["sons"][3], (h - mid_h, w - mid_w))

        top = [tl + tr for tl, tr in zip(top_left, top_right)]
        bottom = [bl + br for bl, br in zip(bottom_left, bottom_right)]
        return top + bottom


def visualize_results(original_bitmap, compressed_result, r):

    height = len(original_bitmap)
    width = len(original_bitmap[0])
    
    red_approx = reconstruct_matrix(compressed_result["red"], (height, width))
    green_approx = reconstruct_matrix(compressed_result["green"], (height, width))
    blue_approx = reconstruct_matrix(compressed_result["blue"], (height, width))
    
    compressed_bitmap = [
        [
            [
                red_approx[i][j],
                green_approx[i][j],
                blue_approx[i][j]
            ]
            for j in range(width)
        ]
        for i in range(height)
    ]
    
    original_bitmap_np = np.array(original_bitmap, dtype=np.uint8)
    compressed_bitmap_np = np.clip(np.array(compressed_bitmap), 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original RGB Bitmap")
    plt.imshow(original_bitmap_np)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Compressed RGB Bitmap (r = {r})")
    plt.imshow(compressed_bitmap_np)
    plt.axis('off')
    
    plt.show()




size = 300  #rozmiar bitmapy
r = 10      #maksymalny rank
epsilon = 10  #minimalna wartosc osobliwa

bitmap = generate_random_bitmap(size)
compressed_result = process_rgb_bitmap(bitmap, r, epsilon)

visualize_results(bitmap, compressed_result, r)
