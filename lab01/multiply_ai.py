# AI matrix multiplication based on an article in Nature - 
# https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/#:~:text=In%20our%20paper,%20published%20today%20in%20Nature,%20we

import numpy as np


def ai_matrix_multiply(A, B):

    a11, a12, a13, a14, a15 = A[0, 0], A[0, 1], A[0, 2], A[0, 3], A[0, 4]
    a21, a22, a23, a24, a25 = A[1, 0], A[1, 1], A[1, 2], A[1, 3], A[1, 4]
    a31, a32, a33, a34, a35 = A[2, 0], A[2, 1], A[2, 2], A[2, 3], A[2, 4]
    a41, a42, a43, a44, a45 = A[3, 0], A[3, 1], A[3, 2], A[3, 3], A[3, 4]

    b11, b12, b13, b14, b15 = B[0, 0], B[0, 1], B[0, 2], B[0, 3], B[0, 4]
    b21, b22, b23, b24, b25 = B[1, 0], B[1, 1], B[1, 2], B[1, 3], B[1, 4]
    b31, b32, b33, b34, b35 = B[2, 0], B[2, 1], B[2, 2], B[2, 3], B[2, 4]
    b41, b42, b43, b44, b45 = B[3, 0], B[3, 1], B[3, 2], B[3, 3], B[3, 4]
    b51, b52, b53, b54, b55 = B[4, 0], B[4, 1], B[4, 2], B[4, 3], B[4, 4]


    h1 = a32 * (-b21 - b25 - b31)
    h2 = (a22 + a25 - a35) * (-b25 - b51)
    h3 = (-a31 - a41 + a42) * (-b11 + b25)
    h4 = (a12 + a14 + a34) * (-b25 - b41)
    h5 = (a15 + a22 + a25) * (-b24 + b51)
    h6 = (-a22 - a25 - a45) * (b23 + b51)
    h7 = (-a11 + a41 - a42) * (b11 + b24)
    h8 = (a32 - a33 - a43) * (-b23 + b31)
    h9 = (-a12 - a14 + a44) * (b23 + b41)
    h10 = (a22 + a25) * b51
    h11 = (-a21 - a41 + a42) * (-b11 + b22)
    h12 = (a41 - a42) * b11
    h13 = (a12 + a14 + a24) * (b22 + b41)
    h14 = (a13 - a32 + a33) * (b24 + b31)
    h15 = (-a12 - a14) * b41
    h16 = (-a32 + a33) * b31
    h17 = (a12 + a14 - a21 + a22 - a23 + a24 - a32 + a33 - a41 + a42) * b22
    h18 = a21 * (b11 + b12 + b52)
    h19 = (-a23) * (b31 + b32 + b52)
    h20 = (-a15 + a21 + a23 - a25) * (-b11 - b12 + b14 - b52)
    h21 = (a21 + a23 - a25) * b52
    h22 = (a13 - a14 - a24) * (b11 + b12 - b14 - b31 - b32 + b34 + b44)
    h23 = a13 * (-b31 + b34 + b44)
    h24 = a15 * (-b44 - b51 + b54)
    h25 = (-a11) * (b11 - b14)
    h26 = (-a13 + a14 + a15) * b44
    h27 = (a13 - a31 + a33) * (b11 - b14 + b15 + b35)
    h28 = (-a34) * (-b35 - b41 - b45)
    h29 = a31 * (b11 + b15 + b35)
    h30 = (a31 - a33 + a34) * b35
    h31 = (-a14 - a15 - a34) * (-b44 - b51 + b54 - b55)
    h32 = (a21 + a41 + a44) * (b13 - b41 - b42 - b43)
    h33 = a43 * (-b31 - b33)
    h34 = a44 * (-b13 + b41 + b43)
    h35 = (-a45) * (b13 + b51 + b53)
    h36 = (a23 - a25 - a45) * (b31 + b32 + b33 + b52)
    h37 = (-a41 - a44 + a45) * b13
    h38 = (-a23 - a31 + a33 - a34) * (b35 + b41 + b42 + b45)
    h39 = (-a31 - a41 - a44 + a45) * (b13 + b51 + b53 + b55)
    h40 = (-a13 + a14 + a15 - a44) * (-b31 - b33 + b34 + b44)
    h41 = (-a11 + a41 - a45) * (b13 + b31 + b33 - b34 + b51 + b53 - b54)
    h42 = (-a21 + a25 - a35) * (-b11 - b12 - b15 + b41 + b42 + b45 - b52)
    h43 = a24 * (b41 + b42)
    h44 = (a23 + a32 - a33) * (b22 - b31)
    h45 = (-a33 + a34 - a43) * (b35 + b41 + b43 + b45 + b51 + b53 + b55)
    h46 = (-a35) * (-b51 - b55)
    h47 = (a21 - a25 - a31 + a35) * (b11 + b12 + b15 - b41 - b42 - b45)
    h48 = (-a23 + a33) * (b22 + b32 + b35 + b41 + b42 + b45)
    h49 = (-a11 - a13 + a14 + a15 - a21 - a23 + a24 + a25) * (-b11 - b12 + b14)
    h50 = (-a14 - a24) * (b22 - b31 - b32 + b34 - b42 + b44)
    h51 = a22 * (b21 + b22 - b51)
    h52 = a42 * (b11 + b21 + b23)
    h53 = (-a12) * (-b21 + b24 + b41)
    h54 = (a12 + a14 - a22 - a25 - a32 + a33 - a42 + a43 - a44 - a45) * b23
    h55 = (a14 - a44) * (-b23 + b31 + b33 - b34 + b43 - b44)
    h56 = (a11 - a15 - a41 + a45) * (b31 + b33 - b34 + b51 + b53 - b54)
    h57 = (-a31 - a41) * (-b13 - b15 - b25 - b51 - b53 - b55)
    h58 = (-a14 - a15 - a34 - a35) * (-b51 + b54 - b55)
    h59 = (-a33 + a34 - a43 + a44) * (b41 + b43 + b45 + b51 + b53 + b55)
    h60 = (a25 + a45) * (b23 - b31 - b32 - b33 - b52 - b53)
    h61 = (a14 + a34) * (b11 - b14 + b15 - b25 - b44 + b45 - b51 + b54 - b55)
    h62 = (a21 + a41) * (b12 + b13 + b22 - b41 - b42 - b43)
    h63 = (-a33 - a43) * (-b23 - b33 - b35 - b41 - b43 - b45)
    h64 = (a11 - a13 - a14 + a31 - a33 - a34) * (b11 - b14 + b15)
    h65 = (-a11 + a41) * (-b13 + b14 + b24 - b51 - b53 + b54)
    h66 = (a11 - a12 + a13 - a15 - a22 - a25 - a32 + a33 - a41 + a42) * b24
    h67 = (a25 - a35) * (b11 + b12 + b15 - b25 - b41 - b42 - b45 + b52 + b55)
    h68 = (a11 + a13 - a14 - a15 - a41 - a43 + a44 + a45) * (-b31 - b33 + b34)
    h69 = (-a13 + a14 - a23 + a24) * (-b24 - b31 - b32 + b34 - b52 + b54)
    h70 = (a23 - a25 + a43 - a45) * (-b31 - b32 - b33)
    h71 = (-a31 + a33 - a34 + a35 - a41 + a43 - a44 + a45) * (-b51 - b53 - b55)
    h72 = (-a21 - a24 - a41 - a44) * (b41 + b42 + b43)
    h73 = (a13 - a14 - a15 + a23 - a24 - a25) * (b11 + b12 - b14 + b24 + b52 - b54)
    h74 = (a21 - a23 + a24 - a31 + a33 - a34) * (b41 + b42 + b45)
    h75 = (-(a12 + a14 - a22 - a25 - a31 + a32 + a34 + a35 - a41 + a42)) * b25
    h76 = (a13 + a33) * (-b11 + b14 - b15 + b24 + b34 - b35)


    c11 = -h10 + h12 + h14 - h15 - h16 + h53 + h5 - h66 - h7
    c21 = h10 + h11 - h12 + h13 + h15 + h16 - h17 - h44 + h51
    c31 = h10 - h12 + h15 + h16 - h1 + h2 + h3 - h4 + h75
    c41 = -h10 + h12 - h15 - h16 + h52 + h54 - h6 - h8 + h9
    c12 = h13 + h15 + h20 + h21 - h22 + h23 + h25 - h43 + h49 + h50
    c22 = -h11 + h12 - h13 - h15 - h16 + h17 + h18 - h19 - h21 + h43 + h44
    c32 = -h16 - h19 - h21 - h28 - h29 - h38 + h42 + h44 - h47 + h48
    c42 = h11 - h12 - h18 + h21 - h32 + h33 - h34 - h36 + h62 - h70
    c13 = h15 + h23 + h24 + h34 - h37 + h40 - h41 + h55 - h56 - h9
    c23 = -h10 + h19 + h32 + h35 + h36 + h37 - h43 - h60 - h6 - h72
    c33 = -h16 - h28 + h33 + h37 - h39 + h45 - h46 + h63 - h71 - h8
    c43 = h10 + h15 + h16 - h33 + h34 - h35 - h37 - h54 + h6 + h8 - h9
    c14 = -h10 + h12 + h14 - h16 + h23 + h24 + h25 + h26 + h5 - h66 - h7
    c24 = h10 + h18 - h19 + h20 - h22 - h24 - h26 - h5 - h69 + h73 
    c34 = -h14 + h16 - h23 - h26 + h27 + h29 + h31 + h46 - h58 + h76
    c44 = h12 + h25 + h26 - h33 - h35 - h40 + h41 + h65 - h68 - h7 
    c15 = h15 + h24 + h25 + h27 - h28 + h30 + h31 - h4 + h61 + h64
    c25 = -h10 - h18 - h2 - h30 - h38 + h42 - h43 + h46 + h67 + h74
    c35 = -h10 + h12 - h15 + h28 + h29 - h2 - h30 - h3 + h46 + h4 - h75
    c45 = -h12 - h29 + h30 - h34 + h35 + h39 + h3 - h45 + h57 + h59
    
    C = np.zeros((4, 5))  

    C[0, 0], C[0, 1], C[0, 2], C[0, 3], C[0, 4] = c11, c12, c13, c14, c15
    C[1, 0], C[1, 1], C[1, 2], C[1, 3], C[1, 4] = c21, c22, c23, c24, c25
    C[2, 0], C[2, 1], C[2, 2], C[2, 3], C[2, 4] = c31, c32, c33, c34, c35
    C[3, 0], C[3, 1], C[3, 2], C[3, 3], C[3, 4] = c41, c42, c43, c44, c45


    return C
