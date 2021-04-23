import numpy as np
import math
import time
import matplotlib.pyplot as plt


def create_matrix_A(a1, a2, a3, N):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i][j] = a1
            if i == j + 1 or i + 1 == j:
                A[i][j] = a2
            if i == j + 2 or i + 2 == j:
                A[i][j] = a3

    return A


def create_vector_b(f, N):
    b = np.zeros(N)
    for i in range(N):
        b[i] = math.sin((i+1)*(f+1))

    return b


def Jacobi_method(A, b, N):
    accuracy = 10 ** (-9)
    x = np.ones(N)
    curr_x = np.ones(N)
    iterations = 0

    t = time.time()
    while count_euclidean_norm(count_residuum_vector(A, x, b)) > accuracy:
        for i in range(N):
            sigma1 = 0
            sigma2 = 0
            for j in range(0, i):
                sigma1 += A[i][j] * x[j]
            for j in range(i+1, N):
                sigma2 += A[i][j] * x[j]
            curr_x[i] = (b[i] - sigma1 - sigma2) / A[i][i]

        for j in range(N):
            x[j] = curr_x[j]

        iterations += 1
    elapsed = time.time() - t

    print("Iterations (Jacobi method for N = " + str(N) + "): " + str(iterations))

    return elapsed


def Gauss_Seidel_method(A, b, N):
    accuracy = 10 ** (-9)
    x = np.ones(N)
    iterations = 0

    t = time.time()
    while count_euclidean_norm(count_residuum_vector(A, x, b)) > accuracy:
        for i in range(N):
            sigma1 = 0
            sigma2 = 0
            for j in range(0, i):
                sigma1 += A[i][j] * x[j]
            for j in range(i + 1, N):
                sigma2 += A[i][j] * x[j]
            x[i] = (b[i] - sigma1 - sigma2) / A[i][i]

        iterations += 1
    elapsed = time.time() - t

    print("Iterations (Gauss-Seidel method for N = " + str(N) + "): " + str(iterations))

    return elapsed


def count_residuum_vector(A, x, b):
    return A.dot(x) - b  # A * x - b


def count_euclidean_norm(r):
    norm = 0
    for i in range(len(r)):
        norm += r[i] * r[i]

    return math.sqrt(norm)


def LU_method(A, b, N):
    t = time.time()
    # finding L and U matrices
    L = np.eye(N)
    U = copy_matrix(A)
    for k in range(N - 1):
        for j in range(k + 1, N):
            L[j][k] = U[j][k] / U[k][k]
            for i in range(k, N):
                U[j][i] -= L[j][k] * U[k][i]

    # L*y = b (forward substitution method)
    y = np.zeros(N)
    for i in range(N):
        S = 0
        for k in range(i):
            S += y[k] * L[i][k]
        y[i] = (b[i] - S) / L[i][i]

    # U*x = y (back substitution method)
    x = np.zeros(N)
    for i in range(N-1, -1, -1):
        S = 0
        for k in range(i+1, N):
            S += x[k] * U[i][k]
        x[i] = (y[i] - S) / U[i][i]

    elapsed = time.time() - t

    return elapsed


def copy_matrix(A):
    B = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            B[i][j] = A[i][j]

    return B


def show_plot(N, Jacobi_time, Gauss_Seidel_time, LU_time):
    fig, ax = plt.subplots()
    ax.plot(N, Jacobi_time, label="Jacobi method")
    ax.plot(N, Gauss_Seidel_time, label="Gauss-Seidel method")
    ax.plot(N, LU_time, label="LU method")
    ax.legend(loc="upper left")

    ax.set_xlabel("rozmiar macierzy A (NxN)")
    ax.set_ylabel("czas [s]")
    ax.set_title("zależność czasu od rozmiaru macierzy A")

    fig.set_figwidth(10)
    fig.set_figheight(5)
    fig.tight_layout()

    plt.show()

