import constant
from functions import *


def main():
    N = [100, 500, 1000, 2000, 3000]
    Jacobi_time = []
    Gauss_Seidel_time = []
    LU_time = []
    for i in range(5):
        A = create_matrix_A(5 + constant.e, -1, -1, N[i])
        b = create_vector_b(constant.f, N[i])
        Jacobi_time.append(Jacobi_method(A, b, N[i]))
        Gauss_Seidel_time.append(Gauss_Seidel_method(A, b, N[i]))
        LU_time.append(LU_method(A, b, N[i]))

    show_plot(N, Jacobi_time, Gauss_Seidel_time, LU_time)


main()

