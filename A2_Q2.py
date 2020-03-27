from pprint import pprint
import numpy
import scipy
import scipy.linalg
import timeit

A = numpy.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]])
n3 = A.shape[0] ** 3.0
lu_ops = (2.0/3.0) * n3
chol_ops = (1.0/3.0) * n3
qr_ops = (2 * n3) - ((2.0/3.0) * n3)

# LU Factorization
def lu_fact():
    P, L, U = scipy.linalg.lu(A)

# # Cholesky Factorization
def chol_fact():
    L = numpy.linalg.cholesky(A)

# QR Factorization
def qr_fact():
    Q, R = numpy.linalg.qr(A)

if __name__ == '__main__':
    lu_t = timeit.timeit("lu_fact()", setup="from __main__ import lu_fact", number=1)
    print("LU FLOPS: " + str(lu_ops/lu_t))

    chol_t = timeit.timeit("chol_fact()", setup="from __main__ import chol_fact", number=1)
    print("CH FLOPS: " + str(chol_ops/chol_t))

    qr_t = timeit.timeit("qr_fact()", setup="from __main__ import qr_fact", number=1)
    print("QR FLOPS: " + str(qr_ops/qr_t))