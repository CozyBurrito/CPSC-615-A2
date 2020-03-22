from pprint import pprint
import numpy
import scipy
import scipy.linalg

A = numpy.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]])

# LU Factorization
print ("LU Factorization:")
P, L, U = scipy.linalg.lu(A)

print ("A:")
pprint(A)

print ("P:")
pprint(P)

print ("L:")
pprint(L)

print ("U:")
pprint(U)

print ()


# Cholesky Factorization
print ("Cholesky Factorization:")
L = numpy.linalg.cholesky(A)

print ("A:")
pprint(A)

print ("L:")
pprint(L)

print ()

# QR Factorization
print ("QR Factorization:")
Q, R = numpy.linalg.qr(A)

print ("A:")
pprint(A)

print ("Q:")
pprint(Q)

print ("R:")
pprint(R)

print ()