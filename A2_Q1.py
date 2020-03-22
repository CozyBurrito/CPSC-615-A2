from pprint import pprint
import numpy as np

# Compute the Q and R for a given A such that A = QR
def qr_factorization(A):
    m, n = A.shape  # Get the rows and columns of A
    Q = np.eye(m)   # Create an identity matrix that is the size of the rows of A, this will be the orthogonal matrix returned
    R = A.copy()    # Create an initial R matrix that is the copy of A, the will be the upper-triangular matrix returned 

    for i in range(n - (m == n)):   # For each column of A, except for the last column if A is square
        Qi = np.eye(m)   # Create an identity matrix that is the size of the rows of A, to use as the Q for this iteration
        Qi[i:, i:] = get_householder(R[i:, i])   # Set this iteration's Q matrix rows and columns based on the computed Householder matrix for this column unit vector
        Q = np.dot(Q, Qi)    # Multiply the Q for this iteration by the Q to be returned
        R = np.dot(Qi, R)    # Multiply the R to be returned by the Q for this iteration

    return Q, R
 
# Compute the Householder matrix for the column unit vector A
def get_householder(A):    
    u = A / (A[0] + np.copysign(np.linalg.norm(A), A[0]))   # Compute the u for the Householder equation
    u[0] = 1    # Set the first value of u to be 1
    H = np.eye(A.shape[0]) - (2 / np.dot(u, u)) * np.dot(u[:, None], u[None, :])    # H = I - (2 / (u * u)) * ((u-transpose) * u)
    return H

# Case 1 : m = n
A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
Q1, R1 = qr_factorization(A)
print ("Case 1 : m = n \nA:")
pprint(A)
print ("Q1:")
pprint(np.array(Q1))
print ("R1:")
pprint(np.array(R1))

print()

# Case 2 : m > n
B = np.array([[12, -51], [6, 167], [-4, 24]])
Q2, R2 = qr_factorization(B)
print ("Case 2 : m > n \nB:")
pprint(B)
print ("Q2:")
pprint(np.array(Q2))
print ("R2:")
pprint(np.array(R2))

print()