import numpy as np
from numpy import linalg as la
from numpy import testing as npt

def find_svd(matrix, verbose=False):
    '''
    Anxp= Unxn Snxp VTpxp
    So to find the eigenvalues of the above entity we compute matrices AAT and ATA.
    Eigenvectors of AAT make up the columns of U
    Eigenvectors of ATA make up the columns of V
    S is the square root of the eigenvalues from AAT or ATA
    :param matrix:
    :param verbose:
    :return: Unxn, Snxp, VTpxp
    '''
    # Unxn
    AAT = np.dot(matrix, matrix.T)
    e_vals, e_vecs = find_eigens(AAT, verbose=False)
    U = e_vecs
    assert np.allclose(np.dot(U.T, U), np.identity(U.shape[0]))

    # VTpxp
    ATA = np.dot(matrix.T, matrix)
    e_vals, e_vecs = find_eigens(ATA, verbose=False)
    V = e_vecs
    assert np.allclose(np.dot(V.T, V), np.identity(V.shape[0]))
    VT = V.T

    # Snxp
    S = np.zeros((U.shape[0], VT.shape[0]))
    np.fill_diagonal(S, np.sort(np.sqrt(e_vals))[::-1])

    if verbose:
        print("AAT = \n ", AAT)
        print("ATA = \n ", ATA)
        print("U", U.shape, "= \n", U)
        print("S", S.shape, "= \n", S)
        print("V", V.shape, "= \n", V)
        print("VT", VT.shape, "= \n", VT)
    decomp = np.dot(np.dot(U,S),V)
    print("matrix = \n", matrix)
    print("decomp = \n", decomp)





def find_eigens(matrix, verbose=False):
    '''
    Given matrix, the function calculates
    it's eigenvalues and eigrnvectors.
    :param matrix: Input matrix
    :return: eigenvalues, eigenvectors sorted in descending order of eigenvalues.
    '''
    e_values, e_vectors = la.eig(matrix)

    idx = e_values.argsort()[::-1]
    e_values = e_values[idx]
    e_vectors = e_vectors[:, idx]

    if verbose:
        print("matrix = \n", matrix)
        print("e_values = \n", e_values)
        print("e_vectors = \n", e_vectors)
        for idx in range(len(e_values)):
            print("eigenvalue = ", e_values[idx], "corresponds to eigenvector = ", e_vectors[idx])

    # Test:
    for idx in range(len(e_values)):
        npt.assert_almost_equal(np.dot(matrix, e_vectors[:, idx]), np.dot(e_vectors[:, idx], e_values[idx]), decimal=10)

    return e_values, e_vectors