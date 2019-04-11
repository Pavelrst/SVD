import numpy as np
from my_svd import find_svd


def main():
    # mat = np.array([[1,2,3],[3,2,1],[1,0,-1]])
    mat = np.array([[2, 4],
                    [1, 3],
                    [0, 0],
                    [0, 0]])
    find_svd(mat, verbose=True)

if __name__ == "__main__":
    main()
