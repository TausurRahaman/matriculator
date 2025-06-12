from core.matrix import Matrix
    
if __name__ == "__main__":

    # Declaration of Matrices
    A = Matrix(2,3,[[1,2,3],[4,5,6]])
    B = Matrix(2,3,[[7,8,9],[10,11,12]])
    C = Matrix(3,3,[[13,14,15],[16,17,18],[54,23,89]])
    D = Matrix(3,3,[[19,20,21],[22,23,24],[22,65,99]])
    E = Matrix(2,3,[[1,2,3],[4,5,6]])
    F = Matrix(5,5)
    M = Matrix(3, 3, [[6, 2, 1],
                  [2, 3, 1],
                  [1, 1, 1]])

    # Testing functions

    # print(A)
    # print(A+B)
    # print(C+D)
    # print(C-D)
    # print(sum([A,B]))
    # print(A*2)
    # print(A*2.6)
    # print(A*2.66476)
    # print(3*A)
    # print(7*A-B*2)
    # print(A@C)
    # print(C@C)
    # print(C*C)
    # print(C.transpose())
    # print(C.determinant())
    # print((C @ D.transpose()).determinant())
    # print(C.inverse())
    # print(C.inverse() * D.transpose() + 2*C)
    # print(A==E)
    # print(F)
    # print(F.identity())
    # print(C**5)
    # print(C**1000)

    # mat_a = input_matrix()
    # print(mat_a**2)

    print("Characteristic Polynomial:", M.characteristic_polynomial())
    print("Eigenvalues:", M.eigenvalues())

    print("Eigenvectors:")
    for val, vec in M.eigenvectors():
        print(f"λ = {val}\n→ {vec}")


