from sympy import Matrix as SymMatrix, symbols, simplify


class Matrix:

    def __init__(self,row=0,col=0,data=None):
        if data is not None:
            self.data = data
            self.row = len(data)
            self.col = len(data[0])

            if self.row != row or self.col != col:
                raise ValueError("Dimension ERROR!!!")

        else:
            self.row = row
            self.col = col
            self.data = [[0 for _ in range(self.col)] for _ in range(self.row)]

    def __add__(self,other):
        if not isinstance(other, Matrix):
            raise TypeError("Subtraction is only supported between Matrix objects.")
        if self.row != other.row or self.col != other.col:
            raise ValueError("Dimension ERROR!!!")

        result = Matrix(self.row,self.col)
        for i in range(self.row):
            for j in range(self.col):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result
    
    def __radd__(self, other):
        if other == 0:  # Allows use with sum()
            return self
        return self.__add__(other)
    
    def __sub__(self,other):
        if not isinstance(other, Matrix):
            raise TypeError("Subtraction is only supported between Matrix objects.")
        if self.row != other.row or self.col != other.col:
            raise ValueError("Dimension ERROR!!!")
        
        result = Matrix(self.row,self.col)
        for i in range(self.row):
            for j in range(self.col):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result
    
    def __mul__(self,multiplier=1):
        if isinstance(multiplier, (int,float)):
            result = Matrix(self.row,self.col,
                        [[round(multiplier*self.data[i][j],5) for j in range(self.col)] for i in range(self.row)]
                        ) 
            return result
        if isinstance(multiplier, Matrix):
            return self.__matmul__(multiplier)
        
        raise TypeError("Can only multiply by a scalar (int or float)")
    
    def __rmul__(self,multiplier=1):
        return self.__mul__(multiplier)
    
    def __matmul__(self,other):
        if not isinstance(other, Matrix):
            raise TypeError("Expected a matrix but got something else!")
        if self.col != other.row:
            raise ValueError("Dimension ERROR!!! Must match the coloum-row property")
        result_data = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.col)) for j in range(other.col)
            ]
            for i in range(self.row)
        ]
        return Matrix(self.row,other.col,result_data)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Only compare with matrices")
        
        if self.row != other.row and self.col != other.col:
            return False
        
        return True if self.data == other.data else False

    def __pow__(self,exponent=1):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Power must be a non-negative integer")
        if self.row != self.col:
            raise ValueError("Only square matrices can be raised to a power")

        result = Matrix.identity(self.row)
        temp = self

        for _ in range(exponent):
            result = result * temp

        return result


    def isSquareMatrix(self):
        return True if self.row==self.col else False

    def transpose(self):
        return Matrix(self.col, self.row,
                  [[self.data[i][j] for i in range(self.row)] for j in range(self.col)])

    def determinant(self):
        if self.row != self.col:
            raise ValueError("Dimension ERROR !!! Must be a Square Matrix")
        
        if self.row == 1:
            return self.data[0][0]
        
        if self.row == 2:
            return self.data[0][0]*self.data[1][1] - self.data[0][1]*self.data[1][0]
        
        det = 0
        for j in range(self.col):
            submatrix = Matrix(self.row-1,self.col-1,
                [self.data[i][:j] + self.data[i][j+1:] for i in range(1,self.row)]
            )
            det += ((-1)**j) * self.data[0][j] * submatrix.determinant()
        return det

    def inverse(self):
        if self.row != self.col:
            raise ValueError("Dimension ERROR!!! Must be a square matrix")
        
        if self.determinant() == 0:
            raise ValueError("Determinant value is 0. Not invertible")
        
        adjugate = Matrix(self.row, self.col)
        for i in range(self.row):
            for j in range(self.col):
                submatrix = Matrix(self.row-1, self.col-1,
                    [self.data[x][:j] + self.data[x][j+1:] for x in range(self.row) if x != i]
                )
                adjugate.data[j][i] = ((-1)**(i+j)) * submatrix.determinant()
        
        determinant = self.determinant()
        inverse_matrix = Matrix(self.row, self.col,[
            [round(adjugate.data[i][j]/determinant,5) for j in range(self.col)] for i in range(self.row)
        ])

        return inverse_matrix

    def identity(n):
        return Matrix(n, n, [[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    def zeros(rows, cols):
        return Matrix(rows, cols, [[0 for _ in range(cols)] for _ in range(rows)])
    
    def to_list(self):
        return self.data


    # EigenVectors and EigenValues
    def to_sympy(self):
        return SymMatrix(self.data)
    def characteristic_polynomial(self):
        if not self.isSquareMatrix():
            raise ValueError("Matrix must be square")

        lam = symbols('λ')
        A = self.to_sympy()
        char_poly = A.charpoly(lam).as_expr()
        return simplify(char_poly)
    def eigenvalues(self):
        A = self.to_sympy()
        return A.eigenvals()  # Returns {eigenvalue: multiplicity}
    def eigenvectors(self):
        A = self.to_sympy()
        eigenvecs = A.eigenvects()  # This returns (val, mult, [vectors])
        result = []
        for item in eigenvecs:
            # SymPy's eigenvects() returns tuples of length 3 by default
            # but just to be safe, let's enforce the structure here
            if len(item) == 3:
                val, mult, vecs = item
            elif len(item) == 2:
                val, vecs = item
                mult = 1  # default multiplicity if missing
            else:
                # Unexpected format, raise error or handle gracefully
                raise ValueError(f"Unexpected eigenvects tuple length: {len(item)}")
            result.append((val, mult, vecs))
        return result


    
    def solve_linear_system(self, b):
        if not isinstance(b, Matrix):
            raise TypeError("Right-hand side must be a Matrix.")

        if self.row != b.row or b.col != 1:
            raise ValueError(f"Dimension mismatch: A is {self.row}x{self.col}, b is {b.row}x{b.col}")

        A = [row[:] for row in self.data]
        B = [row[0] for row in b.data]
        n = self.row
        m = self.col

        # Combine A and B to form augmented matrix
        aug = [A[i] + [B[i]] for i in range(n)]

        # Gaussian elimination
        for i in range(min(n, m)):
            # Find pivot
            pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
            if abs(aug[pivot_row][i]) < 1e-10:
                continue  # Skip this column (free variable)

            # Swap rows
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

            # Normalize pivot row
            pivot_val = aug[i][i]
            aug[i] = [x / pivot_val for x in aug[i]]

            # Eliminate below
            for j in range(i + 1, n):
                factor = aug[j][i]
                aug[j] = [aj - factor * ai for aj, ai in zip(aug[j], aug[i])]

        # Back substitution
        x = [0 for _ in range(m)]
        for i in range(min(m - 1, n - 1), -1, -1):
            if abs(aug[i][i]) < 1e-10:
                if abs(aug[i][-1]) > 1e-10:
                    raise ValueError("No solution: inconsistent system")
                else:
                    x[i] = 0  # Free variable (can also raise warning)
            else:
                x[i] = aug[i][-1] - sum(aug[i][j] * x[j] for j in range(i + 1, m))

        # Return result as Matrix
        return Matrix(n, 1, [[complex(round(val.real, 5), round(val.imag, 5))] for val in x])




    def __str__(self):
        return '\n'.join(' '.join(map(str, row)) for row in self.data)

