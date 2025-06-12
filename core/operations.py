def createIdentityMatrix(self):
    data = [
        [1 if i==j else 0 for j in range(self.col)] for i in range(self.row)
    ]
    return data

def powerMatrix(self,exponent):
    if not isinstance(exponent, (float,int)):
        raise ValueError("Power must be a float or integer")

    if self.row != self.col:
        raise TypeError("Must be a square matrix")

    if exponent == 0:
        return createIdentityMatrix(self)
    if exponent < 0:
        raise ValueError("Power must be a positive integer or float")
    
    # common use cases. So I hard coded it to reduce time
    if exponent == 1:
        return self
    if exponent == 2:
        return self * self
    if exponent == 3:
        return self * self * self
    
    # General solution for any power
    powered_mat = self.identity()
    for _ in range(exponent):
        powered_mat = powered_mat * self

    return powered_mat
    
    