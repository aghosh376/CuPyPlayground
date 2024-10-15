import numpy as np
import cupy as cp
from cupy import cutensor as cutensor

cp.cuda.Stream.null.synchronize()


#arr = cp.ones((1000,500,500))

#makes us wait for gpu to finish before returning
cp.cuda.Stream.null.synchronize()


#A = cp.array([[[1,2,3],
#               [3,4,5],
#               [6,7,8]],
#
#              [[9,10,11],
#               [12,13,14],
#               [15,16,17]]],)

#B = cp.random.rand(3,3,3)


#print("A: ", A)
#print("B: ", B)
#print(A.shape)
#print(B.shape)


def contract(A, X, dim):
    # Takes tensor A of unknown number of dimensions
    # Takes matrix h with 2 dimensions Kk
    # Takes int dim as the dimension to contract over
    if (type(dim) != int and type(dim) == str) : 
      raise ValueError(f"Dim must be an int not a string")

    # Get the dimensions of the tensor and matrix
    A_shape = cp.array(A.shape)
    X_shape = cp.array(X.shape)

    # Check that the size of the dimension matches
    if A_shape[dim] != X_shape[1]:
      raise ValueError(f"Dimension mismatch: A has size {A_shape[dim]} along dimension {dim}, "f"but X has size {X_shape[1]}")
    
    # Compress the dimensions not involved in contraction

    reshape_left = cp.prod(A_shape[:dim]).item()
    contract_dim_size = A_shape[dim].item()
    reshape_right = cp.prod(A_shape[dim+1:]).item()

    #print(f"Reshape Left: {reshape_left} (type: {type(reshape_left)})")
    #print(f"Contract Dim Size: {contract_dim_size} (type: {type(contract_dim_size)})")
    #print(f"Reshape Right: {reshape_right} (type: {type(reshape_right)})")


    As = A.reshape(reshape_left, contract_dim_size, reshape_right)

    result = cp.einsum("jkl, Kk->jKl", As, X)
    #print(*A_shape[:dim].tolist(), X.shape[0], *A_shape[dim + 1:].tolist())
    result = result.reshape(*A_shape[:dim].tolist(), X.shape[0], *A_shape[dim + 1:].tolist())
    
    return result


def contract_with_plan(A, X, dim):
    if (type(dim) != int and type(dim) == str) : 
        raise ValueError(f"Dim must be an int not a string")

    A_shape = cp.array(A.shape)
    X_shape = cp.array(X.shape)

    mode_a = ()
    


alpha = 1.0
beta = 0.0
mode_a = ('a', 'b', 'c')
mode_b = ('d', 'b')
mode_c = ('d', 'e')
mode_ab = ('a','d', 'c')
mode_abc = ('a', 'e')
a = cp.random.random((3, 4, 5))
b = cp.random.random((6, 4))
c = cp.random.random((6, 7))
ab = cp.empty((3, 6, 5))
abc = cp.empty((3, 7))
result_AB = cutensor.contraction(alpha, a, mode_a, b, mode_b, beta, ab, mode_ab)
#cutensor.contraction(alpha, ab, mode_ab, c, mode_c, beta, abc, mode_abc)

cp.cuda.Stream.null.synchronize()

print(result_AB)
print(contract(a, b, 1))
