import numpy as np
import cupy as cp
import time
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
    if (type(dim) != int) :
        raise ValueError(f"Dim must be an int")

    A_shape = A.shape
    X_shape = X.shape

    if A_shape[dim] != X_shape[1]:
      raise ValueError(f"Dimension mismatch: A has size {A_shape[dim]} along dimension {dim}, "f"but X has size {X_shape[1]}")


    A_dims = len(A_shape)
    X_dims = len(X_shape)

    if X_dims > 2:
        raise ValueError(f"Contractiong matrix has {X_dims} dimensions when it should only have 2")

    #reshape the tensor to make contraction easier, needed to make sure they were ints because reshape kept crashing
    collapsed_shape = (int(np.prod(A_shape[:dim]).item()), A_shape[dim], int(np.prod(A_shape[dim+1:]).item()))
    A_collapsed = A.reshape(collapsed_shape)


    #A_mode = tuple(map(chr, range(97, 97 + A_dims)))
    #X_mode = (chr(97 + A_dims), A_mode[dim])
    A_mode = ('a','b','c')
    X_mode = ('d', 'b')

    #result_mode = tuple(A_mode[:dim] + tuple(X_mode[0]) + A_mode[dim+1:])
    #result_shape = A_shape[:dim] + tuple(X_shape[0]) + A_shape[dim+1:]
    #result_empty = cp.empty(result_shape)

    result_mode = ('a', 'd', 'c')
    result_shape_collapsed = (collapsed_shape[0], X_shape[0], collapsed_shape[2])
    result_collapsed = cp.empty(result_shape_collapsed)

    alpha = 1.0
    beta = 0.0
    #result = cutensor.contraction(alpha, A, A_mode, X, X_mode, beta, result_empty, result_mode)
    result = cutensor.contraction(alpha, A_collapsed, A_mode, X, X_mode, beta, result_collapsed, result_mode)

    result_shape = A_shape[:dim] + (X_shape[0],) + A_shape[dim+1:]
    result = result.reshape(result_shape)

    return result
  
def multipleContraction(A, matrices, contraction_dims):
    result = A

    # Check if matrices or contraction_dims is empty
    if not matrices or not contraction_dims or len(matrices) != len(contraction_dims):
        raise ValueError("Matrices and contraction dimensions must not be empty and must be of equal length.")

    # Iterate through each matrix and corresponding contraction dimension
    for i in range(len((matrices))):
        dim = contraction_dims[i]  
        result = contract_with_plan(result, matrices[i], dim)

    return result
      
      

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

result_AB_func = contract_with_plan(a, b, 1)

cp.cuda.Stream.null.synchronize()

print(cp.allclose(result_AB, result_AB_func))

#cutensor.contraction(alpha, ab, mode_ab, c, mode_c, beta, abc, mode_abc)
#
#cp.cuda.Stream.null.synchronize()
#
#
#size = A.data.nybytes

#print(result_AB)
#print(contract(a, b, 1))

#twrite = []
#for i in range(10):
#    a = cp.random.random((3, 4, 5))
#    b = cp.random.random((6, 4))
#    ab = cp.empty(3,6,5)
#    t1 = time.time()
#    result_AB_func = contract_with_plan(a, b, 1)
#    cp.cuda.Stream.null.synchronize()
#    t1 = time.time() - t1
#    twrite.append(t1)
#
#times = np.array(twrite)

#print(f"\tWRITE  mean= {times.mean():12.5f} max= {times.max():12.5f} min= {times.min():12.5f} std= {times.std():12.5f}",flush=True)
