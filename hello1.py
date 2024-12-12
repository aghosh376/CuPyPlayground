import numpy as np
import cupy as cp
import time
from cupy import cutensor as cutensor

# Ensure GPU synchronization at the start
cp.cuda.Stream.null.synchronize()

def contract(A, X, dim):
    """
    Contracts a tensor `A` with a matrix `X` along the specified dimension `dim`.

    Parameters:
        A (array-like): Tensor to be contracted.
        X (array-like): Matrix for contraction.
        dim (int): Dimension to contract over.

    Returns:
        array-like: Resultant tensor after contraction.
    """
    if not isinstance(dim, int):
        raise ValueError("Dim must be an int.")

    # Convert inputs to CuPy arrays if they are NumPy arrays
    A = cp.array(A) if isinstance(A, np.ndarray) else A
    X = cp.array(X) if isinstance(X, np.ndarray) else X

    A_shape = cp.array(A.shape)
    X_shape = cp.array(X.shape)

    if A_shape[dim] != X_shape[1]:
        raise ValueError(
            f"Dimension mismatch: A has size {A_shape[dim]} along dimension {dim}, but X has size {X_shape[1]}"
        )

    # Reshape A for contraction
    reshape_left = cp.prod(A_shape[:dim]).item()
    reshape_right = cp.prod(A_shape[dim + 1:]).item()
    A_reshaped = A.reshape(reshape_left, A_shape[dim], reshape_right)

    # Perform contraction using einsum
    result = cp.einsum("ijk,kl->ijl", A_reshaped, X)

    # Reshape result back to original dimensions
    result_shape = (*A_shape[:dim], X_shape[0], *A_shape[dim + 1:])
    return result.reshape(result_shape)

def contract_with_plan(A, X, dim):
    """
    Contracts a tensor `A` with a matrix `X` using CuTensor for optimized contraction.

    Parameters:
        A (array-like): Tensor to be contracted.
        X (array-like): Matrix for contraction.
        dim (int): Dimension to contract over.

    Returns:
        array-like: Resultant tensor after contraction.
    """
    if not isinstance(dim, int):
        raise ValueError("Dim must be an int.")

    # Convert inputs to CuPy arrays if they are NumPy arrays
    A = cp.array(A) if isinstance(A, np.ndarray) else A
    X = cp.array(X) if isinstance(X, np.ndarray) else X

    A_shape = A.shape
    X_shape = X.shape

    if A_shape[dim] != X_shape[1]:
        raise ValueError(
            f"Dimension mismatch: A has size {A_shape[dim]} along dimension {dim}, but X has size {X_shape[1]}"
        )

    # Collapse dimensions for easier contraction
    collapsed_shape = (
        int(np.prod(A_shape[:dim])),
        A_shape[dim],
        int(np.prod(A_shape[dim + 1:]))
    )
    A_collapsed = A.reshape(collapsed_shape)

    # Define contraction modes
    A_mode = ('a', 'b', 'c')
    X_mode = ('d', 'b')
    result_mode = ('a', 'd', 'c')

    result_collapsed = cp.empty((collapsed_shape[0], X_shape[0], collapsed_shape[2]), dtype=A.dtype)

    # Perform contraction using CuTensor
    alpha = 1.0
    beta = 0.0
    result = cutensor.contraction(
        alpha, A_collapsed, A_mode, X, X_mode, beta, result_collapsed, result_mode
    )

    cp.cuda.Stream.null.synchronize()

    # Reshape result back to original dimensions
    result_shape = (*A_shape[:dim], X_shape[0], *A_shape[dim + 1:])
    return result.reshape(result_shape)

def multipleContraction(A, matrices, contraction_dims):
    """
    Applies multiple contractions sequentially.

    Parameters:
        A (array-like): Initial tensor.
        matrices (list): List of matrices for contraction.
        contraction_dims (list): List of dimensions to contract over.

    Returns:
        array-like: Tensor after multiple contractions.
    """
    if len(matrices) != len(contraction_dims):
        raise ValueError("Matrices and contraction dimensions must have equal length.")

    result = A
    for i, (matrix, dim) in enumerate(zip(matrices, contraction_dims)):
        result = contract_with_plan(result, matrix, dim)
    return result

def sumMultiContraction(A, contraction_lists, coefficients):
    """
    Performs multiple contractions and sums the results with coefficients.

    Parameters:
        A (array-like): Tensor to be contracted.
        contraction_lists (list): [matrices, dimensions] for contractions.
        coefficients (list): Coefficients for summation.

    Returns:
        array-like: Summed result of all contractions.
    """
    matrices_list, dims_list = contraction_lists

    if len(matrices_list) != len(dims_list):
        raise ValueError("Matrices and dimensions lists must have equal length.")

    if not coefficients:
        coefficients = [1.0] * len(matrices_list)

    total = None

    for i, (matrices, dims) in enumerate(zip(matrices_list, dims_list)):
        raw_result = multipleContraction(A, matrices, dims)

        if total is None:
            total = cp.zeros_like(raw_result)

        total += raw_result * coefficients[i]

    return total

def test_sumMultiContraction():
    """
    Tests the sumMultiContraction function with varying tensor dimensions and sizes.
    """
    num_tests = 10
    benchmark_times = []

    np.random.seed(42)

    for num_dims in range(6, 9):
        for dim_size in range(2, 31):
            twrite = []

            try:
                A = np.random.random([dim_size] * num_dims)

                matrices1 = [
                    np.random.random((6, dim_size)),
                    np.random.random((7, 6))
                ]
                matrices2 = [
                    np.random.random((6, dim_size)),
                    np.random.random((7, 6))
                ]

                matrices = [matrices1, matrices2]
                dims = [[1, 1], [1, 1]]
                coefficients = [1.5, -0.5]

                for _ in range(num_tests):
                    t1 = time.time()
                    result = sumMultiContraction(A, [matrices, dims], coefficients)
                    t1 = time.time() - t1
                    twrite.append(t1)

                times = np.array(twrite)
                print(
                    f"Test passed for {num_dims} dimensions, size {dim_size}:\n"
                    f"    Avg Time: {times.mean():.5f}s, Max Time: {times.max():.5f}s, Min Time: {times.min():.5f}s"
                )

                benchmark_times.extend(twrite)

            except Exception as e:
                print(f"Test failed for {num_dims} dimensions, size {dim_size}: {e}")

    if benchmark_times:
        overall_times = np.array(benchmark_times)
        print("\nOverall Benchmark Results:")
        print(f"    Avg Time: {overall_times.mean():.5f}s")
        print(f"    Max Time: {overall_times.max():.5f}s")
        print(f"    Min Time: {overall_times.min():.5f}s")

test_sumMultiContraction()
