import time
import numpy as np

def multiply_matrices(matrix_a, matrix_b):
    """
    Perform matrix multiplication between two 2D lists (matrix_a and matrix_b).
    Args:
        matrix_a (list of list of int/float): Left matrix operand.
        matrix_b (list of list of int/float): Right matrix operand.
    Returns:
        list of list of int/float: The resulting matrix after multiplication.
    """
    num_rows_a = len(matrix_a)
    num_cols_a = len(matrix_a[0])
    num_cols_b = len(matrix_b[0])
    product = [[0 for _ in range(num_cols_b)] for _ in range(num_rows_a)]
    for row in range(num_rows_a):
        for col in range(num_cols_b):
            for k in range(num_cols_a):
                product[row][col] += matrix_a[row][k] * matrix_b[k][col]
    return product

if __name__ == "__main__":
    MATRIX_SIZE = 500  # Should match the size used in the MPI version for fair comparison
    np.random.seed(42)
    matrix_a = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
    matrix_b = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
    start_time = time.time()
    result_matrix = multiply_matrices(matrix_a, matrix_b)
    end_time = time.time()
    print(f"Serial runtime: {end_time - start_time:.6f} s")