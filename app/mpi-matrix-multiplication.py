from mpi4py import MPI
import numpy as np

def multiply_matrices_partial(matrix_a_part, matrix_b):
    """
    Perform matrix multiplication for a subset of rows (matrix_a_part) and the full matrix_b.
    Args:
        matrix_a_part (list of list of int/float): Subset of rows from the left matrix operand.
        matrix_b (list of list of int/float): Full right matrix operand.
    Returns:
        list of list of int/float: The resulting matrix after multiplication for the subset.
    """
    num_rows = len(matrix_a_part)
    num_cols_a = len(matrix_a_part[0])
    num_cols_b = len(matrix_b[0])
    product = [[0 for _ in range(num_cols_b)] for _ in range(num_rows)]
    for row in range(num_rows):
        for col in range(num_cols_b):
            for k in range(num_cols_a):
                product[row][col] += matrix_a_part[row][k] * matrix_b[k][col]
    return product

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start timer for total execution time (across all processes)
    total_start = MPI.Wtime()

    # Only the root process (rank 0) initializes the full matrices and prepares data for distribution
    if rank == 0:
        MATRIX_SIZE = 500  # Size of the square matrices (can be adjusted for testing)
        np.random.seed(42)
        matrix_a = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
        matrix_b = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE)).tolist()
        num_rows_a = len(matrix_a)
        rows_per_proc = num_rows_a // size  # Minimum rows per process
        extras = num_rows_a % size  # Handle cases where rows are not evenly divisible
        counts = [rows_per_proc + (1 if i < extras else 0) for i in range(size)]  # Number of rows for each process
        displs = [sum(counts[:i]) for i in range(size)]  # Displacement (starting row) for each process
    else:
        # Non-root processes initialize variables as None (will be received via broadcast)
        matrix_b = None
        counts = None
        displs = None

    # Start timer for data distribution phase
    dist_start = MPI.Wtime()

    # Broadcast matrix_b to all processes so each has a full copy
    matrix_b = comm.bcast(matrix_b if rank == 0 else None, root=0)

    # Broadcast row counts and displacements to all processes
    counts = comm.bcast(counts if rank == 0 else None, root=0)
    displs = comm.bcast(displs if rank == 0 else None, root=0)
    local_num_rows = counts[rank]  # Number of rows assigned to this process
    # Scatter the appropriate rows of matrix_a to each process
    local_matrix_a = comm.scatter([matrix_a[displs[i]:displs[i]+counts[i]] if rank == 0 else None for i in range(size)], root=0)

    # End timer for data distribution, start timer for computation
    dist_end = MPI.Wtime()
    comp_start = dist_end

    # Each process computes its assigned portion of the matrix multiplication
    local_product = multiply_matrices_partial(local_matrix_a, matrix_b)

    # End timer for computation, start timer for gathering results
    comp_end = MPI.Wtime()
    gather_start = comp_end

    # Gather all partial results at the root process
    gathered = comm.gather(local_product, root=0)

    # End timer for gathering phase
    gather_end = MPI.Wtime()

    # Print timing information for each process
    print(f"Rank {rank}: Data distribution time: {dist_end - dist_start:.6f} s, Computation time: {comp_end - comp_start:.6f} s, Gathering time: {gather_end - gather_start:.6f} s")

    if rank == 0:
        # Flatten the gathered results into a single result matrix
        result_matrix = [row for part in gathered for row in part]
        print("Distributed Result:")
      
        total_end = MPI.Wtime()
        print(f"Total parallel runtime: {total_end - total_start:.6f} s")
