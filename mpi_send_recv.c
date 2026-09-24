#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int A;
    char hostname[256];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(hostname, sizeof(hostname));
    printf("Rank %d is running on %s\n", rank, hostname);

    if (size < 2)
    {
        if (rank == 0)
            printf("This program requires at least 2 MPI processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        A = 10;
        printf("Rank 0 on %s: Sending A = %d to Rank 1\n", hostname, A);
        MPI_Send(&A, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        MPI_Recv(&A, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 on %s: Received A = %d from Rank 0\n", hostname, A);
    }

    MPI_Finalize();
    return 0;
}
