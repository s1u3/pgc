#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000000

int main()
{
    double *data;
    double sum = 0.0;
    double average = 0.0;
    double start, end;

    data = (double *)malloc(N * sizeof(double));
    if (data == NULL)
    {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize dataset with 1.0
    for (long long i = 0; i < N; i++)
    {
        data[i] = 1.0;
    }

    printf("Dataset size = %d\n", N);
    printf("Maximum OpenMP threads = %d\n\n", omp_get_max_threads());
    printf("Parallel Sum and Average\n");
    printf("------------------------\n");

    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; i++)
    {
        sum += data[i];
    }

    end = omp_get_wtime();

    average = sum / N;

    printf("Sum     = %.2f\n", sum);
    printf("Average = %.2f\n", average);
    printf("Time    = %f seconds\n", end - start);

    free(data);

    return 0;
}
