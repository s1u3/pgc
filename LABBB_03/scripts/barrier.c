// barrier.c - Demonstrates #pragma omp barrier phased synchronization
#include <stdio.h>
#include <omp.h>

int main() {
    printf("Barrier Synchronization -\n");
    #pragma omp parallel
    {
        printf("Thread %d: Before barrier\n", omp_get_thread_num());
        #pragma omp barrier
        printf("Thread %d: After barrier\n", omp_get_thread_num());
    }
    return 0;
}
