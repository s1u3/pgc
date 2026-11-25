/*
 * false_sharing.c - False Sharing Benchmark (Adjacent vs Padded Layout)
 * Course: Parallel and GPU Computing (26ECAC304) | Experiment 4
 *
 * Demonstrates the severe performance penalty of false sharing when multiple
 * threads update variables located on the same 64-byte cache line versus
 * padded variables separated across independent cache lines.
 *
 * Compilation: gcc -O2 false_sharing.c -fopenmp -o false_sharing
 * Usage:
 *   OMP_NUM_THREADS=2 ./false_sharing adjacent
 *   OMP_NUM_THREADS=2 ./false_sharing padded
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 100000000L

int main(int argc, char **argv) {
    /* stride = 1 means adjacent (all threads share the same cache line)
     * stride = 8 means padded (8 longs * 8 bytes = 64 bytes, one line per thread) */
    int stride = (argc > 1 && argv[1][0] == 'p') ? 8 : 1;
    int nt = omp_get_max_threads();
    
    size_t bytes = ((size_t)nt * stride * sizeof(long) + 63) / 64 * 64;
    long *cnt = (long *)aligned_alloc(64, bytes);
    if (!cnt) {
        perror("aligned_alloc failed");
        return 1;
    }
    memset(cnt, 0, bytes);

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        volatile long *p = &cnt[tid * stride];
        long iters = N / nt;
        for (long i = 0; i < iters; i++) {
            (*p)++;      /* Read-modify-write own thread counter */
        }
    }
    double t = omp_get_wtime() - t0;

    long total = 0;
    for (int i = 0; i < nt; i++) {
        total += cnt[i * stride];
    }

    printf("Layout: %-25s | Threads: %2d | Final Count: %10ld | Time: %8.4f s\n",
           stride == 1 ? "Adjacent (False Sharing)" : "Padded (No False Sharing)",
           nt, total, t);

    free(cnt);
    return 0;
}
