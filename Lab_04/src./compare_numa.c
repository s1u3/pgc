/*
 * compare_numa.c - Comprehensive False Sharing & NUMA Memory Access Benchmark
 * Course: Parallel and GPU Computing (26ECAC304) | Experiment 4
 *
 * Table 1: Evaluates two-counter adjacent vs padded layout across 1, 2, 4, 8 threads.
 * Table 2: Evaluates memory sweep of a 512 MB buffer across 1, 2, 4 threads.
 *
 * Compilation: gcc -O2 compare_numa.c -fopenmp -o compare_numa
 * Usage:
 *   ./compare_numa                                            # Default placement
 *   numactl --cpunodebind=0 --membind=0 ./compare_numa        # Local memory
 *   numactl --cpunodebind=0 --membind=1 ./compare_numa        # Remote memory
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 100000000L
#define MB 512UL

double run_counters(int nt, int stride, long *total) {
    size_t bytes = ((size_t)nt * stride * sizeof(long) + 63) / 64 * 64;
    long *cnt = (long *)aligned_alloc(64, bytes);
    if (!cnt) {
        perror("aligned_alloc failed");
        exit(1);
    }
    memset(cnt, 0, bytes);
    omp_set_num_threads(nt);
    
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    {
        volatile long *p = &cnt[omp_get_thread_num() * stride];
        long iters = N / nt;
        for (long i = 0; i < iters; i++) {
            (*p)++;
        }
    }
    double t = omp_get_wtime() - t0;
    
    *total = 0;
    for (int i = 0; i < nt; i++) {
        *total += cnt[i * stride];
    }
    free(cnt);
    return t;
}

double run_sweep(char *buf, size_t n, int nt) {
    long sum = 0;
    omp_set_num_threads(nt);
    double t0 = omp_get_wtime();
    for (int pass = 0; pass < 4; pass++) {
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t i = 0; i < n; i += 64) {
            sum += buf[i];
        }
    }
    double t = omp_get_wtime() - t0;
    if (sum == 0) printf("Warning: unexpected checksum zero\n");
    return t;
}

int main(void) {
    int tc[] = {1, 2, 4, 8};
    int num_tc = sizeof(tc) / sizeof(tc[0]);

    printf("=========================================================================\n");
    printf("TABLE 1: FALSE SHARING BENCHMARK (%ld Total Increments)\n", N);
    printf("=========================================================================\n");
    printf("%-10s | %7s | %11s | %-11s | %10s\n", "Layout", "Threads", "Final Count", "Correctness", "Time (s)");
    printf("-------------------------------------------------------------------------\n");
    
    for (int i = 0; i < num_tc; i++) {
        for (int s = 1; s <= 8; s *= 8) {
            long total;
            double t = run_counters(tc[i], s, &total);
            const char *layout_str = (s == 1) ? "adjacent" : "padded";
            const char *status_str = (total == N) ? "Correct" : "INCORRECT";
            printf("%-10s | %7d | %11ld | %-11s | %10.4f\n",
                   layout_str, tc[i], total, status_str, t);
        }
        printf("-------------------------------------------------------------------------\n");
    }

    size_t n = MB << 20;
    char *buf = (char *)malloc(n);
    if (!buf) {
        perror("malloc failed");
        return 1;
    }
    memset(buf, 1, n);  /* First touch policy determines physical NUMA page binding */

    printf("\n=========================================================================\n");
    printf("TABLE 2: NUMA MEMORY SWEEP OF %lu MB (4 Passes)\n", MB);
    printf("=========================================================================\n");
    printf("%7s | %10s | %16s\n", "Threads", "Time (s)", "Throughput (GB/s)");
    printf("-------------------------------------------------------------------------\n");
    
    double total_gb = ((double)n * 4.0) / (1024.0 * 1024.0 * 1024.0);
    for (int i = 0; i < 3; i++) {
        double t = run_sweep(buf, n, tc[i]);
        double bw = total_gb / t;
        printf("%7d | %10.4f | %16.2f\n", tc[i], t, bw);
    }
    printf("-------------------------------------------------------------------------\n");
    
    free(buf);
    return 0;
}
