/*
 * numa_sweep.c - NUMA Memory Sweep Benchmark
 * Course: Parallel and GPU Computing (26ECAC304) | Experiment 4
 *
 * Measures sequential cache-line memory read access latency and throughput
 * over a large memory buffer (default 512 MB) across 4 traversal passes.
 * Pages are mapped using the First-Touch allocation policy.
 *
 * Compilation: gcc -O2 numa_sweep.c -fopenmp -o numa_sweep
 * Usage:
 *   OMP_NUM_THREADS=2 numactl --cpunodebind=0 --membind=0 ./numa_sweep 512   # Local memory
 *   OMP_NUM_THREADS=2 numactl --cpunodebind=0 --membind=1 ./numa_sweep 512   # Remote memory
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? atol(argv[1]) : 512;
    size_t n = mb << 20;  /* Convert MB to bytes */
    
    char *buf = (char *)malloc(n);
    if (!buf) {
        perror("malloc failed");
        return 1;
    }
    
    /* First touch policy: pages are placed on the NUMA node bound by numactl */
    memset(buf, 1, n);

    long sum = 0;
    int nt = omp_get_max_threads();
    double t0 = omp_get_wtime();
    
    /* 4 passes over the memory buffer, reading 1 byte per 64-byte cache line */
    for (int pass = 0; pass < 4; pass++) {
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t i = 0; i < n; i += 64) {
            sum += buf[i];
        }
    }
    double elapsed = omp_get_wtime() - t0;
    double gb_processed = ((double)n * 4.0) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = gb_processed / elapsed;

    printf("Buffer: %4zu MB | Threads: %2d | Checksum: %12ld | Time: %8.4f s | Throughput: %6.2f GB/s\n",
           mb, nt, sum, elapsed, bandwidth);

    free(buf);
    return 0;
}
