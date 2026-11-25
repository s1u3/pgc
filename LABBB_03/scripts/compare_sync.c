// compare_sync.c - Full benchmark comparing four counter-update strategies across 1, 2, 4, 8 threads
#include <stdio.h>
#include <omp.h>

#define N 10000000L

long run_none(int nthreads) {
    long counter = 0;
    omp_set_num_threads(nthreads);
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        counter++;
    }
    return counter;
}

long run_critical(int nthreads) {
    long counter = 0;
    omp_set_num_threads(nthreads);
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        #pragma omp critical
        {
            counter++;
        }
    }
    return counter;
}

long run_atomic(int nthreads) {
    long counter = 0;
    omp_set_num_threads(nthreads);
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        #pragma omp atomic
        counter++;
    }
    return counter;
}

long run_lock(int nthreads) {
    long counter = 0;
    omp_lock_t lock;
    omp_init_lock(&lock);
    omp_set_num_threads(nthreads);
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        omp_set_lock(&lock);
        counter++;
        omp_unset_lock(&lock);
    }
    omp_destroy_lock(&lock);
    return counter;
}

void run_and_report(const char *name, long (*fn)(int), int nthreads) {
    double start = omp_get_wtime();
    long result = fn(nthreads);
    double elapsed = omp_get_wtime() - start;
    const char *correctness = (result == N) ? "Correct" : "INCORRECT";
    printf("%-10s | %7d | %12ld | %10s | %10.4f\n",
           name, nthreads, result, correctness, elapsed);
}

int main(void) {
    int thread_counts[] = {1, 2, 4, 8};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("===============================================================\n");
    printf("             SYNCHRONIZATION MECHANISM COMPARISON              \n");
    printf("===============================================================\n");
    printf("Iterations: %ld\n\n", N);
    printf("Mechanism  | Threads |  Final Count | Correctness |   Time(s)  \n");
    printf("---------------------------------------------------------------\n");

    for (int i = 0; i < num_configs; i++) {
        int t = thread_counts[i];
        run_and_report("None",     run_none,     t);
        run_and_report("Critical", run_critical, t);
        run_and_report("Atomic",   run_atomic,   t);
        run_and_report("Lock",     run_lock,     t);
        printf("---------------------------------------------------------------\n");
    }

    printf("\nExpected correct count = %ld\n", N);
    printf("Observation: 'None' is fastest but incorrect (race condition).\n");
    printf("'Critical' is correct but slowest (coarse-grained locking).\n");
    printf("'Atomic' and 'Lock' are correct; 'Atomic' scales better since it\n");
    printf("uses lightweight hardware instructions instead of OS-level locks.\n");
    return 0;
}
