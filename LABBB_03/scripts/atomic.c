// atomic.c - Uses #pragma omp atomic for lightweight hardware-level synchronization
#include <stdio.h>
#include <omp.h>

int main() {
    long counter = 0;
    long expected = 10000000;

    #pragma omp parallel for
    for (long i = 0; i < expected; i++) {
        #pragma omp atomic
        counter++;
    }

    printf("Atomic Operation -\n");
    printf("Expected Counter: %ld\n", expected);
    printf("Actual Counter:   %ld\n", counter);
    printf("Status:           %s\n", (counter == expected) ? "Correct" : "INCORRECT");
    return 0;
}
