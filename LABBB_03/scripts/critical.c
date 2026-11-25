// critical.c - Uses #pragma omp critical for mutual exclusion
#include <stdio.h>
#include <omp.h>

int main() {
    long counter = 0;
    long expected = 10000000;

    #pragma omp parallel for
    for (long i = 0; i < expected; i++) {
        #pragma omp critical
        {
            counter++;
        }
    }

    printf("Critical Section -\n");
    printf("Expected Counter: %ld\n", expected);
    printf("Actual Counter:   %ld\n", counter);
    printf("Status:           %s\n", (counter == expected) ? "Correct" : "INCORRECT");
    return 0;
}
