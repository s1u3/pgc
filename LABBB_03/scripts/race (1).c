// race.c - Demonstrates a race condition (no synchronization)
#include <stdio.h>
#include <omp.h>

int main() {
    long counter = 0;
    long expected = 10000000;

    #pragma omp parallel for
    for (long i = 0; i < expected; i++) {
        counter++;  // Unsynchronized read-modify-write
    }

    printf("Race Condition -\n");
    printf("Expected Counter: %ld\n", expected);
    printf("Actual Counter:   %ld\n", counter);
    printf("Status:           %s\n", (counter == expected) ? "Correct" : "INCORRECT (Race Detected)");
    return 0;
}
