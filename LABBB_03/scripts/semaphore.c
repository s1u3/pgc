// semaphore.c - Uses POSIX semaphores (sem_t) for thread synchronization
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t semaphore;
int counter = 0;

void* work(void* a) {
    for (int i = 0; i < 100000; i++) {
        sem_wait(&semaphore);
        counter++;
        sem_post(&semaphore);
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    sem_init(&semaphore, 0, 1);

    printf("POSIX Semaphore -\n");
    pthread_create(&t1, NULL, work, NULL);
    pthread_create(&t2, NULL, work, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Expected Counter: %d\n", 200000);
    printf("Actual Counter:   %d\n", counter);
    printf("Status:           %s\n", (counter == 200000) ? "Correct" : "INCORRECT");

    sem_destroy(&semaphore);
    return 0;
}
