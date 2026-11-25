CC = gcc
CFLAGS = -Wall -O2
OMP_FLAGS = -fopenmp
PTHREAD_FLAGS = -pthread

TARGETS = race critical atomic barrier semaphore compare_sync

all: $(TARGETS)

race: race.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< -o $@

critical: critical.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< -o $@

atomic: atomic.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< -o $@

barrier: barrier.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< -o $@

semaphore: semaphore.c
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) $< -o $@

compare_sync: compare_sync.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS) *.o
