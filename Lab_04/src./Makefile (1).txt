CC = gcc
CFLAGS = -O2 -Wall -fopenmp
PYTHON = python3

TARGETS = false_sharing numa_sweep compare_numa

all: $(TARGETS)

false_sharing: false_sharing.c
	$(CC) $(CFLAGS) false_sharing.c -o false_sharing

numa_sweep: numa_sweep.c
	$(CC) $(CFLAGS) numa_sweep.c -o numa_sweep

compare_numa: compare_numa.c
	$(CC) $(CFLAGS) compare_numa.c -o compare_numa

sim:
	$(PYTHON) coherence_sim.py 1000

run: all
	@echo "=== Running Cache Coherence Protocol Simulator (1000 Iterations) ==="
	$(PYTHON) coherence_sim.py 1000
	@echo "\n=== Running Master False Sharing & NUMA Benchmark ==="
	./compare_numa

clean:
	rm -f $(TARGETS) *.o *.out *.exe
