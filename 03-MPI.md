# EXP-03 --- MPI Using Matrix Multiplication

> **Distributed-Memory Matrix Multiplication Using Open MPI Across Four
> Ubuntu Virtual Machines**

![MPI](https://img.shields.io/badge/MPI-Open%20MPI-blue)
![Language](https://img.shields.io/badge/Language-C-orange)
![Platform](https://img.shields.io/badge/Platform-Ubuntu%2024.04-purple)
![VMs](https://img.shields.io/badge/Cluster-1%20Master%20%2B%203%20Workers-green)
![Matrix](https://img.shields.io/badge/Matrix-4000%C3%974000-red)

------------------------------------------------------------------------

## 1. Experiment Overview

This experiment implements **distributed-memory matrix multiplication
using MPI (Message Passing Interface)** across four Ubuntu virtual
machines.

The cluster consists of:

-   **1 Master VM**
-   **3 Worker VMs**
-   **4 MPI processes / ranks**
-   **4000 × 4000 matrices**
-   One MPI rank assigned to each VM

The experiment demonstrates both **point-to-point communication** using
`MPI_Send()` / `MPI_Recv()` and **collective communication** using
`MPI_Scatter()`, `MPI_Bcast()`, and `MPI_Gather()`.

### Objectives

1.  Configure a four-node MPI cluster.
2.  Assign unique hostnames to all VMs.
3.  Identify and verify node IP addresses.
4.  Verify network connectivity.
5.  Configure SSH and passwordless remote access.
6.  Install and verify Open MPI.
7.  Create an MPI hostfile.
8.  Verify distributed process launching.
9.  Implement MPI point-to-point communication.
10. Implement distributed 4000 × 4000 matrix multiplication.
11. Distribute Matrix A using `MPI_Scatter()`.
12. Broadcast Matrix B using `MPI_Bcast()`.
13. Perform local matrix multiplication on each rank.
14. Collect results using `MPI_Gather()`.
15. Measure execution time using `MPI_Wtime()`.
16. Verify the final matrix result.

------------------------------------------------------------------------

## 2. MPI Cluster Architecture

``` text
                         MPI Cluster
                              |
                    +---------+---------+
                    |       Master      |
                    |       Rank 0      |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
          Worker1          Worker2          Worker3
          Rank 1           Rank 2           Rank 3
```

### Node Mapping

  Node       Hostname      MPI Rank Role
  ---------- ----------- ---------- -------------------
  Master     `master`             0 Root / controller
  Worker 1   `worker1`            1 Worker
  Worker 2   `worker2`            2 Worker
  Worker 3   `worker3`            3 Worker

### Actual UTM Network Used

The experiment was performed using UTM virtual machines. The actual node
addresses used during the experiment were:

  Node       IP Address
  ---------- ----------------
  Master     `192.168.64.5`
  Worker 1   `192.168.64.8`
  Worker 2   `192.168.64.7`
  Worker 3   `192.168.64.6`

> The manual contains example VMware IP addresses. The addresses above
> represent the actual UTM setup used for this experiment.

------------------------------------------------------------------------

# 3. Software and Tools

-   Ubuntu 24.04 LTS
-   Open MPI 4.1.6
-   GCC
-   `mpicc`
-   `mpirun`
-   OpenSSH
-   `scp`
-   UTM virtual machines
-   C programming language

------------------------------------------------------------------------

# 4. Hostname Configuration

Each VM requires a unique hostname.

### Verify hostname

``` bash
hostname
```

### Configure hostname

On Master:

``` bash
sudo hostnamectl set-hostname master
```

On Worker 1:

``` bash
sudo hostnamectl set-hostname worker1
```

On Worker 2:

``` bash
sudo hostnamectl set-hostname worker2
```

On Worker 3:

``` bash
sudo hostnamectl set-hostname worker3
```

Verify again:

``` bash
hostname
```

------------------------------------------------------------------------

# 5. Identify IP Addresses

Run on every VM:

``` bash
hostname -I
```

The resulting addresses were used to establish communication between the
cluster nodes.

------------------------------------------------------------------------

# 6. Network Connectivity

From the Master, each Worker was tested using `ping`.

``` bash
ping -c 4 worker1
ping -c 4 worker2
ping -c 4 worker3
```

A successful connection should show:

``` text
4 packets transmitted, 4 received, 0% packet loss
```

------------------------------------------------------------------------

# 7. SSH Configuration

MPI uses SSH to launch processes on remote worker nodes.

## Install OpenSSH Server

Run on each Worker:

``` bash
sudo apt update
sudo apt install openssh-server -y
```

Enable SSH:

``` bash
sudo systemctl enable --now ssh
```

Verify:

``` bash
sudo systemctl status ssh
```

Expected:

``` text
Active: active (running)
```

------------------------------------------------------------------------

# 8. Test SSH Communication

From the Master:

``` bash
ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname
```

Expected:

``` text
worker1
worker2
worker3
```

------------------------------------------------------------------------

# 9. Configure Passwordless SSH

Generate an SSH key on the Master:

``` bash
ssh-keygen -t rsa
```

Copy the public key to each Worker:

``` bash
ssh-copy-id worker1@worker1
ssh-copy-id worker1@worker2
ssh-copy-id worker1@worker3
```

Test passwordless access:

``` bash
ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname
```

No password should be required after configuration.

------------------------------------------------------------------------

# 10. SSH Host Aliases

The Master uses the local username `master`, while the Worker machines
use the `worker1` account.

Edit the SSH configuration:

``` bash
nano ~/.ssh/config
```

Configuration:

``` text
Host worker1
    HostName worker1
    User worker1

Host worker2
    HostName worker2
    User worker1

Host worker3
    HostName worker3
    User worker1
```

Set secure permissions:

``` bash
chmod 600 ~/.ssh/config
```

Verify:

``` bash
ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname
```

------------------------------------------------------------------------

# 11. Install Open MPI

Install MPI on **all four nodes**:

``` bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev -y
```

Verify:

``` bash
mpirun --version
mpicc --version
```

Open MPI provides:

-   `mpirun` --- MPI process launcher
-   `mpicc` --- MPI-aware C compiler

------------------------------------------------------------------------

# 12. MPI Hostfile

Create the hostfile on the Master:

``` bash
nano hosts
```

Contents:

``` text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

Verify:

``` bash
cat hosts
```

The hostfile assigns one MPI process slot to each VM.

------------------------------------------------------------------------

# 13. Verify Distributed MPI Process Launch

Run:

``` bash
mpirun -np 4 --hostfile hosts hostname
```

A successful launch should produce all four hostnames:

``` text
master
worker1
worker2
worker3
```

The order may vary because MPI processes execute concurrently.

------------------------------------------------------------------------

# 14. MPI Send/Receive Demonstration

Before matrix multiplication, point-to-point communication was
demonstrated.

Rank 0 creates the value:

``` text
A = 10
```

and sends it to Rank 1.

Rank 1 receives the value.

## Source Code --- `mpi_send_recv.c`

``` c
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int A;
    char hostname[256];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(hostname, sizeof(hostname));

    printf("Rank %d is running on %s\n", rank, hostname);

    if (size < 2)
    {
        if (rank == 0)
            printf("This program requires at least 2 MPI processes.\n");

        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        A = 10;

        printf(
            "Rank 0 on %s: Sending A = %d to Rank 1\n",
            hostname,
            A
        );

        MPI_Send(
            &A,
            1,
            MPI_INT,
            1,
            0,
            MPI_COMM_WORLD
        );
    }
    else if (rank == 1)
    {
        MPI_Recv(
            &A,
            1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        printf(
            "Rank 1 on %s: Received A = %d from Rank 0\n",
            hostname,
            A
        );
    }

    MPI_Finalize();

    return 0;
}
```

------------------------------------------------------------------------

# 15. Compile MPI Send/Receive Program

Compile:

``` bash
mpicc mpi_send_recv.c -o mpi_send_recv
```

Verify:

``` bash
ls -l mpi_send_recv
```

------------------------------------------------------------------------

# 16. Copy Send/Receive Executable to Workers

``` bash
scp mpi_send_recv worker1:~/
scp mpi_send_recv worker2:~/
scp mpi_send_recv worker3:~/
```

Verify:

``` bash
ssh worker1 "ls -l ~/mpi_send_recv"
ssh worker2 "ls -l ~/mpi_send_recv"
ssh worker3 "ls -l ~/mpi_send_recv"
```

------------------------------------------------------------------------

# 17. Execute MPI Send/Receive

Run:

``` bash
env -u DISPLAY mpirun -np 4 --hostfile hosts sh -c '$HOME/mpi_send_recv'
```

A successful result demonstrates:

``` text
Rank 0 → MPI_Send → Rank 1
Rank 1 → MPI_Recv → receives A = 10
```

Example output:

``` text
Rank 0 is running on master
Rank 0 on master: Sending A = 10 to Rank 1
Rank 1 is running on worker1
Rank 2 is running on worker2
Rank 3 is running on worker3
Rank 1 on worker1: Received A = 10 from Rank 0
```

------------------------------------------------------------------------

# 18. Matrix Multiplication Problem

The main MPI application computes:

``` text
C = A × B
```

where:

``` text
A = 4000 × 4000
B = 4000 × 4000
C = 4000 × 4000
```

Every element of `A` and `B` is initialized to `1.0`.

Therefore:

``` text
C[i][j] = 1 + 1 + ... + 1
         = 4000
```

The expected verification value is:

``` text
C[0][0] = 4000.00
```

------------------------------------------------------------------------

# 19. Distribution of Matrix A

The 4000 rows of Matrix A are divided among four MPI ranks.

``` text
4000 rows / 4 processes = 1000 rows per process
```

    Rank Node       Rows
  ------ ---------- ------------
       0 Master     0--999
       1 Worker 1   1000--1999
       2 Worker 2   2000--2999
       3 Worker 3   3000--3999

`MPI_Scatter()` performs this distribution.

------------------------------------------------------------------------

# 20. Distribution of Matrix B

Every rank needs the complete Matrix B.

Therefore:

``` c
MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

broadcasts Matrix B from Rank 0 to all processes.

------------------------------------------------------------------------

# 21. Collection of Matrix C

Each rank computes its local result:

``` text
local_C
```

containing 1000 rows.

`MPI_Gather()` collects the local result buffers on Rank 0 and
reconstructs the complete Matrix C.

------------------------------------------------------------------------

# 22. Complete Matrix MPI Program

## `matrix_mpi.c`

``` c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define N 4000

int main(int argc, char *argv[])
{
    int rank, size;
    int i, j, k;
    int rows_per_process;
    char hostname[256];

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    double *local_A;
    double *local_C;

    double start, end;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(hostname, sizeof(hostname));

    if (N % size != 0)
    {
        if (rank == 0)
            printf(
                "Matrix size must be divisible by number of processes.\n"
            );

        MPI_Finalize();
        return 0;
    }

    rows_per_process = N / size;

    local_A = (double *)malloc(
        rows_per_process * N * sizeof(double)
    );

    local_C = (double *)malloc(
        rows_per_process * N * sizeof(double)
    );

    B = (double *)malloc(
        N * N * sizeof(double)
    );

    if (rank == 0)
    {
        A = (double *)malloc(
            N * N * sizeof(double)
        );

        C = (double *)malloc(
            N * N * sizeof(double)
        );

        printf(
            "Initializing %d x %d matrices...\n",
            N,
            N
        );

        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    start = MPI_Wtime();

    MPI_Scatter(
        A,
        rows_per_process * N,
        MPI_DOUBLE,
        local_A,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        B,
        N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    printf(
        "Rank %d on %s computing %d rows\n",
        rank,
        hostname,
        rows_per_process
    );

    for (i = 0; i < rows_per_process; i++)
    {
        for (j = 0; j < N; j++)
        {
            local_C[i * N + j] = 0.0;

            for (k = 0; k < N; k++)
            {
                local_C[i * N + j] +=
                    local_A[i * N + k] *
                    B[k * N + j];
            }
        }
    }

    MPI_Gather(
        local_C,
        rows_per_process * N,
        MPI_DOUBLE,
        C,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    end = MPI_Wtime();

    if (rank == 0)
    {
        printf("\nMPI Matrix Multiplication Completed\n");

        printf(
            "Matrix Size = %d x %d\n",
            N,
            N
        );

        printf(
            "Number of MPI Processes = %d\n",
            size
        );

        printf(
            "Execution Time = %f seconds\n",
            end - start
        );

        printf(
            "Verification C[0][0] = %.2f\n",
            C[0]
        );

        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();

    return 0;
}
```

------------------------------------------------------------------------

# 23. Matrix Program Compilation

On the Master:

``` bash
mpicc matrix_mpi.c -o matrix_mpi
```

Verify:

``` bash
ls -l matrix_mpi
```

------------------------------------------------------------------------

# 24. Copy Matrix Executable to Workers

``` bash
scp matrix_mpi worker1:~/
scp matrix_mpi worker2:~/
scp matrix_mpi worker3:~/
```

Verify:

``` bash
ssh worker1 "ls -l ~/matrix_mpi"
ssh worker2 "ls -l ~/matrix_mpi"
ssh worker3 "ls -l ~/matrix_mpi"
```

------------------------------------------------------------------------

# 25. Execute Distributed Matrix Multiplication

Run from the Master:

``` bash
env -u DISPLAY mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

If the environment generates GUI authorization messages, the cleaned
execution form can be used:

``` bash
env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY -u DBUS_SESSION_BUS_ADDRESS \
mpirun --mca plm_rsh_agent "ssh -x" \
-np 4 --hostfile hosts sh -c '$HOME/matrix_mpi' 2>/dev/null
```

The second form removes irrelevant GUI/session environment variables
from the MPI-launched processes and keeps the terminal output clean for
experimental evidence.

------------------------------------------------------------------------

# 26. Expected Final Output

``` text
Initializing 4000 x 4000 matrices...

Rank 0 on master computing 1000 rows
Rank 1 on worker1 computing 1000 rows
Rank 2 on worker2 computing 1000 rows
Rank 3 on worker3 computing 1000 rows

MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = 4
Execution Time = [measured time] seconds
Verification C[0][0] = 4000.00
```

The order of the rank messages can vary because all MPI processes
execute concurrently.

------------------------------------------------------------------------

# 27. MPI Execution Flow

``` text
                     Master
                     Rank 0
                       |
                       | mpirun
                       v
             +---------------------+
             | 4 MPI Processes     |
             +---------------------+
               |    |    |    |
               v    v    v    v
             R0    R1   R2   R3
           Master  W1   W2   W3
               |
               v
        Initialize A and B
               |
               v
         MPI_Scatter(A)
               |
       +-------+-------+-------+
       |       |       |       |
     1000    1000    1000    1000
     rows    rows    rows    rows
       |       |       |       |
       +-------+-------+-------+
               |
          MPI_Bcast(B)
               |
               v
       Local Matrix Multiply
               |
               v
          MPI_Gather()
               |
               v
        Complete Matrix C
            on Rank 0
               |
               v
       Verify C[0][0] = 4000
```

------------------------------------------------------------------------

# 28. MPI Functions Used

  MPI Function        Purpose
  ------------------- ------------------------------------
  `MPI_Init()`        Initializes MPI
  `MPI_Comm_rank()`   Obtains process rank
  `MPI_Comm_size()`   Obtains number of processes
  `MPI_Send()`        Sends a point-to-point message
  `MPI_Recv()`        Receives a point-to-point message
  `MPI_Barrier()`     Synchronizes all ranks
  `MPI_Wtime()`       Measures wall-clock execution time
  `MPI_Scatter()`     Distributes Matrix A chunks
  `MPI_Bcast()`       Broadcasts Matrix B
  `MPI_Gather()`      Collects local Matrix C chunks
  `MPI_Finalize()`    Terminates MPI

------------------------------------------------------------------------

# 29. Memory Distribution

The MPI application follows a distributed-memory model.

``` text
MASTER / Rank 0
----------------
A
B
C
local_A
local_C

WORKER 1 / Rank 1
-----------------
B
local_A
local_C

WORKER 2 / Rank 2
-----------------
B
local_A
local_C

WORKER 3 / Rank 3
-----------------
B
local_A
local_C
```

Each MPI process has its own memory space. Data is explicitly
transferred using MPI communication functions.

------------------------------------------------------------------------

# 30. Performance Measurement

Execution time is measured using:

``` c
start = MPI_Wtime();
```

and:

``` c
end = MPI_Wtime();
```

The elapsed time is:

``` c
end - start
```

The timed distributed phase includes:

-   Matrix A distribution
-   Matrix B broadcast
-   Local matrix computation
-   Result gathering
-   Synchronization

> Record the actual execution time produced by your run in the final
> report. Do not substitute a sample/manual execution time.

------------------------------------------------------------------------

# 31. Verification

Because every element of A and B is `1.0`:

``` text
C[i][j]
= Σ A[i][k] × B[k][j]
= 1 + 1 + ... + 1
= 4000
```

Therefore:

``` text
Expected C[0][0] = 4000.00
```

The experiment is considered successfully verified when the final output
reports:

``` text
Verification C[0][0] = 4000.00
```

------------------------------------------------------------------------

# 32. Complete Command Sequence

## Cluster and Network

``` bash
hostname
hostname -I

ping -c 4 worker1
ping -c 4 worker2
ping -c 4 worker3
```

## SSH

``` bash
sudo apt update
sudo apt install openssh-server -y
sudo systemctl enable --now ssh
sudo systemctl status ssh

ssh-keygen -t rsa

ssh-copy-id worker1@worker1
ssh-copy-id worker1@worker2
ssh-copy-id worker1@worker3

ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname
```

## MPI Installation

``` bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev -y

mpirun --version
mpicc --version
```

## Hostfile

``` bash
nano hosts
cat hosts
```

Contents:

``` text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

## Distributed Process Test

``` bash
mpirun -np 4 --hostfile hosts hostname
```

## MPI Send/Receive

``` bash
nano mpi_send_recv.c
mpicc mpi_send_recv.c -o mpi_send_recv

scp mpi_send_recv worker1:~/
scp mpi_send_recv worker2:~/
scp mpi_send_recv worker3:~/

ssh worker1 "ls -l ~/mpi_send_recv"
ssh worker2 "ls -l ~/mpi_send_recv"
ssh worker3 "ls -l ~/mpi_send_recv"

env -u DISPLAY mpirun -np 4 --hostfile hosts sh -c '$HOME/mpi_send_recv'
```

## Matrix MPI

``` bash
nano matrix_mpi.c
mpicc matrix_mpi.c -o matrix_mpi
ls -l matrix_mpi

scp matrix_mpi worker1:~/
scp matrix_mpi worker2:~/
scp matrix_mpi worker3:~/

ssh worker1 "ls -l ~/matrix_mpi"
ssh worker2 "ls -l ~/matrix_mpi"
ssh worker3 "ls -l ~/matrix_mpi"

env -u DISPLAY mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

------------------------------------------------------------------------

# 33. Experimental Evidence / Screenshots

Recommended screenshot sequence:

    No. Screenshot
  ----- ---------------------------------
     01 UTM four-VM cluster
     02 UTM VM configuration
     03 Master hostname and IP
     04 Worker hostname/IP verification
     05 Network ping verification
     06 SSH service active
     07 Passwordless SSH
     08 Open MPI installation/version
     09 MPI hostfile
     10 Four-node MPI launch
     11 MPI Send/Receive source
     12 MPI Send/Receive result
     13 Matrix MPI source
     14 Matrix MPI compilation
     15 Matrix executable on workers
     16 Final 4000 × 4000 MPI result

The most important evidence is:

1.  Four-node MPI process launch
2.  Successful `MPI_Send()` / `MPI_Recv()`
3.  All four ranks participating in matrix multiplication
4.  4000 × 4000 matrix size
5.  1000 rows computed by each rank
6.  Final execution time
7.  `C[0][0] = 4000.00`

------------------------------------------------------------------------

# 34. Result

The MPI cluster was successfully configured using one Master VM and
three Worker VMs.

The experiment demonstrated:

-   Network communication between nodes
-   SSH-based remote access
-   Passwordless SSH
-   Open MPI installation
-   Distributed process launching
-   Point-to-point MPI communication
-   `MPI_Send()` and `MPI_Recv()`
-   `MPI_Scatter()` for Matrix A
-   `MPI_Bcast()` for Matrix B
-   Distributed local computation
-   `MPI_Gather()` for Matrix C
-   MPI execution-time measurement
-   Final result verification

The 4000 × 4000 matrix multiplication was distributed across four MPI
ranks, with each rank processing **1000 rows**.

The expected verification value was:

``` text
C[0][0] = 4000.00
```

------------------------------------------------------------------------

# 35. Conclusion

The experiment successfully implemented **distributed-memory matrix
multiplication using MPI across four Ubuntu virtual machines**.

The point-to-point communication experiment demonstrated explicit data
transfer between independent MPI processes using `MPI_Send()` and
`MPI_Recv()`.

The matrix multiplication application demonstrated collective MPI
communication through `MPI_Scatter()`, `MPI_Bcast()`, and
`MPI_Gather()`. Matrix A was divided into equal row blocks, Matrix B was
broadcast to every rank, each process computed its assigned rows
independently, and the local results were gathered on Rank 0.

The final verification value of `C[0][0] = 4000.00` confirms the
correctness of the matrix multiplication.

------------------------------------------------------------------------

## 36. Key Takeaways

-   MPI follows a **distributed-memory programming model**.
-   Every MPI rank has its own process memory.
-   Data must be explicitly communicated between processes.
-   `MPI_Send()` / `MPI_Recv()` provide point-to-point communication.
-   `MPI_Scatter()` distributes data from the root.
-   `MPI_Bcast()` distributes the same data to all ranks.
-   `MPI_Gather()` collects distributed results.
-   `MPI_Wtime()` measures elapsed wall-clock time.
-   Four MPI processes can divide a 4000-row matrix into four 1000-row
    chunks.
-   Correctness is verified using the known result `C[0][0] = 4000.00`.

------------------------------------------------------------------------

## 37. Experiment Status

**Status: Completed successfully**

``` text
Cluster Configuration      ✓
Network Connectivity       ✓
SSH Configuration          ✓
Passwordless SSH           ✓
Open MPI Installation      ✓
Hostfile Configuration     ✓
4-Node MPI Launch          ✓
MPI Send/Receive            ✓
Matrix Distribution         ✓
Distributed Computation    ✓
Result Gathering            ✓
Execution Time Measurement ✓
Result Verification        ✓
```

------------------------------------------------------------------------

### Repository Structure

``` text
exp-03-mpi/
│
├── README.md
├── hosts
├── mpi_send_recv.c
├── matrix_mpi.c
│
└── screenshots/
    ├── SS01_UTM_4_VM_Cluster.png
    ├── SS02_UTM_VM_Configuration.png
    ├── SS03_Master_IP_Hostname.png
    ├── SS04_Worker1_IP_Hostname.png
    ├── SS05_Worker2_IP_Hostname.png
    ├── SS06_Worker3_IP_Hostname.png
    ├── SS07_Ping_All_Workers.png
    ├── SS08_SSH_Service_Active.png
    ├── SS09_Passwordless_SSH.png
    ├── SS10_OpenMPI_Installation.png
    ├── SS11_MPI_Hostfile.png
    ├── SS12_MPI_4_Node_Test.png
    ├── SS13_MPI_SendRecv_Source.png
    ├── SS14_MPI_SendRecv_Executable.png
    ├── SS15_MPI_SendRecv_Result.png
    ├── SS16_Matrix_MPI_Source.png
    ├── SS17_Matrix_MPI_Compilation.png
    ├── SS18_Matrix_Executable_On_Workers.png
    └── SS19_Final_MPI_Matrix_Result.png
```

------------------------------------------------------------------------

**Experiment:** EXP-03 --- MPI Using Matrix Multiplication\
**Implementation:** Open MPI\
**Language:** C\
**Cluster:** 1 Master + 3 Workers\
**Matrix:** 4000 × 4000\
**Processes:** 4
