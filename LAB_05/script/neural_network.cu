// Experiment 5: Neural Network Layers in CUDA
// Implements matrix multiplication + bias, ReLU, and sigmoid.
// Compares a CPU forward pass with a custom CUDA forward pass.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int BATCH       = 256;
constexpr int INPUT_SIZE  = 128;
constexpr int HIDDEN_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;
constexpr int REPEATS     = 20;

// ============================================================================
// Phase 2 of 8: The three GPU kernels
// ============================================================================

// Each CUDA thread calculates one output matrix element and adds its bias.
__global__ void matrixMulBiasKernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}

// Each thread applies ReLU to one element.
__global__ void reluKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}

// Each thread applies sigmoid to one element.
__global__ void sigmoidKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = 1.0f / (1.0f + expf(-values[idx]));
    }
}

// ============================================================================
// Phase 3 of 8: The CPU version of the network
// ============================================================================

void cpuMatMulBias(const std::vector<float>& A,
                   const std::vector<float>& B,
                   const std::vector<float>& bias,
                   std::vector<float>& C,
                   int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum + bias[col];
        }
    }
}

void cpuReLU(std::vector<float>& values)
{
    for (float& x : values) x = std::max(0.0f, x);
}

void cpuSigmoid(std::vector<float>& values)
{
    for (float& x : values) x = 1.0f / (1.0f + std::exp(-x));
}

void cpuForwardPass(const std::vector<float>& X,
                    const std::vector<float>& W1,
                    const std::vector<float>& b1,
                    const std::vector<float>& W2,
                    const std::vector<float>& b2,
                    std::vector<float>& output)
{
    std::vector<float> z1(BATCH * HIDDEN_SIZE);
    cpuMatMulBias(X, W1, b1, z1, BATCH, HIDDEN_SIZE, INPUT_SIZE);
    cpuReLU(z1);

    std::vector<float> z2(BATCH * OUTPUT_SIZE);
    cpuMatMulBias(z1, W2, b2, z2, BATCH, OUTPUT_SIZE, HIDDEN_SIZE);
    cpuSigmoid(z2);

    output = z2;
}

// ============================================================================
// Phase 4 of 8: Launching the GPU forward pass
// ============================================================================

void launchGpuForwardPass(const float* dX, const float* dW1, const float* db1,
                         float* dZ1, const float* dW2, const float* db2,
                         float* dZ2)
{
    dim3 block(16, 16);
    dim3 grid1((HIDDEN_SIZE + block.x - 1) / block.x,
               (BATCH + block.y - 1) / block.y);
    matrixMulBiasKernel<<<grid1, block>>>(
        dX, dW1, db1, dZ1, BATCH, HIDDEN_SIZE, INPUT_SIZE);

    int hiddenCount = BATCH * HIDDEN_SIZE;
    int threads = 256;
    int blocks = (hiddenCount + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(dZ1, hiddenCount);

    dim3 grid2((OUTPUT_SIZE + block.x - 1) / block.x,
               (BATCH + block.y - 1) / block.y);
    matrixMulBiasKernel<<<grid2, block>>>(
        dZ1, dW2, db2, dZ2, BATCH, OUTPUT_SIZE, HIDDEN_SIZE);

    int outputCount = BATCH * OUTPUT_SIZE;
    int outputBlocks = (outputCount + threads - 1) / threads;
    sigmoidKernel<<<outputBlocks, threads>>>(dZ2, outputCount);

    CUDA_CHECK(cudaGetLastError());
}

bool nearlyEqual(float a, float b, float tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}

// ============================================================================
// Phase 5 of 8: The three kernel tests
// ============================================================================

bool testMatrixKernel()
{
    // A is 2x3, B is 3x2. Expected A*B = [22,28;49,64].
    // Add bias [1,-1] => [23,27;50,63].
    std::vector<float> hA = {1, 2, 3, 4, 5, 6};
    std::vector<float> hB = {1, 2, 3, 4, 5, 6};
    std::vector<float> hBias = {1, -1};
    std::vector<float> hC(4, 0.0f);

    float *dA = nullptr, *dB = nullptr, *dBias = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, hBias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), hBias.size() * sizeof(float), cudaMemcpyHostToDevice));

    matrixMulBiasKernel<<<dim3(1, 1), dim3(16, 16)>>>(
        dA, dB, dBias, dC, 2, 2, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = nearlyEqual(hC[0], 23) && nearlyEqual(hC[1], 27) &&
              nearlyEqual(hC[2], 50) && nearlyEqual(hC[3], 63);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dBias);
    cudaFree(dC);
    return ok;
}

bool testReLUKernel()
{
    std::vector<float> h = {-2.0f, 0.0f, 3.0f, -4.0f, 5.0f};
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (static_cast<int>(h.size()) + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d, static_cast<int>(h.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = nearlyEqual(h[0], 0) && nearlyEqual(h[1], 0) &&
              nearlyEqual(h[2], 3) && nearlyEqual(h[3], 0) &&
              nearlyEqual(h[4], 5);

    cudaFree(d);
    return ok;
}

bool testSigmoidKernel()
{
    std::vector<float> h = {-2.0f, 0.0f, 2.0f};
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (static_cast<int>(h.size()) + threads - 1) / threads;
    sigmoidKernel<<<blocks, threads>>>(d, static_cast<int>(h.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = nearlyEqual(h[0], 1.0f / (1.0f + std::exp(2.0f))) &&
              nearlyEqual(h[1], 0.5f) &&
              nearlyEqual(h[2], 1.0f / (1.0f + std::exp(-2.0f)));

    cudaFree(d);
    return ok;
}

// ============================================================================
// Phase 6, 7, 8: main()
// ============================================================================

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable NVIDIA GPU was detected." << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA\n";
    std::cout << "GPU detected: " << prop.name << "\n\n";

    std::cout << "Testing individual CUDA kernels...\n";
    bool matrixOK  = testMatrixKernel();
    bool reluOK    = testReLUKernel();
    bool sigmoidOK = testSigmoidKernel();

    std::cout << "Matrix multiplication + bias: " << (matrixOK  ? "PASSED" : "FAILED") << "\n";
    std::cout << "ReLU: "                         << (reluOK    ? "PASSED" : "FAILED") << "\n";
    std::cout << "Sigmoid: "                      << (sigmoidOK ? "PASSED" : "FAILED") << "\n";

    if (!(matrixOK && reluOK && sigmoidOK)) {
        std::cerr << "A kernel test failed. Stop and debug before continuing.\n";
        return EXIT_FAILURE;
    }

    // Deterministic input and weights: both CPU and GPU receive identical values.
    std::vector<float> X(BATCH * INPUT_SIZE);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> b1(HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> b2(OUTPUT_SIZE);

    for (size_t i = 0; i < X.size(); ++i)
        X[i] = 0.5f * std::sin(static_cast<float>(i) * 0.013f);
    for (size_t i = 0; i < W1.size(); ++i)
        W1[i] = 0.08f * std::cos(static_cast<float>(i) * 0.017f);
    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.01f * static_cast<float>(i % 7);
    for (size_t i = 0; i < W2.size(); ++i)
        W2[i] = 0.10f * std::sin(static_cast<float>(i) * 0.019f);
    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.02f * static_cast<float>(i % 5);

    std::vector<float> cpuOutput(BATCH * OUTPUT_SIZE);
    auto cpuStart = std::chrono::steady_clock::now();
    for (int r = 0; r < REPEATS; ++r) {
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);
    }
    auto cpuStop = std::chrono::steady_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count() / REPEATS;

    // GPU allocations
    float *dX = nullptr, *dW1 = nullptr, *db1 = nullptr, *dZ1 = nullptr;
    float *dW2 = nullptr, *db2 = nullptr, *dZ2 = nullptr;

    CUDA_CHECK(cudaMalloc(&dX,  X.size()  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ1, BATCH * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ2, BATCH * OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dX,  X.data(),  X.size()  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up: first launches can include one-time initialization overhead.
    launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel-only timing: input/weights already reside on the GPU.
    cudaEvent_t eventStart, eventStop;
    CUDA_CHECK(cudaEventCreate(&eventStart));
    CUDA_CHECK(cudaEventCreate(&eventStop));

    CUDA_CHECK(cudaEventRecord(eventStart));
    for (int r = 0; r < REPEATS; ++r) {
        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    }
    CUDA_CHECK(cudaEventRecord(eventStop));
    CUDA_CHECK(cudaEventSynchronize(eventStop));

    float gpuKernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuKernelMs, eventStart, eventStop));
    gpuKernelMs /= REPEATS;

    // End-to-end timing: includes input/weight/bias transfers and output transfer.
    std::vector<float> gpuOutput(BATCH * OUTPUT_SIZE);
    auto gpuTotalStart = std::chrono::steady_clock::now();
    for (int r = 0; r < REPEATS; ++r) {
        CUDA_CHECK(cudaMemcpy(dX,  X.data(),  X.size()  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(gpuOutput.data(), dZ2, gpuOutput.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    auto gpuTotalStop = std::chrono::steady_clock::now();
    double gpuEndToEndMs = std::chrono::duration<double, std::milli>(gpuTotalStop - gpuTotalStart).count() / REPEATS;

    // Compare results
    float maxDifference = 0.0f;
    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        maxDifference = std::max(maxDifference, std::fabs(cpuOutput[i] - gpuOutput[i]));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nForward-pass dimensions:\n";
    std::cout << "Input: "        << BATCH << " x " << INPUT_SIZE  << "\n";
    std::cout << "Hidden layer: " << BATCH << " x " << HIDDEN_SIZE << "\n";
    std::cout << "Output: "       << BATCH << " x " << OUTPUT_SIZE << "\n";

    std::cout << "\nAverage CPU compute time: "     << cpuMs          << " ms\n";
    std::cout << "Average GPU kernel-only time: "  << gpuKernelMs    << " ms\n";
    std::cout << "Average GPU end-to-end time: "    << gpuEndToEndMs  << " ms\n";

    if (gpuKernelMs > 0.0f)
        std::cout << "Kernel-only speedup: " << cpuMs / gpuKernelMs << "x\n";
    if (gpuEndToEndMs > 0.0)
        std::cout << "End-to-end speedup: "   << cpuMs / gpuEndToEndMs << "x\n";

    std::cout << "\nMaximum absolute output difference: " << maxDifference << "\n";
    std::cout << "CPU and GPU outputs: "
              << (maxDifference < 1e-4f ? "MATCH (within tolerance)" : "DO NOT MATCH")
              << "\n";

    std::cout << "\nFirst 10 output values (GPU):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << gpuOutput[i] << (i == 9 ? '\n' : ' ');
    }

    CUDA_CHECK(cudaEventDestroy(eventStart));
    CUDA_CHECK(cudaEventDestroy(eventStop));
    cudaFree(dX);
    cudaFree(dW1);
    cudaFree(db1);
    cudaFree(dZ1);
    cudaFree(dW2);
    cudaFree(db2);
    cudaFree(dZ2);

    return (maxDifference < 1e-4f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
