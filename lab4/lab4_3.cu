#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define ROWS 100
#define COLS 200
#define BASE_TYPE float

inline int toMultiple(int a, int b) {
    return (a + b - 1) / b * b;
}

__global__ void matrixAdd(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        int idx = y * cols + x;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int rows = toMultiple(ROWS, BLOCK_SIZE);
    int cols = toMultiple(COLS, BLOCK_SIZE);
    size_t size = rows * cols * sizeof(BASE_TYPE);

    BASE_TYPE *h_A = (BASE_TYPE*)malloc(size);
    BASE_TYPE *h_B = (BASE_TYPE*)malloc(size);
    BASE_TYPE *h_C = (BASE_TYPE*)malloc(size);

    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = (BASE_TYPE)rand() / RAND_MAX;
        h_B[i] = (BASE_TYPE)rand() / RAND_MAX;
    }

    BASE_TYPE *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Matrix addition (%dx%d) time: %.3f ms\n", rows, cols, time_ms);

    // Verify
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < rows * cols && ok; ++i)
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) ok = false;
    printf("Verification %s\n", ok ? "PASSED" : "FAILED");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
