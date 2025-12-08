#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // для fabs()

#define BLOCK_SIZE 16
// Тип, который будут иметь элементы матриц
#define BASE_TYPE double

// Функция перемножения матриц
__global__ void matrixMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C,
                           int Acols, int Bcols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;  // строка в матрице C (и A)
    int j = blockDim.x * blockIdx.x + threadIdx.x;  // столбец в матрице C (и B)

    // Проверка выхода за пределы (важно, если размеры не кратны BLOCK_SIZE)
    if (i >= gridDim.y * blockDim.y || j >= gridDim.x * blockDim.x) {
        return;
    }

    BASE_TYPE sum = 0.0;
    for (int k = 0; k < Acols; ++k) {
        sum += A[i * Acols + k] * B[k * Bcols + j];
    }

    C[i * Bcols + j] = sum;
}

// Округление вверх до кратного BLOCK_SIZE
int toMultiple(int a, int b) {
    int mod = a % b;
    if (mod != 0) {
        return a + (b - mod);
    }
    return a;
}

int main() {
    // События для замера времени выполнения ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Исходные размеры матриц
    int Arows = 100;
    int Acols = 200;
    int Brows = Acols;  
    int Bcols = 150;

    // Приведение к кратности BLOCK_SIZE
    Arows = toMultiple(Arows, BLOCK_SIZE);
    printf("Arows = %d\n", Arows);

    Acols = toMultiple(Acols, BLOCK_SIZE);
    printf("Acols = %d\n", Acols);

    Brows = toMultiple(Brows, BLOCK_SIZE);
    printf("Brows = %d\n", Brows);

    Bcols = toMultiple(Bcols, BLOCK_SIZE);
    printf("Bcols = %d\n", Bcols);

    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);

    // Выделение памяти на хосте
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE *)malloc(Csize);

    // Инициализация матриц случайными значениями [0, 1)
    for (int i = 0; i < Arows * Acols; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    for (int i = 0; i < Brows * Bcols; ++i) {
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    }

    // Выделение и копирование данных на устройство
    BASE_TYPE *d_A = nullptr;
    cudaMalloc((void **)&d_A, Asize);

    BASE_TYPE *d_B = nullptr;
    cudaMalloc((void **)&d_B, Bsize);

    BASE_TYPE *d_C = nullptr;
    cudaMalloc((void **)&d_C, Csize);

    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);

    // Конфигурация запуска ядра
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);

    // Запуск ядра и замер времени
    cudaEventRecord(start, 0);
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Acols, Bcols);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float KernelTime;
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("KernelTime: %.2f milliseconds\n", KernelTime);

    // Копирование результата обратно на хост
    cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);

    // Проверка корректности результата
    printf("Test STARTED\n");
    bool passed = true;
    for (int i = 0; i < Arows; ++i) {
        for (int j = 0; j < Bcols; ++j) {
            BASE_TYPE sum = 0.0;
            for (int k = 0; k < Acols; ++k) {
                sum += h_A[i * Acols + k] * h_B[k * Bcols + j];
            }
            BASE_TYPE diff = fabs(h_C[i * Bcols + j] - sum);
            if (diff > 1e-3) {
                fprintf(stderr, "Result verification failed at element [%d, %d]! "
                                "diff = %.6f (expected %.6f, got %.6f)\n",
                        i, j, diff, sum, h_C[i * Bcols + j]);
                passed = false;
                // Не выходим сразу — можно оставить для полного лога или выйти сразу:
                // exit(EXIT_FAILURE);
            }
        }
    }

    if (passed) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
        exit(EXIT_FAILURE);
    }

    // Очистка ресурсов
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
