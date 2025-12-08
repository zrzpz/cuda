#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Ядро: заполняет матрицу значениями на GPU
__global__ void createMatrix(int *A, const int n)
{
    A[threadIdx.y * n + threadIdx.x] = 10 * threadIdx.y + threadIdx.x;
}

int main()
{
    const int n = 10;
    size_t size = n * n * sizeof(int);

    // Выделяем память для матрицы на CPU
    int *h_A = (int *)malloc(size);
    if (!h_A) {
        fprintf(stderr, "Error: malloc failed for h_A\n");
        return EXIT_FAILURE;
    }

    // Инициализируем матрицу A на CPU
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[j * n + i] = 10 * j + i;
        }
    }

    // Указатель на память GPU
    int *d_B = NULL;

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));

    // Определение размеров блока и сетки
    dim3 threadsPerBlock(10, 10);
    dim3 blocksPerGrid(1, 1, 1);

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Запуск замера и выполнение ядра
    CUDA_CHECK(cudaEventRecord(start, 0));
    createMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_B, n);

    // Проверка на ошибки при запуске ядра (асинхронная ошибка!)
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Получаем время выполнения ядра
    float kernelTimeMs;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));

    // Освобождаем события
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Выделяем память для результата на CPU
    int *h_B = (int *)malloc(size);
    if (!h_B) {
        fprintf(stderr, "Error: malloc failed for h_B\n");
        cudaFree(d_B);
        free(h_A);
        return EXIT_FAILURE;
    }

    // Копируем данные с GPU на CPU
    CUDA_CHECK(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));

    // Проверяем совпадение
    bool match = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (h_A[j * n + i] != h_B[j * n + i]) {
                printf("Mismatch at [%d][%d]: h_A[%d] = %d, h_B[%d] = %d\n",
                       j, i, j * n + i, h_A[j * n + i], j * n + i, h_B[j * n + i]);
                match = false;
            }
        }
    }

    if (match) {
        printf("Matrices match!\n");
    }

    printf("Kernel execution time: %.6f ms\n", kernelTimeMs);

    // Освобождение памяти
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
