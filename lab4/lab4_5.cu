#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16
#define BASE_TYPE double

// Функция вычисления числа, которое больше числа a и кратное числу b
int toMultiple(int a, int b) {
    int mod = a % b;
    if (mod != 0) {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

// Ядро: оптимизированное умножение матриц с использованием разделяемой памяти
__global__ void matrixMult(const BASE_TYPE* A, const BASE_TYPE* B, 
                           BASE_TYPE* C, int Acols, int Bcols) {
    // Индекс начала первой подматрицы A, которую обрабатывает блок
    int aBegin = Acols * BLOCK_SIZE * blockIdx.y;
    // Индекс конца подматрицы A, которую обрабатывает блок
    int aEnd = aBegin + Acols - 1;
    // Шаг для перебора подматриц A
    int aStep = BLOCK_SIZE;
    
    // Индекс начала первой подматрицы B, которую обрабатывает блок
    int bBegin = BLOCK_SIZE * blockIdx.x;
    // Шаг для перебора подматриц B
    int bStep = BLOCK_SIZE * Bcols;
    
    // Выделение разделяемой памяти для подматриц
    __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Переменная для вычисления элемента подматрицы
    BASE_TYPE sum = 0.0;
    
    // Проход по всем тайлам (подматрицам)
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        // Загрузка подматриц A и B из глобальной памяти в разделяемую
        as[threadIdx.y][threadIdx.x] = A[ia + Acols * threadIdx.y + threadIdx.x];
        bs[threadIdx.y][threadIdx.x] = B[ib + Bcols * threadIdx.y + threadIdx.x];
        
        // Синхронизация нитей
        __syncthreads();
        
        // Перемножение двух подматриц
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        }
        
        // Синхронизация нитей перед загрузкой следующих тайлов
        __syncthreads();
    }
    
    // Индекс результирующего элемента в глобальной памяти
    int ind = Bcols * (BLOCK_SIZE * blockIdx.y + threadIdx.y) + 
              BLOCK_SIZE * blockIdx.x + threadIdx.x;
    
    // Запись результата в глобальную память
    C[ind] = sum;
}

int main() {
    // Объекты событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Количество строк и столбцов матриц
    int Arows = 100, Acols = 200;
    int Brows = Acols, Bcols = 150;
    
    // Приведение размеров кратными BLOCK_SIZE
    Arows = toMultiple(Arows, BLOCK_SIZE);
    Acols = toMultiple(Acols, BLOCK_SIZE);
    Brows = toMultiple(Brows, BLOCK_SIZE);
    Bcols = toMultiple(Bcols, BLOCK_SIZE);
    
    printf("Matrix multiplication: A(%d,%d) * B(%d,%d) = C(%d,%d)\n", 
           Arows, Acols, Brows, Bcols, Arows, Bcols);
    
    // Размеры в байтах
    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);
    
    // Выделение памяти на хосте
    BASE_TYPE* h_A = (BASE_TYPE*)malloc(Asize);
    BASE_TYPE* h_B = (BASE_TYPE*)malloc(Bsize);
    BASE_TYPE* h_C = (BASE_TYPE*)malloc(Csize);
    
    // Инициализация матриц случайными числами
    for (int i = 0; i < Arows * Acols; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    for (int i = 0; i < Brows * Bcols; ++i) {
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    
    // Выделение памяти на девайсе
    BASE_TYPE* d_A = NULL;
    BASE_TYPE* d_B = NULL;
    BASE_TYPE* d_C = NULL;
    cudaMalloc((void**)&d_A, Asize);
    cudaMalloc((void**)&d_B, Bsize);
    cudaMalloc((void**)&d_C, Csize);
    
    // Копирование данных с CPU на GPU
    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
    
    // Определение размеров сетки и блоков
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
    
    // Копирование результата обратно на CPU
    cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);
    
    // Проверка правильности работы ядра (только первые 10?10 элементов)
    printf("Validation in progress...\n");
    bool success = true;
    for (int i = 0; i < Arows && i < 10; i++) {
        for (int j = 0; j < Bcols && j < 10; j++) {
            BASE_TYPE sum = 0;
            for (int k = 0; k < Acols; k++) {
                sum += h_A[i * Acols + k] * h_B[k * Bcols + j];
            }
            if (fabs(h_C[i * Bcols + j] - sum) > 1e-3) {
                fprintf(stderr, "Result verification failed at element [%d, %d]!\n", i, j);
                success = false;
                break;
            }
        }
        if (!success) break;
    }
    
    printf("Test %s\n", success ? "PASSED" : "FAILED");
    
    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    // Удаление объектов событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return success ? 0 : 1;
}
