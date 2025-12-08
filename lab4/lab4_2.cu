#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// размер блока 
#define BLOCK_SIZE 16

// тип, который будут иметь элементы матриц 
#define BASE_TYPE float

// Ядро
// Функция транспонирования матрицы
__global__ void matrixTranspose(const BASE_TYPE *A, BASE_TYPE *AT, int rows, int cols)
{
    // Индекс элемента в исходной матрице
    int iA = cols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

    // Индекс соответствующего элемента в
    // транспонированной матрице
    int iAT = rows * (blockDim.x * blockIdx.x + threadIdx.x) + blockDim.y * blockIdx.y + threadIdx.y;
    AT[iAT] = A[iA];
}

// Функция вычисления числа, которое больше числа а
// и кратное числу b
int toMultiple(int a, int b)
{
    int mod = a % b;
    if (mod != 0)
    {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

int main()
{
    // Объекты событий 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Количество строк и столбцов матрицы 
    int rows = 1000;
    int cols = 2000;

    // Меняем количество строк и столбцов матрицы
    // на число, кратное размеру блока (16) 
    rows = toMultiple(rows, BLOCK_SIZE);
    printf("rows = %d\n", rows);
    cols = toMultiple(cols, BLOCK_SIZE);
    printf("cols = %d\n", cols);
    size_t size = rows * cols * sizeof(BASE_TYPE);

    // Выделение памяти под матрицы на хосте
    // Исходная матрица
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(size);
    // Транспонированная матрица
    BASE_TYPE *h_AT = (BASE_TYPE *)malloc(size);

    // Инициализация матрицы
    for (int i = 0; i < rows * cols; ++i)
    {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    }

    // Выделение глобальной памяти на девайсе
    // для исходной матрицы 
    BASE_TYPE *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    // Выделение глобальной памяти на девайсе для
    // транспонированной матрицы 
    BASE_TYPE *d_AT = NULL;
    cudaMalloc((void **)&d_AT, size);

    // Копируем матрицу из CPU на GPU 
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Определяем размер блока и сетки 
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(cols / BLOCK_SIZE, rows / BLOCK_SIZE);

    // Начать отсчет времени 
    cudaEventRecord(start, 0);

    // Запуск ядра
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_AT, rows, cols);

    // Окончание работы ядра, остановка времени
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float KernelTime;
    cudaEventElapsedTime(&KernelTime, start, stop);

    printf("KernelTime: %.2f milliseconds\n", KernelTime);

    // Копируем матрицу из GPU на CPU
    cudaMemcpy(h_AT, d_AT, size, cudaMemcpyDeviceToHost);

    // Проверка правильности работы ядра
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            if (h_A[i * cols + j] != h_AT[j * rows + i])
                fprintf(stderr, "Result verification failed at element [%d, %d]!\n", i, j);
            exit(EXIT_FAILURE);
        }

    printf("Test PASSED\n");

    // Освобождаем память на GPU
    cudaFree(d_A);
    // Освобождаем память на GPU
    cudaFree(d_AT);
    // Освобождаем память на CPU
    free(h_A);
    free(h_AT);

    // Удаляем объекты событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Done\n");
    return 0;
}
