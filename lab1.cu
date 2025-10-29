#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846
#define dtype1 float
#define dtype2 double

// CUDA ядро для инициализации массива с использованием sinf для float
__global__ void compute_sinf_float(dtype1* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = sinf((idx % 360) * PI / 180.0f);
    }
}

// CUDA ядро для инициализации массива с использованием _sinf для float
__global__ void compute__sinf_float(dtype1* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = __sinf((idx % 360) * PI / 180.0f);
    }
}

// CUDA ядро для инициализации массива с использованием sin для double
__global__ void compute_sin_double(dtype2* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = sin((idx % 360) * PI / 180.0);
    }
}

// Функция подсчета ошибки для float
void compute_error(dtype1* arr, int N, dtype1 (*function)(dtype1), dtype1& err) {
    err = 0.0f;
for (int i = 0; i < N; ++i) {
        err += fabs(function((i % 360) * PI / 180.0f) - arr[i]);
    }
    err /= N;
}

// Функция подсчета ошибки для double
void compute_error(dtype2* arr, int N, dtype2 (*function)(dtype2), dtype2& err) {
    err = 0.0;
for (int i = 0; i < N; ++i) {
        err += fabs(function((i % 360) * PI / 180.0) - arr[i]);
    }
    err /= N;
}

#define N 100000000

int main() {
    // Выделение памяти на устройстве
    dtype1* d_arr_float;
    dtype2* d_arr_double;

    cudaMalloc(&d_arr_float, N * sizeof(dtype1));
    cudaMalloc(&d_arr_double, N * sizeof(dtype2));

    // Настройка параметров запуска
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Время выполнения для __sinf (float)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    compute__sinf_float<<<numBlocks, blockSize>>>(d_arr_float, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    // Копирование результата обратно на хост
    dtype1* h_arr_float = new dtype1[N];
    cudaMemcpy(h_arr_float, d_arr_float, N * sizeof(dtype1), cudaMemcpyDeviceToHost);

    // Подсчет ошибки
    dtype1 error_float;
    compute_error(h_arr_float, N, sinf, error_float);
    
    std::cout << "Error using __sinf (float): " << error_float << " | Time: " << time_ms << " ms" << std::endl;

    // Время выполнения для sinf (float)
    cudaEventRecord(start);
    compute_sinf_float<<<numBlocks, blockSize>>>(d_arr_float, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&time_ms, start, stop);
    
    cudaMemcpy(h_arr_float, d_arr_float, N * sizeof(dtype1), cudaMemcpyDeviceToHost);
    
    compute_error(h_arr_float, N, sinf, error_float);
    
    std::cout << "Error using sinf (float): " << error_float << " | Time: " << time_ms << " ms" << std::endl;

    // Время выполнения для sin (double)
    cudaEventRecord(start);
    compute_sin_double<<<numBlocks, blockSize>>>(d_arr_double, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&time_ms, start, stop);
    
    dtype2* h_arr_double = new dtype2[N];
    cudaMemcpy(h_arr_double, d_arr_double, N * sizeof(dtype2), cudaMemcpyDeviceToHost);
    
    dtype2 error_double;
    compute_error(h_arr_double, N, sin, error_double);
    
    std::cout << "Error using sin (double): " << error_double << " | Time: " << time_ms << " ms" << std::endl;

    // Освобождение памяти
    delete[] h_arr_float;
    delete[] h_arr_double;
    cudaFree(d_arr_float);
    cudaFree(d_arr_double);

    return 0;
}






