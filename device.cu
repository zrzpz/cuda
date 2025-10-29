#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Получить количество CUDA-совместимых устройств

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop; // Инициализация структуры cudaDeviceProp
        cudaGetDeviceProperties(&prop, i); // Получить свойства для устройства 'i'

        std::cout << "--- Device Number: " << i << " ---" << std::endl;
        std::cout << "  Device Name: " << prop.name << std::endl;
        std::cout << "  Total Global Memory (bytes): " << prop.totalGlobalMem << std::endl;
        std::cout << "  Shared Memory per Block (bytes): " << prop.sharedMemPerBlock << std::endl;

        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Clock Rate (kHz): " << prop.clockRate << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Registers Per Block: " << prop.regsPerBlock << std::endl; // Добавлено
        std::cout << "  Memory Pitch (bytes): " << prop.memPitch << std::endl; // Добавлено
        std::cout << "  ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }

    return 0;
}