#!/bin/bash

#SBATCH -p gpuserv
#SBATCH -J lab2
#SBATCH -o lab2.%j.out
#SBATCH -e lab2.%j.err
#SBATCH --mem=10G
#SBATCH -t 00:20:00

### Начало выполнения команд ###

echo "Задача запущена на узле: $(hostname)"
echo "Время начала: $(date)"

cd /home/mag/j.karpova/lab2_3

# 1. Загружаем модуль CUDA
module load nvidia/cuda   

# 2. Компилируем нашу CUDA-программу
echo "Начинаю компиляцию..."
nvcc -std=c++17 -O3 -o lab2 lab2.cu -lpng -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lstdc++


# 3. Если компиляция прошла успешно, запускаем исполняемый файл
if [ $? -eq 0 ]; then
    echo "Компиляция успешна. Запускаю вычисления..."
    ./lab2 cat.png 
else
    echo "Ошибка компиляции!"
fi

echo "Время завершения: $(date)"
