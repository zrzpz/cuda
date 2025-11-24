#!/bin/bash

#SBATCH -p gpuserv
#SBATCH -J lab3
#SBATCH -o lab3_3.%j.out
#SBATCH -e lab3_3.%j.err
#SBATCH --mem=10G
#SBATCH -t 00:20:00

### Начало выполнения команд ###

echo "Задача запущена на узле: $(hostname)"
echo "Время начала: $(date)"

cd /home/mag/j.karpova/lab3

# 1. Загружаем модуль CUDA
module load nvidia/cuda   

# 2. Компилируем нашу CUDA-программу
echo "Начинаю компиляцию..."
nvcc -o lab3 lab3.cu -lpng -lopencv_core -lopencv_imgcodecs -lopencv_highgui 

# 3. Если компиляция прошла успешно, запускаем исполняемый файл
if [ $? -eq 0 ]; then
    echo "Компиляция успешна. Запускаю вычисления..."
    ./lab3 images.png 
else
    echo "Ошибка компиляции!"
fi

echo "Время завершения: $(date)"
