#!/bin/bash

#SBATCH -p gpuserv
#SBATCH -J lab4_5
#SBATCH -o lab4_5.%j.out
#SBATCH -e lab4_5.%j.err
#SBATCH --mem=10G
#SBATCH -t 00:20:00

### Начало выполнения команд ###

echo "Задача запущена на узле: $(hostname)"
echo "Время начала: $(date)"

# 1. Загружаем модуль CUDA
module load nvidia/cuda   

# 2. Компилируем нашу CUDA-программу
echo "Начинаю компиляцию..."
nvcc lab4_5.cu -o lab4_5 

# 3. Если компиляция прошла успешно, запускаем исполняемый файл
if [ $? -eq 0 ]; then
    echo "Компиляция успешна. Запускаю вычисления..."
    ./lab4_5 
else
    echo "Ошибка компиляции!"
fi

echo "Время завершения: $(date)"
