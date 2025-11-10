# Отчет по лабораторной №1
Лабы по Программированию ГП ФИТ, НГУ

Задание

Выделить на GPU массив arr из 10^9 элементов типа float и инициализировать его с помощью ядра следующим образом: arr[i] = sin((i%360)*Pi/180). Скопировать массив в память центрального процессора и посчитать ошибку err = sum_i(abs(sin((i%360)*Pi/180) - arr[i]))/10^9. Провести исследование зависимости результата от использования функций: sin, sinf, __sin. Объяснить результат. Проверить результат при использовании массива типа double.

Характеристики устройства
  Device Name: Quadro RTX 4000
  Total Global Memory (bytes): 8167620608
  Shared Memory per Block (bytes): 49152
  Compute Capability: 7.5
  Max Threads per Block: 1024
  Multiprocessor Count: 36
  Clock Rate (kHz): 1545000
  Warp Size: 32
  Registers Per Block: 65536
  Memory Pitch (bytes): 2147483647
  ECC Enabled: No

Результат выполнения программы:
Error using __sinf (float): 8.01664e-08 | Time: 19.2553 ms
Error using sinf (float): 8.3571e-09 | Time: 18.7772 ms
Error using sin (double): 8.77963e-18 | Time: 43.977 ms

Выводы:
