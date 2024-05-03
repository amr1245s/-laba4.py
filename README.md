Laba4
#Вар.12 12.	Формируется матрица F следующим образом: скопировать в нее А и если в В количество простых чисел в нечетных столбцах больше, 
# чем сумма чисел в четных строках, то поменять местами В и Е симметрично, иначе С и Е поменять местами несимметрично. 
# При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, то вычисляется 
# выражение: A-1*AT – K * F-1, иначе вычисляется выражение (A-1 +G-F-1)*K, где G-нижняя треугольная матрица, полученная из А. 
# Выводятся по мере формирования А, F и все матричные операции последовательно.
 
import numpy as np
import matplotlib.pyplot as plt
 
def create_matrix(N):
    return np.random.randint(-10, 11, size=(N, N))
 
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(np.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True
 
def count_primes_in_odd_columns(B):
    count = 0
    odd_columns = B[:, 1::2]
    for value in np.nditer(odd_columns):
        if is_prime(value):
            count += 1
    return count
 
def sum_in_even_rows(C):
    even_rows = C[1::2, :]
    return np.sum(even_rows)
 
def main():
    K = int(input("Введите число K: "))
    N = int(input("Введите размерность матрицы N: "))
 
    A = create_matrix(N)
    print("Матрица A:")
    print(A)
 
    mid = N // 2
    B = A[:mid, :mid]
    C = A[:mid, mid:]
    D = A[mid:, :mid]
    E = A[mid:, mid:]
 
    F = A.copy()
    if count_primes_in_odd_columns(B) > sum_in_even_rows(C):
        F[:mid, :mid], F[mid:, mid:] = E.T, B.T
    else:
        F[:mid, mid:], F[mid:, mid:] = E, C
 
    print("Матрица F после преобразований:")
    print(F)
 
    det_A = np.linalg.det(A)
    sum_diag_F = np.trace(F)
 
    if det_A > sum_diag_F:
        result = np.linalg.inv(A) @ A.T - K * np.linalg.inv(F)
    else:
        G = np.tril(A)
        result = (np.linalg.inv(A) + G - np.linalg.inv(F)) * K
 
    print("Результат выражения:")
    print(result)
 
    plt.matshow(A)
    plt.title('Матрица A')
    plt.colorbar()
    plt.show()
 
    plt.matshow(F)
    plt.title('Матрица F')
    plt.colorbar()
    plt.show()
 
    plt.matshow(result)
    plt.title('Результат выражения')
    plt.colorbar()
    plt.show()
 
if __name__ == "__main__":
    main()

Задана рекуррентная функция. Область определения функции – натуральные числа. Написать программу сравнительного вычисления данной функции рекурсивно и итерационно.
#Определить границы применимости рекурсивного и итерационного подхода. Результаты сравнительного исследования времени вычисления представить в табличной и графической форме.
#Вариант 12: F(1) = 1, F(n) = (-1)^n* (F(n–1) + (n - 1)! /(2n)!), при n > 1
"""
import timeit
import matplotlib.pyplot as plt
 
"""
Кэш для хранения вычисленных значений факториалов
"""
factorial_cache = {0: 1, 1: 1}
 
"""
Динамическая функция для вычисления факториала
"""
def dynamic_factorial(n):
    if n not in factorial_cache:
        factorial_cache[n] = n * dynamic_factorial(n-1)
    return factorial_cache[n]
 
"""
Рекурсивная функция для вычисления факториала
"""
def recursive_factorial(n):
    if n == 0:
        return 1
    else:
        return n * recursive_factorial(n-1)
 
"""
Итеративная функция для вычисления факториала
"""
def iterative_factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result
 
"""
Динамическая функция для вычисления F(n)
"""
def dynamic_F(n, cache={1: 1}):
    if n in cache:
        return cache[n]
    else:
        """
        Здесь используем dynamic_factorial для вычисления факториалов
        """
        result = (-1)**n * (dynamic_F(n-1, cache) + dynamic_factorial(n-1) / dynamic_factorial(2*n))
        cache[n] = result
        return result
 
"""
Функция для измерения времени выполнения
"""
def score_time(func, n):
    return timeit.timeit(lambda: func(n), number=1000)
 
"""
Значения n для которых мы хотим измерить время выполнения
"""
n_values = range(1, 10)
recursive_times = []
iterative_times = []
dynamic_times = []
 
"""
Измерение времени выполнения для каждого значения n
"""
for n in n_values:
    recursive_times.append(score_time(recursive_factorial, n))
    iterative_times.append(score_time(iterative_factorial, n))
    dynamic_times.append(score_time(dynamic_F, n))
 
"""
Вывод результатов в табличной форме
"""
print(f"{'n':<10}{'Рекурсивное время (мс)':<25}{'Итерационное время (мс)':<25}{'Динамическое время (мс)':<25}")
for i, n in enumerate(n_values):
    print(f"{n:<10}{recursive_times[i]:<25}{iterative_times[i]:<25}{dynamic_times[i]:<25}")
 
"""
Построение и вывод графика результатов
"""
plt.plot(n_values, recursive_times, label='Рекурсивно')
plt.plot(n_values, iterative_times, label='Итерационно')
plt.plot(n_values, dynamic_times, label='Динамическое')
plt.xlabel('n')
plt.ylabel('Время (в миллисекундах)')
plt.legend()
plt.title('Сравнение времени вычисления функции F(n)')
plt.show()
