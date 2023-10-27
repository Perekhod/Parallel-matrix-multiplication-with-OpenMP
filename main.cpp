#include <iostream>
#include <omp.h>
#include <random>

using namespace std;

// Константы для размеров матриц
const int N = 1000;
const int M = 1000;
const int P = 1000;

// Функция для генерации случайных чисел
double random_number() 
{
    // Создаем генератор случайных чисел с фиксированным зерном для воспроизводимости
    static std::mt19937 gen(42);
    // Создаем равномерное распределение от 0.0 до 1.0
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    // Возвращаем случайное число из распределения
    return dis(gen);
}

// Функция для создания и заполнения матрицы
double** create_matrix(int rows, int cols) 
{
    double** matrix = new double* [rows];
    for (int i = 0; i < rows; i++) 
    {
        matrix[i] = new double[cols];
        for (int j = 0; j < cols; j++) 
        {
            matrix[i][j] = random_number();
        }
    }
    return matrix;
}

// Функция для освобождения памяти матрицы
void delete_matrix(double** matrix, int rows) 
{
    for (int i = 0; i < rows; i++) 
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Функция для умножения двух матриц с использованием OpenMP
double** multiply_matrix(double** A, double** B, int num_threads) 
{
    double** C = new double* [N];
    for (int i = 0; i < N; i++) 
    {
        C[i] = new double[P];
        for (int j = 0; j < P; j++) 
        {
            C[i][j] = 0.0;
        }
    }

    // Засекаем время начала
    double start_time = omp_get_wtime();

    // Устанавливаем количество потоков
    omp_set_num_threads(num_threads);

    // Распараллеливаем внешний цикл с динамическим расписанием
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < P; j++) 
        {
            // Объявляем локальную переменную скалярного типа
            double sum = 0.0;
            // Применяем метод reduction для внутреннего цикла
#pragma omp parallel for reduction(+:sum)
            for (int k = 0; k < M; k++) 
            {
                sum += A[i][k] * B[k][j];
            }
            // Присваиваем значение локальной переменной элементу матрицы
            C[i][j] = sum;
        }
    }

    // Засекаем время окончания
    double end_time = omp_get_wtime();

    // Выводим время работы
    cout << "Time with " << num_threads << " threads: " << end_time - start_time << " seconds." << endl;

    return C;
}

int main() 
{

    // Создаем и заполняем матрицы A и B
    cout << "Creating matrices A and B..." << endl;
    double** A = create_matrix(N, M);
    double** B = create_matrix(M, P);

    // Умножаем матрицы с разным количеством потоков
    cout << "Multiplying matrices with OpenMP..." << endl;
    double** C1 = multiply_matrix(A, B, 1);
    double** C2 = multiply_matrix(A, B, 2);
    double** C4 = multiply_matrix(A, B, 4);
    double** C8 = multiply_matrix(A, B, 8);
    double** C16 = multiply_matrix(A, B, 16);

    // Освобождаем память матриц
    cout << "Deleting matrices..." << endl;
    delete_matrix(A, N);
    delete_matrix(B, M);
    delete_matrix(C1, N);
    delete_matrix(C2, N);
    delete_matrix(C4, N);
    delete_matrix(C8, N);
    delete_matrix(C16, N);

    return 0;
}
