// =============================================================================
// Simulación de la ecuación de Poisson en 2D — Ejemplo 3
// Actividad 7: Paralelización con tareas (#pragma omp task)
//
// Ecuación: ∇²V = 4
// Dominio:  x ∈ [1, 2],  y ∈ [0, 2]
//
// Cambios respecto a parallel_for:
//   - Se divide el dominio en bloques (ej. 64x64).
//   - Se usa una región paralela con un bloque 'single' creador de tareas.
//   - Se lanza una tarea por cada bloque para actualizar sus valores.
//   - Se calcula un delta local por tarea y se actualiza el global con 'critical'.
//
// Compilación:
//   g++ -O2 -fopenmp -o poisson_task poisson_task.cpp
//
// Ejecución:
//   ./poisson_task
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

const double X0  = 1.0, XF = 2.0;
const double Y0  = 0.0, YF = 2.0;
const double TOL = 1e-6;

void initialize_grid(int M, int N,
                     std::vector<std::vector<double>>& V,
                     double& h, double& k)
{
    h = (XF - X0) / M;
    k = (YF - Y0) / N;

    V.assign(M + 1, std::vector<double>(N + 1, 0.0));

    for (int j = 0; j <= N; ++j) {
        double y = Y0 + j * k;
        V[0][j]  = (1.0 - y) * (1.0 - y);
    }
    for (int j = 0; j <= N; ++j) {
        double y = Y0 + j * k;
        V[M][j]  = (2.0 - y) * (2.0 - y);
    }
    for (int i = 0; i <= M; ++i) {
        double x = X0 + i * h;
        V[i][0]  = x * x;
    }
    for (int i = 0; i <= M; ++i) {
        double x = X0 + i * h;
        V[i][N]  = (x - 2.0) * (x - 2.0);
    }
}

// -----------------------------------------------------------------------------
// solve_poisson  —  VERSIÓN ACTIVIDAD 7: task
// -----------------------------------------------------------------------------
void solve_poisson(std::vector<std::vector<double>>& V,
                   int M, int N,
                   double h, double k,
                   int& iterations)
{
    const double h2       = h * h;
    const double k2       = k * k;
    const double denom    = 2.0 * (h2 + k2);
    const double rhs_term = 4.0 * h2 * k2;

    double delta = 1.0;
    iterations   = 0;

    // Tamaño del bloque para dividir la grilla (puedes ajustar este valor)
    const int block_size = 64;

    while (delta > TOL) {
        delta = 0.0;

        // Se abre la región paralela dentro del while
        #pragma omp parallel
        {
            // Un solo hilo se encarga de recorrer la grilla y crear las tareas
            #pragma omp single
            {
                // Iteramos por bloques saltando de 'block_size' en 'block_size'
                for (int bi = 1; bi < M; bi += block_size) {
                    for (int bj = 1; bj < N; bj += block_size) {

                        // Por cada cuadrante, lanzamos una tarea independiente
                        #pragma omp task
                        {
                            double local_delta = 0.0;

                            // Calculamos los límites reales de este bloque
                            int i_end = std::min(bi + block_size, M);
                            int j_end = std::min(bj + block_size, N);

                            // Bucle original restringido al tamaño del bloque
                            for (int i = bi; i < i_end; ++i) {
                                for (int j = bj; j < j_end; ++j) {

                                    double V_new = (  (V[i + 1][j] + V[i - 1][j]) * k2
                                                    + (V[i][j + 1] + V[i][j - 1]) * h2
                                                    - rhs_term )
                                                   / denom;

                                    local_delta = std::max(local_delta, std::abs(V_new - V[i][j]));
                                    V[i][j] = V_new;
                                }
                            }

                            // Actualizamos el delta global de forma segura
                            #pragma omp critical
                            {
                                if (local_delta > delta) {
                                    delta = local_delta;
                                }
                            }
                        } // Fin de la tarea
                    }
                }

                // Barrera de sincronización: esperamos a que TODAS las tareas
                // de esta iteración terminen antes de volver a evaluar el while
                #pragma omp taskwait

            } // Fin de single (hay una barrera implícita aquí también)
        } // Fin de parallel

        ++iterations;
    }
}

double compute_error(const std::vector<std::vector<double>>& V,
                     int M, int N,
                     double h, double k)
{
    double max_error = 0.0;
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x       = X0 + i * h;
            double y       = Y0 + j * k;
            double V_exact = (x - y) * (x - y);
            double error   = std::abs(V[i][j] - V_exact);
            max_error      = std::max(max_error, error);
        }
    }
    return max_error;
}

void export_to_file(const std::vector<std::vector<double>>& V,
                    double h, double k,
                    int M, int N,
                    const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: no se pudo abrir el archivo " << filename << std::endl;
        return;
    }

    file << "# Solución numérica de ∇²V = 4  en  [1,2] x [0,2]\n";
    file << "# Actividad 7: task\n";
    file << "# Columnas: x  y  V_num  V_exact  error\n";

    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x       = X0 + i * h;
            double y       = Y0 + j * k;
            double V_exact = (x - y) * (x - y);
            double error   = std::abs(V[i][j] - V_exact);

            file << x         << "\t"
                 << y         << "\t"
                 << V[i][j]   << "\t"
                 << V_exact   << "\t"
                 << error     << "\n";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Resultados exportados a: " << filename << std::endl;
}

int main()
{
    int M = 1024, N = 1024;
    double h, k;
    int iterations = 0;

    std::vector<std::vector<double>> V;
    int max_threads = omp_get_max_threads();

    std::cout << "============================================\n";
    std::cout << "  Poisson 2D — Ejemplo 3 (Actividad 7)\n";
    std::cout << "  Directiva: task (Por bloques)\n";
    std::cout << "  ∇²V = 4,  solución exacta: V = (x-y)²\n";
    std::cout << "  Dominio: x ∈ [1,2],  y ∈ [0,2]\n";
    std::cout << "============================================\n";
    std::cout << "Malla: M=" << M << " x N=" << N
              << "  (h=" << (XF - X0) / M
              << ", k="  << (YF - Y0) / N << ")\n";
    std::cout << "Threads disponibles (omp_get_max_threads): "
              << max_threads << "\n";

    initialize_grid(M, N, V, h, k);

    auto t_start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, iterations);
    auto t_end   = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    double max_error = compute_error(V, M, N, h, k);

    std::cout << "--------------------------------------------\n";
    std::cout << "Tiempo de ejecución        : " << elapsed    << " s\n";
    std::cout << "Número de iteraciones      : " << iterations << "\n";
    std::cout << "Tolerancia (TOL)           : " << TOL        << "\n";
    std::cout << "Error máximo (vs analítica): " << max_error  << "\n";
    std::cout << "Threads usados             : " << max_threads << "\n";
    std::cout << "--------------------------------------------\n";

    export_to_file(V, h, k, M, N, "solucion_task.dat");

    std::cout << "Simulación completada.\n";
    return 0;
}
