// =============================================================================
// Simulación de la ecuación de Poisson en 2D — Ejemplo 3
// Actividad 6: Conteo de iteraciones con #pragma omp atomic
//
// Ecuación: ∇²V = 4
// Dominio:  x ∈ [1, 2],  y ∈ [0, 2]
//
// Condiciones de frontera (Dirichlet) — consistentes con V(x,y) = (x-y)²:
//   V(1, y) = (1 - y)²     (borde izquierdo)
//   V(2, y) = (2 - y)²     (borde derecho)
//   V(x, 0) = x²           (borde inferior)
//   V(x, 2) = (x - 2)²     (borde superior)
//
// Solución analítica exacta: V(x, y) = (x - y)²
//
// Cambios respecto a poisson_critical.cpp:
//   - #pragma omp critical reemplazado por #pragma omp atomic.
//   - atomic usa instrucciones atómicas del hardware (p.ej. LOCK XADD
//     en x86) en lugar de un mutex → menor overhead que critical para
//     operaciones simples como ++iterations.
//   - Solo aplica a una única instrucción de lectura-modificación-escritura.
//     No puede proteger bloques de código arbitrarios como critical.
//
// Diferencia clave critical vs atomic:
//   critical → mutex software, protege bloques arbitrarios, mayor overhead.
//   atomic   → instrucción hardware, solo operaciones simples, menor overhead.
//   Para ++iterations, atomic es siempre preferible sobre critical.
//
// Compilación:
//   g++ -O2 -fopenmp -o poisson_atomic poisson_atomic.cpp
//
// Ejecución:
//   ./poisson_atomic
//   OMP_NUM_THREADS=4 ./poisson_atomic
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

// -----------------------------------------------------------------------------
// initialize_grid — igual que actividad 1
// -----------------------------------------------------------------------------
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
// solve_poisson — VERSIÓN ACTIVIDAD 6: atomic
//
// iterations: cuenta filas procesadas en total → protegido con atomic.
// iterations_while: cuenta vueltas del while → serial, sin protección.
//
// ¿Por qué atomic es mejor que critical aquí?
//   ++iterations es una operación atómica natural: leer, sumar 1, escribir.
//   El hardware puede garantizar esa secuencia sin interrupciones mediante
//   una sola instrucción con prefijo LOCK. No hace falta un mutex completo.
//   Resultado: menor contención entre threads y mejor rendimiento.
// -----------------------------------------------------------------------------
void solve_poisson(std::vector<std::vector<double>>& V,
                   int M, int N,
                   double h, double k,
                   int& iterations,
                   int& iterations_while)
{
    const double h2       = h * h;
    const double k2       = k * k;
    const double denom    = 2.0 * (h2 + k2);
    const double rhs_term = 4.0 * h2 * k2;

    double delta     = 1.0;
    iterations       = 0;
    iterations_while = 0;

    while (delta > TOL) {
        delta = 0.0;

        #pragma omp parallel for reduction(max:delta)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                double V_new = (  (V[i+1][j] + V[i-1][j]) * k2
                                + (V[i][j+1] + V[i][j-1]) * h2
                                - rhs_term )
                               / denom;
                delta = std::max(delta, std::abs(V_new - V[i][j]));
                V[i][j] = V_new;
            }

            // atomic: instrucción hardware de lectura-modificación-escritura.
            // Más liviano que critical para operaciones simples como esta.
            #pragma omp atomic
            ++iterations;
        }

        ++iterations_while;
    }
}

// -----------------------------------------------------------------------------
// compute_error — igual que actividad 1
// -----------------------------------------------------------------------------
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
            max_error = std::max(max_error, std::abs(V[i][j] - V_exact));
        }
    }
    return max_error;
}

// -----------------------------------------------------------------------------
// export_to_file — igual que actividad 1
// -----------------------------------------------------------------------------
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
    file << "# Actividad 6: atomic\n";
    file << "# Columnas: x  y  V_num  V_exact  error\n";

    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x       = X0 + i * h;
            double y       = Y0 + j * k;
            double V_exact = (x - y) * (x - y);
            file << x       << "\t"
                 << y       << "\t"
                 << V[i][j] << "\t"
                 << V_exact << "\t"
                 << std::abs(V[i][j] - V_exact) << "\n";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Resultados exportados a: " << filename << std::endl;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main()
{
    int M = 1024, N = 1024;
    double h, k;
    int iterations       = 0;
    int iterations_while = 0;

    std::vector<std::vector<double>> V;

    int max_threads = omp_get_max_threads();

    std::cout << "============================================\n";
    std::cout << "  Poisson 2D — Ejemplo 3 (Actividad 6)\n";
    std::cout << "  Directiva: atomic\n";
    std::cout << "  ∇²V = 4,  solución exacta: V = (x-y)²\n";
    std::cout << "  Dominio: x ∈ [1,2],  y ∈ [0,2]\n";
    std::cout << "============================================\n";
    std::cout << "Malla: M=" << M << " x N=" << N
              << "  (h=" << (XF - X0) / M
              << ", k="  << (YF - Y0) / N << ")\n";
    std::cout << "Threads disponibles: " << max_threads << "\n";

    initialize_grid(M, N, V, h, k);

    auto t_start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, iterations, iterations_while);
    auto t_end   = std::chrono::high_resolution_clock::now();

    double elapsed   = std::chrono::duration<double>(t_end - t_start).count();
    double max_error = compute_error(V, M, N, h, k);

    std::cout << "--------------------------------------------\n";
    std::cout << "Tiempo de ejecución        : " << elapsed          << " s\n";
    std::cout << "Iteraciones (filas)        : " << iterations       << "\n";
    std::cout << "Iteraciones (while)        : " << iterations_while << "\n";
    std::cout << "Tolerancia (TOL)           : " << TOL              << "\n";
    std::cout << "Error máximo (vs analítica): " << max_error        << "\n";
    std::cout << "Threads usados             : " << max_threads      << "\n";
    std::cout << "--------------------------------------------\n";

    export_to_file(V, h, k, M, N, "solucion_atomic.dat");

    std::cout << "Simulación completada.\n";
    return 0;
}
