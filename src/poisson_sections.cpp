// =============================================================================
// Simulación de la ecuación de Poisson en 2D — Ejemplo 3
// Actividad 3: Paralelización con sections
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
// Cambios respecto a la actividad 2 (poisson_collapse.cpp):
//   - initialize_grid se divide en dos funciones independientes:
//       * allocate_grid   → reserva memoria e inicializa en cero
//       * apply_boundary  → impone las 4 condiciones de frontera
//   - En main, ambas se lanzan en paralelo con #pragma omp parallel sections
//   - solve_poisson permanece igual que en la actividad 2 (collapse + reduction)
//
// Reflexión sobre el reordenamiento:
//   allocate_grid DEBE ejecutarse antes que apply_boundary porque esta última
//   necesita que la memoria ya esté reservada. Por eso allocate_grid no puede
//   ir dentro de una section junto a apply_boundary en el mismo nivel.
//   La solución es llamar allocate_grid de forma serial primero, y luego
//   paralelizar solo las partes que son verdaderamente independientes:
//   los cuatro bordes entre sí (izquierdo, derecho, inferior, superior).
//
// Compilación:
//   g++ -O2 -fopenmp -o poisson_s poisson_sections.cpp
//
// Ejecución:
//   ./poisson_s
//   OMP_NUM_THREADS=4 ./poisson_s
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
// allocate_grid
//
// Reserva memoria para V y la inicializa en cero.
// Calcula h y k. Debe ejecutarse ANTES de apply_boundary.
// -----------------------------------------------------------------------------
void allocate_grid(int M, int N,
                   std::vector<std::vector<double>>& V,
                   double& h, double& k)
{
    h = (XF - X0) / M;
    k = (YF - Y0) / N;
    V.assign(M + 1, std::vector<double>(N + 1, 0.0));
}

// -----------------------------------------------------------------------------
// apply_boundary
//
// Impone las cuatro condiciones de frontera de Dirichlet.
// Los cuatro bordes son independientes entre sí (no comparten celdas
// excepto las esquinas, que ambos bordes escriben el mismo valor),
// por lo que se pueden asignar a sections distintas sin conflicto.
// -----------------------------------------------------------------------------
void apply_boundary(int M, int N,
                    std::vector<std::vector<double>>& V,
                    double h, double k)
{
    // Se usan sections para ejecutar los bordes en paralelo.
    // Con 4 sections OpenMP usa hasta 4 threads simultáneos.
    // Los bordes izquierdo/derecho e inferior/superior no comparten
    // celdas internas; las esquinas reciben el mismo valor desde
    // ambas sections que las tocan → sin condición de carrera.
    #pragma omp parallel sections
    {
        // Sección 1 — Borde izquierdo: V(1, y) = (1 - y)²
        #pragma omp section
        {
            for (int j = 0; j <= N; ++j) {
                double y = Y0 + j * k;
                V[0][j]  = (1.0 - y) * (1.0 - y);
            }
        }

        // Sección 2 — Borde derecho: V(2, y) = (2 - y)²
        #pragma omp section
        {
            for (int j = 0; j <= N; ++j) {
                double y = Y0 + j * k;
                V[M][j]  = (2.0 - y) * (2.0 - y);
            }
        }

        // Sección 3 — Borde inferior: V(x, 0) = x²
        #pragma omp section
        {
            for (int i = 0; i <= M; ++i) {
                double x = X0 + i * h;
                V[i][0]  = x * x;
            }
        }

        // Sección 4 — Borde superior: V(x, 2) = (x - 2)²
        #pragma omp section
        {
            for (int i = 0; i <= M; ++i) {
                double x = X0 + i * h;
                V[i][N]  = (x - 2.0) * (x - 2.0);
            }
        }
    }
    // Barrera implícita al salir de parallel sections:
    // todos los bordes están escritos antes de continuar.
}

// -----------------------------------------------------------------------------
// solve_poisson — igual que en la actividad 2
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
        }
        ++iterations;
    }
}

// -----------------------------------------------------------------------------
// compute_error
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
            max_error      = std::max(max_error, std::abs(V[i][j] - V_exact));
        }
    }
    return max_error;
}

// -----------------------------------------------------------------------------
// export_to_file
// -----------------------------------------------------------------------------
void export_to_file(const std::vector<std::vector<double>>& V,
                    double h, double k,
                    int M, int N,
                    const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: no se pudo abrir " << filename << std::endl;
        return;
    }

    file << "# Solución numérica de ∇²V = 4  en  [1,2] x [0,2]\n";
    file << "# Actividad 3: sections\n";
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
    int iterations = 0;

    std::vector<std::vector<double>> V;

    std::cout << "============================================\n";
    std::cout << "  Poisson 2D — Ejemplo 3 (Actividad 3)\n";
    std::cout << "  Directiva: sections\n";
    std::cout << "  ∇²V = 4,  solución exacta: V = (x-y)²\n";
    std::cout << "  Dominio: x ∈ [1,2],  y ∈ [0,2]\n";
    std::cout << "============================================\n";
    std::cout << "Malla: M=" << M << " x N=" << N << "\n";
    std::cout << "Threads disponibles: " << omp_get_max_threads() << "\n";

    // Paso 1 (serial): reservar memoria — apply_boundary la necesita
    auto t0 = std::chrono::high_resolution_clock::now();
    allocate_grid(M, N, V, h, k);

    // Paso 2 (parallel sections): los 4 bordes en paralelo
    apply_boundary(M, N, V, h, k);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_init = std::chrono::duration<double>(t1 - t0).count();

    // Paso 3: resolver
    auto t2 = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, iterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_solve = std::chrono::duration<double>(t3 - t2).count();

    double max_error = compute_error(V, M, N, h, k);

    std::cout << "--------------------------------------------\n";
    std::cout << "Tiempo init + boundary (sections) : " << t_init  << " s\n";
    std::cout << "Tiempo solve_poisson              : " << t_solve << " s\n";
    std::cout << "Tiempo total                      : " << t_init + t_solve << " s\n";
    std::cout << "Número de iteraciones             : " << iterations << "\n";
    std::cout << "Tolerancia (TOL)                  : " << TOL << "\n";
    std::cout << "Error máximo (vs analítica)       : " << max_error << "\n";
    std::cout << "--------------------------------------------\n";

    export_to_file(V, h, k, M, N, "solucion_sections.dat");

    std::cout << "Simulación completada.\n";
    return 0;
}
