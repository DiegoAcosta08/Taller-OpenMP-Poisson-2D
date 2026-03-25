// =============================================================================
// Simulación de la ecuación de Poisson en 2D — Ejemplo 3
// Actividad 5: nowait, single, barrier, critical
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
// Estructura de solve_poisson:
//   - #pragma omp single (verbose): imprime delta por iteración.
//     Solo activo cuando verbose=true. Su overhead distorsiona el tiempo,
//     por eso la medición de referencia se hace con verbose=false.
//   - #pragma omp for + reduction(max:delta): actualización principal.
//   - #pragma omp for nowait: bucle independiente de norma L1.
//     nowait permite que los threads que terminen antes avancen al
//     critical sin esperar a los demás.
//   - #pragma omp critical: acumula aportes locales en norma_L1.
//     Necesario porque varios threads escriben sobre la misma variable.
//   - #pragma omp single (convergencia): un solo hilo actualiza
//     iterations, evalúa convergencia y resetea delta y norma_L1.
//
// Dos ejecuciones en main:
//   1. verbose=false → tiempo limpio, sin overhead de print (referencia)
//   2. verbose=true  → muestra comportamiento iterativo, tiempo no reportado
//
// Compilación:
//   g++ -O2 -fopenmp -o poisson_sync poisson_sync.cpp
//
// Ejecución:
//   ./poisson_sync
//   OMP_NUM_THREADS=4 ./poisson_sync
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <omp.h>

const double X0  = 1.0, XF = 2.0;
const double Y0  = 0.0, YF = 2.0;
const double TOL = 1e-6;

// -----------------------------------------------------------------------------
// allocate_grid — igual que actividades anteriores
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
// apply_boundary — igual que actividades anteriores
// -----------------------------------------------------------------------------
void apply_boundary(int M, int N,
                    std::vector<std::vector<double>>& V,
                    double h, double k)
{
    #pragma omp parallel sections
    {
        #pragma omp section
        for (int j = 0; j <= N; ++j) {
            double y = Y0 + j * k;
            V[0][j]  = (1.0 - y) * (1.0 - y);
        }

        #pragma omp section
        for (int j = 0; j <= N; ++j) {
            double y = Y0 + j * k;
            V[M][j]  = (2.0 - y) * (2.0 - y);
        }

        #pragma omp section
        for (int i = 0; i <= M; ++i) {
            double x = X0 + i * h;
            V[i][0]  = x * x;
        }

        #pragma omp section
        for (int i = 0; i <= M; ++i) {
            double x = X0 + i * h;
            V[i][N]  = (x - 2.0) * (x - 2.0);
        }
    }
}

// -----------------------------------------------------------------------------
// solve_poisson
//
// verbose=false → ejecución limpia, sin print, tiempo de referencia
// verbose=true  → imprime delta cada iteración para observar convergencia
// -----------------------------------------------------------------------------
void solve_poisson(std::vector<std::vector<double>>& V,
                   int M, int N,
                   double h, double k,
                   int& iterations,
                   double& norma_L1,
                   bool verbose)
{
    const double h2       = h * h;
    const double k2       = k * k;
    const double denom    = 2.0 * (h2 + k2);
    const double rhs_term = 4.0 * h2 * k2;

    double delta   = 1.0;
    bool converged = false;
    iterations     = 0;
    norma_L1       = 0.0;

    #pragma omp parallel shared(V, delta, iterations, converged, norma_L1)
    {
        while (!converged) {

            // ------------------------------------------------------------------
            // PRINT CONTROLADO — solo cuando verbose=true
            // El single introduce una barrera implícita que serializa threads
            // al inicio de cada iteración → overhead medible.
            // Con verbose=false este bloque desaparece completamente,
            // dando el tiempo limpio de nowait + critical.
            // ------------------------------------------------------------------
            if (verbose) {
                #pragma omp single
                {
                    if (iterations < 20 || iterations % 5000 == 0) {
                        std::cout << "  Iter " << std::setw(6) << iterations + 1
                                  << "  |  delta = "
                                  << std::scientific << std::setprecision(6)
                                  << delta << "\n";
                    }
                }
            }

            // ------------------------------------------------------------------
            // ACTUALIZACIÓN PRINCIPAL
            // parallel for con reduction: cada thread acumula su máximo local
            // en delta; OpenMP los combina al salir del for.
            // Barrera implícita al salir: todos los threads tienen delta global.
            // ------------------------------------------------------------------
            #pragma omp for reduction(max:delta) schedule(static)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    double V_new = ((V[i+1][j] + V[i-1][j]) * k2
                                  + (V[i][j+1] + V[i][j-1]) * h2
                                  - rhs_term) / denom;
                    delta = std::max(delta, std::abs(V_new - V[i][j]));
                    V[i][j] = V_new;
                }
            }

            // ------------------------------------------------------------------
            // BUCLE INDEPENDIENTE CON nowait
            // Calcula la norma L1 sobre la grilla actualizada. Es independiente
            // del siguiente single (que solo lee delta, no norma_L1 todavía),
            // por lo que no hace falta esperar a que todos los threads terminen
            // este for antes de avanzar al critical.
            // nowait elimina la barrera implícita al final del for.
            // ------------------------------------------------------------------
            double aporte_local = 0.0;

            #pragma omp for nowait schedule(static)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    aporte_local += std::abs(V[i][j]);
                }
            }

            // ------------------------------------------------------------------
            // ACUMULACIÓN CON critical
            // Cada thread suma su aporte_local a norma_L1 de forma exclusiva.
            // Sin critical habría condición de carrera: varios threads
            // leerían y escribirían norma_L1 simultáneamente.
            // ------------------------------------------------------------------
            #pragma omp critical
            {
                norma_L1 += aporte_local;
            }

            // ------------------------------------------------------------------
            // CONVERGENCIA CON single
            // Un solo hilo actualiza el estado global: incrementa iterations,
            // evalúa si se alcanzó TOL y resetea delta y norma_L1 si no.
            // Barrera implícita al salir: todos ven el estado actualizado
            // antes de la siguiente vuelta del while.
            // ------------------------------------------------------------------
            #pragma omp single
            {
                ++iterations;

                if (delta <= TOL) {
                    converged = true;
                    if (verbose) {
                        std::cout << "  Iter " << std::setw(6) << iterations
                                  << "  |  delta final = "
                                  << std::scientific << std::setprecision(6)
                                  << delta << "\n";
                    }
                } else {
                    norma_L1 = 0.0;
                    delta    = 0.0;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// compute_error — igual que actividades anteriores
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
// main — dos ejecuciones: limpia (referencia) y verbose (observación)
// -----------------------------------------------------------------------------
int main()
{
    int M = 1024, N = 1024;
    double h, k;
    int    iterations = 0;
    double norma_L1   = 0.0;

    std::cout << "============================================\n";
    std::cout << "  Poisson 2D — Ejemplo 3 (Actividad 5)\n";
    std::cout << "  Directivas: nowait, single, critical\n";
    std::cout << "  ∇²V = 4,  solución exacta: V = (x-y)²\n";
    std::cout << "  Dominio: x ∈ [1,2],  y ∈ [0,2]\n";
    std::cout << "============================================\n";
    std::cout << "Threads disponibles: " << omp_get_max_threads() << "\n\n";

    // ------------------------------------------------------------------
    // EJECUCIÓN 1: verbose=false — tiempo de referencia limpio
    // ------------------------------------------------------------------
    std::cout << "--- Ejecucion 1: sin print (tiempo de referencia) ---\n";

    std::vector<std::vector<double>> V1;
    allocate_grid(M, N, V1, h, k);
    apply_boundary(M, N, V1, h, k);

    auto t0 = std::chrono::high_resolution_clock::now();
    solve_poisson(V1, M, N, h, k, iterations, norma_L1, false);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_clean = std::chrono::duration<double>(t1 - t0).count();

    double max_error1 = compute_error(V1, M, N, h, k);

    std::cout << "Tiempo de ejecucion : " << elapsed_clean << " s\n";
    std::cout << "Iteraciones         : " << iterations    << "\n";
    std::cout << "Error maximo        : " << max_error1    << "\n";
    std::cout << "Norma L1            : " << norma_L1      << "\n\n";

    // ------------------------------------------------------------------
    // EJECUCIÓN 2: verbose=true — observación del comportamiento
    // Tiempo no reportado como referencia por overhead del single+print
    // ------------------------------------------------------------------
    std::cout << "--- Ejecucion 2: con print (solo observacion) ---\n";

    std::vector<std::vector<double>> V2;
    double norma_L1_v = 0.0;
    int    iters_v    = 0;
    allocate_grid(M, N, V2, h, k);
    apply_boundary(M, N, V2, h, k);

    solve_poisson(V2, M, N, h, k, iters_v, norma_L1_v, true);

    double max_error2 = compute_error(V2, M, N, h, k);

    std::cout << "Iteraciones         : " << iters_v    << "\n";
    std::cout << "Error maximo        : " << max_error2 << "\n";
    std::cout << "(Tiempo no reportado — overhead del print distorsiona)\n\n";

    std::cout << "============================================\n";
    std::cout << "Simulacion completada.\n";
    return 0;
}
