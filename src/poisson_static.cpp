// =============================================================================
// Simulación de la ecuación de Poisson en 2D — Ejemplo 3
// Actividad 4: parallel y for separados + schedule(static)
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
// Nota sobre la implementación del while dentro de la región paralela:
//   Usar 'break' dentro de un while que contiene directivas OpenMP produce
//   comportamiento indefinido según el estándar de OpenMP. La razón es que
//   las barreras implícitas del 'for' y del 'single' deben ser alcanzadas
//   por TODOS los threads en cada iteración. Si un thread sale por 'break'
//   antes de llegar a una barrera, los demás quedan bloqueados esperando
//   indefinidamente → los threads aparecen vivos en htop pero sin hacer
//   trabajo útil.
//
//   Solución: reemplazar el break por una variable bandera compartida
//   'converged' que se evalúa en el while y se actualiza dentro del single,
//   garantizando que todos los threads pasen por todas las barreras en
//   cada iteración sin excepción.
//
// Estructura de sincronización por iteración del while (2 barreras):
//   1. Barrera implícita al salir del #pragma omp for
//      → delta contiene el máximo global real de esta iteración
//   2. #pragma omp single: evalúa convergencia, actualiza converged y delta
//      Barrera implícita al salir del single
//      → todos ven converged y delta correctos antes de la siguiente vuelta
//
// Compilación:
//   g++ -O2 -fopenmp -o poisson_static poisson_static.cpp
//
// Ejecución:
//   ./poisson_static
//   OMP_NUM_THREADS=4 ./poisson_static
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
// allocate_grid — igual que actividad 3
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
// apply_boundary — igual que actividad 3
// -----------------------------------------------------------------------------
void apply_boundary(int M, int N,
                    std::vector<std::vector<double>>& V,
                    double h, double k)
{
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int j = 0; j <= N; ++j) {
                double y = Y0 + j * k;
                V[0][j]  = (1.0 - y) * (1.0 - y);
            }
        }
        #pragma omp section
        {
            for (int j = 0; j <= N; ++j) {
                double y = Y0 + j * k;
                V[M][j]  = (2.0 - y) * (2.0 - y);
            }
        }
        #pragma omp section
        {
            for (int i = 0; i <= M; ++i) {
                double x = X0 + i * h;
                V[i][0]  = x * x;
            }
        }
        #pragma omp section
        {
            for (int i = 0; i <= M; ++i) {
                double x = X0 + i * h;
                V[i][N]  = (x - 2.0) * (x - 2.0);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// solve_poisson — VERSIÓN ACTIVIDAD 4: parallel + for separados + schedule
//
// La región #pragma omp parallel se abre UNA sola vez fuera del while,
// evitando el overhead de crear y destruir threads en cada iteración.
// Dentro del while, #pragma omp for distribuye el trabajo entre los threads
// ya activos.
//
// Variable 'converged':
//   Bandera compartida que reemplaza al 'break'. Se actualiza dentro del
//   'single' después de cada iteración. Todos los threads la leen en la
//   condición del while, garantizando que ningún thread abandone la región
//   paralela antes de que todos hayan pasado por las barreras.
//
// Flujo por iteración del while:
//
//   [todos los threads]
//   │
//   ├─ #pragma omp for reduction(max:delta) schedule(static)
//   │    Cada thread actualiza su bloque de filas (asignación estática,
//   │    bloques del mismo tamaño, determinista). Acumula delta local.
//   │    Barrera implícita al salir: delta = máximo global de esta iteración.
//   │
//   └─ #pragma omp single
//        Un thread evalúa delta vs TOL:
//          - Si convergió: converged = true  (no resetea delta)
//          - Si no:        delta = 0.0       (resetea para la siguiente vuelta)
//        Incrementa iterations.
//        Barrera implícita al salir: todos ven converged y delta actualizados.
//
//   → while (!converged): todos evalúan la misma bandera y salen juntos.
//
// Parámetros:
//   V          — matriz solución (entrada/salida)
//   M, N       — número de subdivisiones en x e y
//   h, k       — pasos de malla
//   iterations — número de iteraciones realizadas (salida)
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

    double delta    = 1.0;
    bool converged  = false;   // Bandera compartida de convergencia
    iterations      = 0;

    // La región paralela se abre UNA sola vez aquí.
    // Todos los threads comparten V, delta, iterations y converged.
    #pragma omp parallel shared(V, delta, iterations, converged)
    {
        while (!converged) {

            // -----------------------------------------------------------------
            // Fase 1: actualizar todos los nodos interiores
            //
            // schedule(static): divide las M-1 filas en bloques iguales y los
            // asigna a los threads de forma determinista. Óptimo cuando el
            // costo por fila es uniforme (todas tienen el mismo N-1 nodos).
            //
            // reduction(max:delta): cada thread acumula su máximo local de
            // |V_new - V[i][j]|. Al salir, OpenMP combina todos con max()
            // y escribe el resultado en delta (variable shared).
            // -----------------------------------------------------------------
            #pragma omp for reduction(max:delta) schedule(static)
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
            // Barrera implícita del for: delta contiene el máximo global real.
            // Todos los threads están sincronizados aquí.

            // -----------------------------------------------------------------
            // Fase 2: evaluar convergencia y preparar siguiente iteración
            //
            // Un solo thread ejecuta este bloque. Los demás esperan en la
            // barrera implícita al salir del single.
            //
            // Si delta <= TOL: se marca converged = true. NO se resetea delta
            //   (no hay siguiente iteración).
            // Si delta > TOL:  se resetea delta = 0.0 para la siguiente vuelta.
            //   En ambos casos se incrementa el contador de iteraciones.
            // -----------------------------------------------------------------
            #pragma omp single
            {
                ++iterations;
                if (delta <= TOL) {
                    converged = true;   // Señal de salida para todos los threads
                } else {
                    delta = 0.0;        // Reseteo seguro: todos esperan en la
                }                       // barrera antes de leer delta de nuevo
            }
            // Barrera implícita del single: todos ven converged y delta
            // actualizados antes de evaluar while(!converged).
        }
    }
    // Fin de la región paralela: todos los threads han convergido juntos.
}

// -----------------------------------------------------------------------------
// compute_error — igual que actividad 3
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
// export_to_file — igual que actividad 3
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
    file << "# Actividad 4: parallel + for separados + schedule(static)\n";
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
    std::cout << "  Poisson 2D — Ejemplo 3 (Actividad 4)\n";
    std::cout << "  Directiva: parallel + for + schedule(static)\n";
    std::cout << "  ∇²V = 4,  solución exacta: V = (x-y)²\n";
    std::cout << "  Dominio: x ∈ [1,2],  y ∈ [0,2]\n";
    std::cout << "============================================\n";
    std::cout << "Malla: M=" << M << " x N=" << N << "\n";
    std::cout << "Threads disponibles: " << omp_get_max_threads() << "\n";

    allocate_grid(M, N, V, h, k);
    apply_boundary(M, N, V, h, k);

    auto t0 = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double max_error = compute_error(V, M, N, h, k);

    std::cout << "--------------------------------------------\n";
    std::cout << "Tiempo de ejecución        : " << elapsed    << " s\n";
    std::cout << "Número de iteraciones      : " << iterations << "\n";
    std::cout << "Tolerancia (TOL)           : " << TOL        << "\n";
    std::cout << "Error máximo (vs analítica): " << max_error  << "\n";
    std::cout << "Threads usados             : " << omp_get_max_threads() << "\n";
    std::cout << "--------------------------------------------\n";

    export_to_file(V, h, k, M, N, "solucion_static.dat");

    std::cout << "Simulación completada.\n";
    return 0;
}
