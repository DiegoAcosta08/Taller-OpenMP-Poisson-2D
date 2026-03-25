// =============================================================================
// Simulación de la ecuación de Poisson en 2D — Ejemplo 3
// Actividad 2:Paralelización con parallel for + collapse(2)
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
// Cambios respecto al serial (poisson_serial.cpp):
//   - Se agrega #include <omp.h>
//   - En solve_poisson: se añade #pragma omp parallel for reduction(max:delta)
//     sobre el bucle exterior (i). El bucle interior (j) queda serial dentro
//     de cada hilo.
//   - En main: se reporta el número de threads activos.
//
// Nota sobre el método iterativo:
//   Al paralelizar con parallel for, varios hilos actualizan filas distintas
//   simultáneamente. Esto rompe la dependencia secuencial de Gauss-Seidel:
//   el esquema se convierte implícitamente en uno tipo Jacobi, donde los
//   vecinos leídos pueden ser valores de la iteración anterior o de la actual
//   dependiendo del orden de ejecución de los hilos. Por ello el número de
//   iteraciones puede diferir respecto al serial.
//
// Compilación:
//   g++ -O2 -fopenmp -o poisson_c poisson_collapse.cpp
//
// Ejecución:
//   ./poisson_c
//   OMP_NUM_THREADS=20 ./poisson_c   (controlar número de threads)
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>        // Necesario para directivas y funciones de OpenMP

// -----------------------------------------------------------------------------
// Parámetros del dominio
// Se usan nombres en mayúscula para evitar colisión con la función y0()
// de la librería estándar de C (<math.h>).
// -----------------------------------------------------------------------------
const double X0  = 1.0, XF = 2.0;  // Límites en x
const double Y0  = 0.0, YF = 2.0;  // Límites en y
const double TOL = 1e-6;            // Tolerancia de convergencia

// -----------------------------------------------------------------------------
// initialize_grid
//
// Reserva memoria para la matriz V (solución numérica), define los pasos de
// malla h = Δx y k = Δy, e impone las condiciones de frontera de Dirichlet
// sobre los cuatro bordes del dominio.
//
// Parámetros:
//   M, N   — número de subdivisiones en x e y respectivamente
//   V      — matriz de solución (salida)
//   h, k   — pasos de malla (salida)
// -----------------------------------------------------------------------------
void initialize_grid(int M, int N,
                     std::vector<std::vector<double>>& V,
                     double& h, double& k)
{
    h = (XF - X0) / M;   // Δx
    k = (YF - Y0) / N;   // Δy

    // Inicializar toda la grilla en cero antes de imponer fronteras
    V.assign(M + 1, std::vector<double>(N + 1, 0.0));

    // Borde izquierdo: V(1, y) = (1 - y)²   →  columna i = 0
    for (int j = 0; j <= N; ++j) {
        double y = Y0 + j * k;
        V[0][j]  = (1.0 - y) * (1.0 - y);
    }

    // Borde derecho: V(2, y) = (2 - y)²   →  columna i = M
    for (int j = 0; j <= N; ++j) {
        double y = Y0 + j * k;
        V[M][j]  = (2.0 - y) * (2.0 - y);
    }

    // Borde inferior: V(x, 0) = x²   →  fila j = 0
    for (int i = 0; i <= M; ++i) {
        double x = X0 + i * h;
        V[i][0]  = x * x;
    }

    // Borde superior: V(x, 2) = (x - 2)²   →  fila j = N
    for (int i = 0; i <= M; ++i) {
        double x = X0 + i * h;
        V[i][N]  = (x - 2.0) * (x - 2.0);
    }
}

// -----------------------------------------------------------------------------
// solve_poisson  —  VERSIÓN ACTIVIDAD 2: parallel for + collapse + reduction
//
// Cambio clave respecto al serial:
//   Se agrega la directiva:
//     #pragma omp parallel for collapse(2) reduction (max:delta)
//   sobre el bucle exterior (i).
//
// ¿Por qué reduction(max:delta)?
//   Cada hilo calcula su propio máximo local de |V_new - V[i][j]| sobre las
//   filas que le corresponden. Sin reduction, todos los hilos escribirían sobre
//   la misma variable 'delta' simultáneamente → condición de carrera (race
//   condition), resultado indeterminado. Con reduction(max:delta), OpenMP crea
//   una copia privada de 'delta' por hilo, cada uno actualiza la suya sin
//   interferencia, y al final del bucle hace un max() global entre todas las
//   copias y lo almacena en 'delta'. Así se garantiza corrección y seguridad.
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
    const double rhs_term = 4.0 * h2 * k2;   // f * h² * k²  con f = 4

    double delta = 1.0;
    iterations   = 0;

    while (delta > TOL) {
        delta = 0.0;

        // -----------------------------------------------------------------
        // DIRECTIVA ACTIVIDAD 2:
        //   #pragma omp parallel for collapse(2) reduction(max:delta)
        //
        // - parallel for: divide las iteraciones del bucle de i entre los
        //   threads disponibles. Cada thread procesa un subconjunto de filas.
        // - reduction(max:delta): cada thread mantiene su propio delta local;
        //   al terminar el bucle, OpenMP combina todos con max() y escribe
        //   el resultado en delta.
        // - El bucle interior (j) es serial dentro de cada thread.
        // -----------------------------------------------------------------
        #pragma omp parallel for collapse(2) reduction(max:delta)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {

                // Esquema de diferencias finitas de 5 puntos para ∇²V = 4
                double V_new = (  (V[i + 1][j] + V[i - 1][j]) * k2
                                + (V[i][j + 1] + V[i][j - 1]) * h2
                                - rhs_term )
                               / denom;

                delta = std::max(delta, std::abs(V_new - V[i][j]));
                V[i][j] = V_new;
            }
        }
        // Al llegar aquí, OpenMP ya realizó la barrera implícita y el
        // reduction: 'delta' contiene el máximo global entre todos los hilos.

        ++iterations;
    }
}

// -----------------------------------------------------------------------------
// compute_error
//
// Calcula el error máximo absoluto entre la solución numérica V y la
// solución analítica exacta V_exact(x, y) = (x - y)².
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
            double error   = std::abs(V[i][j] - V_exact);
            max_error      = std::max(max_error, error);
        }
    }

    return max_error;
}

// -----------------------------------------------------------------------------
// export_to_file
//
// Exporta la solución numérica a un archivo .dat con cinco columnas:
//   x    y    V_num    V_exact    error
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
    file << "# Actividad 1: parallel for + reduction(max:delta)\n";
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

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main()
{
    int M = 1024, N = 1024;
    double h, k;
    int iterations = 0;

    std::vector<std::vector<double>> V;

    // Reportar configuración de threads ANTES de resolver
    // omp_get_max_threads(): número máximo de threads que usará OpenMP,
    // según OMP_NUM_THREADS o el número de núcleos del sistema.
    int max_threads = omp_get_max_threads();

    std::cout << "============================================\n";
    std::cout << "  Poisson 2D — Ejemplo 3 (Actividad 2)\n";
    std::cout << "  Directiva: parallel for + collapse(2)\n";
    std::cout << "  ∇²V = 4,  solución exacta: V = (x-y)²\n";
    std::cout << "  Dominio: x ∈ [1,2],  y ∈ [0,2]\n";
    std::cout << "============================================\n";
    std::cout << "Malla: M=" << M << " x N=" << N
              << "  (h=" << (XF - X0) / M
              << ", k="  << (YF - Y0) / N << ")\n";
    std::cout << "Threads disponibles (omp_get_max_threads): "
              << max_threads << "\n";

    // Inicializar grilla y condiciones de frontera
    initialize_grid(M, N, V, h, k);

    // Resolver y medir tiempo
    auto t_start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, iterations);
    auto t_end   = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // Error respecto a la solución analítica
    double max_error = compute_error(V, M, N, h, k);

    // Reporte final
    std::cout << "--------------------------------------------\n";
    std::cout << "Tiempo de ejecución        : " << elapsed    << " s\n";
    std::cout << "Número de iteraciones      : " << iterations << "\n";
    std::cout << "Tolerancia (TOL)           : " << TOL        << "\n";
    std::cout << "Error máximo (vs analítica): " << max_error  << "\n";
    std::cout << "Threads usados             : " << max_threads << "\n";
    std::cout << "--------------------------------------------\n";

    export_to_file(V, h, k, M, N, "solucion_collapse.dat");

    std::cout << "Simulación completada.\n";
    return 0;
}
