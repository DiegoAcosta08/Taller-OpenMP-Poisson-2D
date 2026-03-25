// =============================================================================
// Simulación de la ecuación de Poisson en 2D — Ejemplo 3
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
// Nota sobre las condiciones de frontera:
//   La tabla del enunciado indica V(1,y) = y² y V(2,y) = (y-1)², pero estas
//   no son consistentes con la solución analítica V(x,y) = (x-y)².
//   Al evaluar la solución exacta en los bordes se obtiene:
//     V(1,y) = (1-y)²  y  V(2,y) = (2-y)²
//   Se usan estas formas corregidas para garantizar coherencia matemática.
//
// Solución analítica exacta: V(x, y) = (x - y)²
//
// Método numérico: diferencias finitas centradas (esquema de 5 puntos)
// Iteración: Gauss-Seidel in-place hasta convergencia (delta < TOL)
//
// Fórmula de actualización (despejando V[i][j] del esquema de 5 puntos):
//   V[i][j] = ( (V[i+1][j] + V[i-1][j]) * k²
//             + (V[i][j+1] + V[i][j-1]) * h²
//             -  4 * h² * k² )
//             / ( 2 * (h² + k²) )
//
// El término -4*h²*k² proviene del lado derecho f = 4 de la ecuación.
//
// Compilación:
//   g++ -O2 -o poisson_serial poisson_serial.cpp
//
// Ejecución:
//   ./poisson_serial
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>       // Para medir tiempo de ejecución

// -----------------------------------------------------------------------------
// Parámetros del dominio
// Se usan nombres en mayúscula para evitar colisión con la función y0()
// de la librería estándar de C (<math.h>).
// -----------------------------------------------------------------------------
const double X0 = 1.0, XF = 2.0;   // Límites en x
const double Y0 = 0.0, YF = 2.0;   // Límites en y
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
    // Pasos de discretización
    h = (XF - X0) / M;   // Δx
    k = (YF - Y0) / N;   // Δy

    // Inicializar toda la grilla en cero antes de imponer fronteras
    V.assign(M + 1, std::vector<double>(N + 1, 0.0));

    // -------------------------------------------------------------------------
    // Condiciones de frontera (Dirichlet)
    // -------------------------------------------------------------------------

    // Borde izquierdo: V(1, y) = (1 - y)²   →  columna i = 0
    // Verificación: V(x,y) = (x-y)² evaluada en x=1 → (1-y)²
    for (int j = 0; j <= N; ++j) {
        double y = Y0 + j * k;
        V[0][j]  = (1.0 - y) * (1.0 - y);
    }

    // Borde derecho: V(2, y) = (2 - y)²   →  columna i = M
    // Verificación: V(x,y) = (x-y)² evaluada en x=2 → (2-y)²
    for (int j = 0; j <= N; ++j) {
        double y = Y0 + j * k;
        V[M][j]  = (2.0 - y) * (2.0 - y);
    }

    // Borde inferior: V(x, 0) = x²   →  fila j = 0
    // Verificación: V(x,y) = (x-y)² evaluada en y=0 → x²  ✓
    for (int i = 0; i <= M; ++i) {
        double x = X0 + i * h;
        V[i][0]  = x * x;
    }

    // Borde superior: V(x, 2) = (x - 2)²   →  fila j = N
    // Verificación: V(x,y) = (x-y)² evaluada en y=2 → (x-2)²  ✓
    for (int i = 0; i <= M; ++i) {
        double x = X0 + i * h;
        V[i][N]  = (x - 2.0) * (x - 2.0);
    }
}

// -----------------------------------------------------------------------------
// solve_poisson
//
// Resuelve ∇²V = 4 iterativamente usando Gauss-Seidel con el esquema de
// diferencias finitas centradas de 5 puntos, hasta que la variación máxima
// entre iteraciones sea menor que TOL.
//
// Fórmula de actualización para cada nodo interior (i,j):
//
//   V[i][j] = [ (V[i+1][j] + V[i-1][j]) * k²
//             + (V[i][j+1] + V[i][j-1]) * h²
//             - 4 * h² * k² ]
//             / [ 2 * (h² + k²) ]
//
// El término -4*h²*k² viene de discretizar el lado derecho f = 4:
//   f * h² * k² = 4 * h² * k²
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
    const double h2       = h * h;             // h²
    const double k2       = k * k;             // k²
    const double denom    = 2.0 * (h2 + k2);   // 2(h² + k²)
    const double rhs_term = 4.0 * h2 * k2;     // f * h² * k²  con f = 4

    double delta = 1.0;
    iterations   = 0;

    while (delta > TOL) {
        delta = 0.0;

        // Recorrer únicamente los nodos interiores.
        // Los bordes (i=0, i=M, j=0, j=N) tienen valores fijos y no se tocan.
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {

                // Esquema de diferencias finitas de 5 puntos para ∇²V = 4
                double V_new = (  (V[i + 1][j] + V[i - 1][j]) * k2
                                + (V[i][j + 1] + V[i][j - 1]) * h2
                                - rhs_term )
                               / denom;

                // Registrar la mayor variación en esta iteración (criterio de paro)
                delta = std::max(delta, std::abs(V_new - V[i][j]));

                // Actualizar el valor en la grilla (Gauss-Seidel in-place:
                // los vecinos ya actualizados en esta misma iteración se reusan)
                V[i][j] = V_new;
            }
        }

        ++iterations;
    }
}

// -----------------------------------------------------------------------------
// compute_error
//
// Calcula el error máximo absoluto entre la solución numérica V y la
// solución analítica exacta V_exact(x, y) = (x - y)².
// Recorre todos los nodos de la grilla, incluyendo los de frontera.
//
// Retorna: error máximo absoluto sobre toda la grilla.
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
            double V_exact = (x - y) * (x - y);        // Solución analítica
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
//
// El formato incluye una línea en blanco entre bloques de i, compatible
// con gnuplot para graficar superficies con 'splot'.
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

    // Encabezado descriptivo
    file << "# Solución numérica de ∇²V = 4  en  [1,2] x [0,2]\n";
    file << "# Solución analítica: V(x,y) = (x-y)²\n";
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
        file << "\n";   // Línea en blanco entre bloques (formato gnuplot)
    }

    file.close();
    std::cout << "Resultados exportados a: " << filename << std::endl;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main()
{
    // -------------------------------------------------------------------------
    // Parámetros de la malla
    // M = 50 divisiones en x  (dominio de longitud 1: x ∈ [1,2])
    // N = 100 divisiones en y (dominio de longitud 2: y ∈ [0,2])
    // → h = k = 0.02 (malla uniforme en ambas direcciones)
    // -------------------------------------------------------------------------
    int M = 1024, N = 1024;
    double h, k;
    int iterations = 0;

    std::vector<std::vector<double>> V;

    std::cout << "============================================\n";
    std::cout << "  Poisson 2D — Ejemplo 3 (Serial)\n";
    std::cout << "  ∇²V = 4,  solución exacta: V = (x-y)²\n";
    std::cout << "  Dominio: x ∈ [1,2],  y ∈ [0,2]\n";
    std::cout << "============================================\n";
    std::cout << "Malla: M=" << M << " x N=" << N
              << "  (h=" << (XF - X0) / M
              << ", k="  << (YF - Y0) / N << ")\n";

    // Paso 1: inicializar grilla y condiciones de frontera
    initialize_grid(M, N, V, h, k);

    // Paso 2: resolver la ecuación de Poisson y medir tiempo
    auto t_start = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, iterations);
    auto t_end   = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // Paso 3: calcular error respecto a la solución analítica
    double max_error = compute_error(V, M, N, h, k);

    // Reporte en consola
    std::cout << "--------------------------------------------\n";
    std::cout << "Tiempo de ejecución        : " << elapsed    << " s\n";
    std::cout << "Número de iteraciones      : " << iterations << "\n";
    std::cout << "Tolerancia (TOL)           : " << TOL        << "\n";
    std::cout << "Error máximo (vs analítica): " << max_error  << "\n";
    std::cout << "Número de threads          : 1 (secuencial)\n";
    std::cout << "--------------------------------------------\n";

    // Paso 4: exportar resultados a archivo
    export_to_file(V, h, k, M, N, "solucion_serial.dat");

    std::cout << "Simulación completada.\n";
    return 0;
}
