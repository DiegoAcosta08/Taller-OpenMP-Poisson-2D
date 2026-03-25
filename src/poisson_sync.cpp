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
// allocate_grid
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
// solve_poisson — Actividad 5 mejorada
// -----------------------------------------------------------------------------
void solve_poisson(std::vector<std::vector<double>>& V,
                   int M, int N,
                   double h, double k,
                   int& iterations,
                   double& norma_L1)
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

            // -------- PRINT CONTROLADO (menos overhead) --------
            #pragma omp single
            {
                if (iterations < 20 || iterations % 5000 == 0) {
                    std::cout << "  Iter " << iterations + 1
                              << "  |  delta = "
                              << std::scientific << std::setprecision(6)
                              << delta << "\n";
                }
            }

            // -------- ACTUALIZACIÓN PRINCIPAL --------
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

            // -------- BUCLE INDEPENDIENTE (nowait) --------
            double aporte_local = 0.0;

            #pragma omp for nowait schedule(static)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    aporte_local += std::abs(V[i][j]);
                }
            }

            // -------- ACUMULACIÓN (critical) --------
            #pragma omp critical
            {
                norma_L1 += aporte_local;
            }

            // -------- CONVERGENCIA --------
            #pragma omp single
            {
                ++iterations;

                if (delta <= TOL) {
                    converged = true;

                    std::cout << "  Iter " << iterations
                              << "  |  delta final = "
                              << std::scientific << std::setprecision(6)
                              << delta << "\n";
                } else {
                    norma_L1 = 0.0;
                    delta    = 0.0;
                }
            }
        }
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
            double x = X0 + i * h;
            double y = Y0 + j * k;
            double V_exact = (x - y) * (x - y);

            max_error = std::max(max_error,
                                 std::abs(V[i][j] - V_exact));
        }
    }
    return max_error;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main()
{
    int M = 1024, N = 1024;
    double h, k;
    int    iterations = 0;
    double norma_L1   = 0.0;

    std::vector<std::vector<double>> V;

    std::cout << "Poisson 2D — Actividad 5\n";
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    allocate_grid(M, N, V, h, k);
    apply_boundary(M, N, V, h, k);

    auto t0 = std::chrono::high_resolution_clock::now();
    solve_poisson(V, M, N, h, k, iterations, norma_L1);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double max_error = compute_error(V, M, N, h, k);

    std::cout << "----------------------------------\n";
    std::cout << "Tiempo        : " << elapsed    << " s\n";
    std::cout << "Iteraciones   : " << iterations << "\n";
    std::cout << "Error máximo  : " << max_error  << "\n";
    std::cout << "Norma L1      : " << norma_L1   << "\n";
    std::cout << "----------------------------------\n";

    return 0;
}
