#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

int idx(int i, int j, int Nx)
{
    return j * Nx + i;
}

void compute_Ax(const std::vector<double>& x,
                std::vector<double>& Ax,
                int Nx,
                int Ny,
                double h)
{
    #pragma omp parallel for
    for (size_t k = 0; k < Ax.size(); ++k) {
        Ax[k] = 0.0;
    }

    const double inv_h2 = 1.0 / (h * h);

    #pragma omp parallel for collapse(2)
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            const int k = idx(i, j, Nx);

            Ax[k] = (x[idx(i + 1, j, Nx)]
                     + x[idx(i - 1, j, Nx)]
                     + x[idx(i, j + 1, Nx)]
                     + x[idx(i, j - 1, Nx)]
                     - 4.0 * x[k]) * inv_h2;
        }
    }
}

double dot_product(const std::vector<double>& a,
                   const std::vector<double>& b)
{
    double result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

double norm(const std::vector<double>& v)
{
    return std::sqrt(dot_product(v, v));
}

struct BenchmarkResult {
    int N;
    int cells;
    int cg_iterations;
    int sor_iterations;
    double cg_time;
    double sor_time;
    double cg_residual;
    double sor_residual;
    double cg_error;
    double sor_error;
};

struct CGThreadResult {
    int threads;
    int N;
    int cells;
    int iterations;
    double time;
    double residual;
    double error;
    double speedup;
};

struct UnknownSourceResult {
    int N;
    int cells;
    int iterations;
    double time;
    double residual;
};

// Initializes the exact solution and right-hand side for the Poisson problem.
void initialize_problem(std::vector<double>& phi_exact,
                        std::vector<double>& rhs,
                        int Nx,
                        int Ny,
                        double h)
{
    const double alpha = 100.0;

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const int k = idx(i, j, Nx);

            // Boundary conditions: phi = 0 on the boundary.
            if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
                phi_exact[k] = 0.0;
                continue;
            }

            const double x = i * h;
            const double y = j * h;

            const double dx1 = x - 0.65;
            const double dy1 = y - 0.24;
            const double r1_squared = dx1 * dx1 + dy1 * dy1;
            const double gauss1 = std::exp(-alpha * r1_squared);

            const double dx2 = x - 0.13;
            const double dy2 = y - 0.90;
            const double r2_squared = dx2 * dx2 + dy2 * dy2;
            const double gauss2 = std::exp(-alpha * r2_squared);

            const double bx = x * (1.0 - x);
            const double by = y * (1.0 - y);

            phi_exact[k] = bx * by * (gauss1 - 0.8 * gauss2);
        }
    }

    // Compute the rhs by applying the Laplacian to the exact solution.
    compute_Ax(phi_exact, rhs, Nx, Ny, h);
}

void initialize_unknown_source(std::vector<double>& rhs,
                               int Nx,
                               int Ny,
                               double h)
{
    const double pi = std::acos(-1.0);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const int k = idx(i, j, Nx);

            if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
                rhs[k] = 0.0;
                continue;
            }

            const double x = i * h;
            const double y = j * h;

            const double bg_sinemode =
                0.45 * std::sin(2.0 * pi * x + 0.2) * std::sin(3.0 * pi * y)
              + 0.30 * std::cos(4.0 * pi * x + 0.5) * std::sin(2.0 * pi * y + 0.3)
              - 0.25 * std::sin(5.0 * pi * x + 0.9) * std::cos(4.0 * pi * y + 0.6);

            const double large_dx = x - 0.34;
            const double large_dy = y - 0.68;
            const double large_r2 = large_dx * large_dx + large_dy * large_dy;
            const double large_gaussian = std::exp(-30.0 * large_r2);

            const double small_dx = x - 0.73;
            const double small_dy = y - 0.31;
            const double small_r2 = small_dx * small_dx + small_dy * small_dy;
            const double small_gaussian = std::exp(-220.0 * small_r2);

            rhs[k] = bg_sinemode
                   + 1.25 * large_gaussian
                   - 0.85 * small_gaussian;
        }
    }
}

void write_field_csv(const std::string& path,
                     const std::vector<double>& field,
                     int Nx,
                     int Ny)
{
    std::ofstream file(path);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            if (i > 0) {
                file << ",";
            }
            file << field[idx(i, j, Nx)];
        }
        file << "\n";
    }
}

int CG(std::vector<double>& phi,
        const std::vector<double>& rhs,
        int Nx,
        int Ny,
        double h,
        int max_iter,
        double tol)
{
    std::vector<double> r(rhs.size(), 0.0);
    std::vector<double> p(rhs.size(), 0.0);
    std::vector<double> Ap(rhs.size(), 0.0);

    compute_Ax(phi, Ap, Nx, Ny, h);

    #pragma omp parallel for
    for (size_t k = 0; k < rhs.size(); ++k) {
        // For CG, solve (-A)phi = -rhs, so r = (-rhs) - (-Aphi) = Aphi - rhs.
        r[k] = Ap[k] - rhs[k];
        p[k] = r[k];
    }

    double dprold = dot_product(r, r);

    for (int iter = 0; iter < max_iter; ++iter) {
        compute_Ax(p, Ap, Nx, Ny, h);

        const double alpha = dprold / (-dot_product(p, Ap));

        #pragma omp parallel for
        for (size_t k = 0; k < phi.size(); ++k) {
            phi[k] += alpha * p[k];
            r[k] += alpha * Ap[k];
        }

        const double dprnew = dot_product(r, r);

        if (std::sqrt(dprnew) < tol) {
            return iter + 1;
        }

        const double beta = dprnew / dprold;

        #pragma omp parallel for
        for (size_t k = 0; k < p.size(); ++k) {
            p[k] = r[k] + beta * p[k];
        }

        dprold = dprnew;
    }

    return max_iter;
}

int SOR(std::vector<double>& phi,
        const std::vector<double>& rhs,
        int Nx,
        int Ny,
        double h,
        double omega,
        int max_iter,
        double tol)
{
    std::vector<double> Aphi(rhs.size(), 0.0);
    std::vector<double> residual(rhs.size(), 0.0);

    const double h2 = h * h;

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                const int k = idx(i, j, Nx);
                const double old_phi = phi[k];

                const double gs_value = 0.25 * (
                    phi[idx(i + 1, j, Nx)]
                    + phi[idx(i - 1, j, Nx)]
                    + phi[idx(i, j + 1, Nx)]
                    + phi[idx(i, j - 1, Nx)]
                    - h2 * rhs[k]);

                phi[k] = (1.0 - omega) * old_phi + omega * gs_value;
            }
        }

        compute_Ax(phi, Aphi, Nx, Ny, h);

        #pragma omp parallel for
        for (size_t k = 0; k < residual.size(); ++k) {
            residual[k] = rhs[k] - Aphi[k];
        }

        if (norm(residual) < tol) {
            return iter + 1;
        }
    }

    return max_iter;
}

// Runs the solvers for a given grid size and returns the benchmark results.
BenchmarkResult simulation(int N, int max_iter, double tol, double omega, bool run_sor)
{
    const int Nx = N;
    const int Ny = N;
    const double h = 1.0 / (Nx - 1);

    std::vector<double> phi_exact(Nx * Ny, 0.0);
    std::vector<double> rhs(Nx * Ny, 0.0);
    std::vector<double> phi(Nx * Ny, 0.0);
    std::vector<double> phi_sor(Nx * Ny, 0.0);

    initialize_problem(phi_exact, rhs, Nx, Ny, h);

    // CG solver
    auto cg_start = std::chrono::high_resolution_clock::now();

    int cg_iterations = CG(phi, rhs, Nx, Ny, h, max_iter, tol);

    auto cg_end = std::chrono::high_resolution_clock::now();

    double cg_time = std::chrono::duration<double>(cg_end - cg_start).count();

    int sor_iterations = 0;
    double sor_time = 0.0;

    if (run_sor) {
        // SOR solver
        auto sor_start = std::chrono::high_resolution_clock::now();

        sor_iterations = SOR(phi_sor, rhs, Nx, Ny, h, omega, max_iter, tol);

        auto sor_end = std::chrono::high_resolution_clock::now();

        sor_time = std::chrono::duration<double>(sor_end - sor_start).count();
    }

    // Compute residuals.
    std::vector<double> Aphi_cg(Nx * Ny, 0.0);
    std::vector<double> cg_residual(Nx * Ny, 0.0);
    std::vector<double> Aphi_sor(Nx * Ny, 0.0);
    std::vector<double> sor_residual(Nx * Ny, 0.0);

    compute_Ax(phi, Aphi_cg, Nx, Ny, h);
    if (run_sor) {
        compute_Ax(phi_sor, Aphi_sor, Nx, Ny, h);
    }

    #pragma omp parallel for
    for (size_t k = 0; k < cg_residual.size(); ++k) {
        cg_residual[k] = rhs[k] - Aphi_cg[k];
        if (run_sor) {
            sor_residual[k] = rhs[k] - Aphi_sor[k];
        }
    }

    // Compute solution errors.
    std::vector<double> CG_error(Nx * Ny, 0.0);
    std::vector<double> SOR_error(Nx * Ny, 0.0);

    #pragma omp parallel for
    for (size_t k = 0; k < CG_error.size(); ++k) {
        CG_error[k] = phi[k] - phi_exact[k];
        if (run_sor) {
            SOR_error[k] = phi_sor[k] - phi_exact[k];
        }
    }

    return {
        N,
        Nx * Ny,
        cg_iterations,
        sor_iterations,
        cg_time,
        sor_time,
        norm(cg_residual),
        run_sor ? norm(sor_residual) : 0.0,
        norm(CG_error),
        run_sor ? norm(SOR_error) : 0.0
    };
}

// Runs the CG solver with a specified number of threads and returns the benchmark results.
CGThreadResult cg_thread_simulation(int threads,
                                    int N,
                                    int max_iter,
                                    double tol,
                                    double baseline_time)
{
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    const int Nx = N;
    const int Ny = N;
    const double h = 1.0 / (Nx - 1);

    std::vector<double> phi_exact(Nx * Ny, 0.0);
    std::vector<double> rhs(Nx * Ny, 0.0);
    std::vector<double> phi(Nx * Ny, 0.0);

    initialize_problem(phi_exact, rhs, Nx, Ny, h);

    // CG solver
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = CG(phi, rhs, Nx, Ny, h, max_iter, tol);
    auto end = std::chrono::high_resolution_clock::now();
    const double time = std::chrono::duration<double>(end - start).count();

    // Compute residuals and errors.
    std::vector<double> Aphi(Nx * Ny, 0.0);
    std::vector<double> residual(Nx * Ny, 0.0);
    std::vector<double> error(Nx * Ny, 0.0);

    compute_Ax(phi, Aphi, Nx, Ny, h);

    #pragma omp parallel for
    for (size_t k = 0; k < residual.size(); ++k) {
        residual[k] = rhs[k] - Aphi[k];
        error[k] = phi[k] - phi_exact[k];
    }

    const double speedup = baseline_time > 0.0 ? baseline_time / time : 1.0;

    return {
        threads,
        N,
        Nx * Ny,
        iterations,
        time,
        norm(residual),
        norm(error),
        speedup
    };
}

// Runs a CG simulation with an unknown source term and writes the results to CSV files.
UnknownSourceResult unknown_source_simulation(int N,
                                              int max_iter,
                                              double tol)
{
    const int Nx = N;
    const int Ny = N;
    const double h = 1.0 / (Nx - 1);

    std::vector<double> rhs(Nx * Ny, 0.0);
    std::vector<double> phi(Nx * Ny, 0.0);

    initialize_unknown_source(rhs, Nx, Ny, h);

    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = CG(phi, rhs, Nx, Ny, h, max_iter, tol);
    auto end = std::chrono::high_resolution_clock::now();
    const double time = std::chrono::duration<double>(end - start).count();

    std::vector<double> Aphi(Nx * Ny, 0.0);
    std::vector<double> residual(Nx * Ny, 0.0);

    compute_Ax(phi, Aphi, Nx, Ny, h);

    #pragma omp parallel for
    for (size_t k = 0; k < residual.size(); ++k) {
        residual[k] = rhs[k] - Aphi[k];
    }

    write_field_csv("results/unknown_source.csv", rhs, Nx, Ny);
    write_field_csv("results/unknown_solution.csv", phi, Nx, Ny);

    return {
        N,
        Nx * Ny,
        iterations,
        time,
        norm(residual)
    };
}

int main()
{
    // basic parameters
    const int max_iter = 100000;
    const double tol = 1e-8;
    const double omega = 1.8;
    const bool run_sor = true;  // Set whether to run SOR in the grid scaling benchmark(it would increase the computation time).
    const std::vector<int> grid_sizes = {32, 64, 128, 256};

    // Output the results.
    std::filesystem::create_directories("results");
    std::ofstream csv("results/benchmark.csv");
    std::ofstream thread_csv("results/thread_scaling.csv");

    csv << "N,cells,cg_iterations,cg_time,cg_residual,cg_error,"
        << "sor_iterations,sor_time,sor_residual,sor_error\n";

    thread_csv << "threads,N,cells,cg_iterations,cg_time,cg_residual,"
               << "cg_error,cg_speedup\n";

    std::cout << "CG Poisson Solver Project\n";
#ifdef _OPENMP
    std::cout << "OpenMP enabled with max threads = " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not enabled; running serial fallback.\n";
#endif

    // Run the grid scaling benchmark.
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    std::cout << "Grid scaling benchmark uses 1 thread.\n";

    for (int N : grid_sizes) {
        const BenchmarkResult result = simulation(N, max_iter, tol, omega, run_sor);

        csv << result.N << ","
            << result.cells << ","
            << result.cg_iterations << ","
            << result.cg_time << ","
            << result.cg_residual << ","
            << result.cg_error << ","
            << result.sor_iterations << ","
            << result.sor_time << ","
            << result.sor_residual << ","
            << result.sor_error << "\n";

        std::cout << "  N = " << result.N << " done\n";
    }

    std::cout << "Benchmark CSV written to results/benchmark.csv\n";

    // Run the thread scaling benchmark at a fixed grid size.
    const int thread_scaling_N = 1024;
    const std::vector<int> thread_counts = {1, 2, 4, 8};
    double baseline_time = 0.0;

    std::cout << "CG OpenMP thread scaling at N = " << thread_scaling_N << "\n";

    for (int threads : thread_counts) {
        CGThreadResult result = cg_thread_simulation(
            threads,
            thread_scaling_N,
            max_iter,
            tol,
            baseline_time
        );

        if (threads == 1) {
            baseline_time = result.time;
            result.speedup = 1.0;
        }

        thread_csv << result.threads << ","
                   << result.N << ","
                   << result.cells << ","
                   << result.iterations << ","
                   << result.time << ","
                   << result.residual << ","
                   << result.error << ","
                   << result.speedup << "\n";

        std::cout << "  threads = " << result.threads << " done\n";
    }

    std::cout << "Thread scaling CSV written to results/thread_scaling.csv\n";

    // Run a final unknown-source CG demo.
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    const int unknown_source_N = 1024;
    const UnknownSourceResult unknown_result = unknown_source_simulation(
        unknown_source_N,
        max_iter,
        tol
    );

    std::ofstream unknown_csv("results/unknown_summary.csv");
    unknown_csv << "N,cells,cg_iterations,cg_time,cg_residual\n";
    unknown_csv << unknown_result.N << ","
                << unknown_result.cells << ","
                << unknown_result.iterations << ","
                << unknown_result.time << ","
                << unknown_result.residual << "\n";

    std::cout << "Unknown source CSV files written to results/unknown_summary.csv\n";

    return 0;
}
