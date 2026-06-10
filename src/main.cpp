#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

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
    std::fill(Ax.begin(), Ax.end(), 0.0);

    const double inv_h2 = 1.0 / (h * h);

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

    for (size_t k = 0; k < rhs.size(); ++k) {
        r[k] = rhs[k] - Ap[k];
        p[k] = r[k];
    }

    double dprold = dot_product(r, r);

    for (int iter = 0; iter < max_iter; ++iter) {
        compute_Ax(p, Ap, Nx, Ny, h);

        const double alpha = dprold / dot_product(p, Ap);

        for (size_t k = 0; k < phi.size(); ++k) {
            phi[k] += alpha * p[k];
            r[k] -= alpha * Ap[k];
        }

        const double dprnew = dot_product(r, r);

        if (std::sqrt(dprnew) < tol) {
            return iter + 1;
        }

        const double beta = dprnew / dprold;

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

        for (size_t k = 0; k < residual.size(); ++k) {
            residual[k] = rhs[k] - Aphi[k];
        }

        if (norm(residual) < tol) {
            return iter + 1;
        }
    }

    return max_iter;
}

BenchmarkResult simulation(int N, int max_iter, double tol, double omega)
{
    const int Nx = N;
    const int Ny = N;
    const double pi = std::acos(-1.0);
    const double h = 1.0 / (Nx - 1);

    std::vector<double> phi_exact(Nx * Ny, 0.0);
    std::vector<double> rhs(Nx * Ny, 0.0);
    std::vector<double> phi(Nx * Ny, 0.0);
    std::vector<double> phi_sor(Nx * Ny, 0.0);

    // Set the source term and the exact solution.
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const double x = i * h;
            const double y = j * h;
            const double term1 = std::sin(pi * x) * std::sin(pi * y);
            const double term2 = std::sin(3.0 * pi * x) * std::sin(3.0 * pi * y);

            const double exact = term1 + 0.5 * term2;
            const double source = -2.0 * pi * pi * term1
                                - 0.5 * 18.0 * pi * pi * term2;

            phi_exact[idx(i, j, Nx)] = exact;
            rhs[idx(i, j, Nx)] = source;
        }
    }

    // CG solver
    auto cg_start = std::chrono::high_resolution_clock::now();

    int cg_iterations = CG(phi, rhs, Nx, Ny, h, max_iter, tol);

    auto cg_end = std::chrono::high_resolution_clock::now();

    double cg_time = std::chrono::duration<double>(cg_end - cg_start).count();

    // SOR solver
    auto sor_start = std::chrono::high_resolution_clock::now();

    int sor_iterations = SOR(phi_sor, rhs, Nx, Ny, h, omega, max_iter, tol);

    auto sor_end = std::chrono::high_resolution_clock::now();

    double sor_time = std::chrono::duration<double>(sor_end - sor_start).count();

    // Compute residuals.
    std::vector<double> Aphi_cg(Nx * Ny, 0.0);
    std::vector<double> cg_residual(Nx * Ny, 0.0);
    std::vector<double> Aphi_sor(Nx * Ny, 0.0);
    std::vector<double> sor_residual(Nx * Ny, 0.0);

    compute_Ax(phi, Aphi_cg, Nx, Ny, h);
    compute_Ax(phi_sor, Aphi_sor, Nx, Ny, h);

    for (size_t k = 0; k < cg_residual.size(); ++k) {
        cg_residual[k] = rhs[k] - Aphi_cg[k];
        sor_residual[k] = rhs[k] - Aphi_sor[k];
    }

    // Compute solution errors.
    std::vector<double> CG_error(Nx * Ny, 0.0);
    std::vector<double> SOR_error(Nx * Ny, 0.0);

    for (size_t k = 0; k < CG_error.size(); ++k) {
        CG_error[k] = phi[k] - phi_exact[k];
        SOR_error[k] = phi_sor[k] - phi_exact[k];
    }

    return {
        N,
        Nx * Ny,
        cg_iterations,
        sor_iterations,
        cg_time,
        sor_time,
        norm(cg_residual),
        norm(sor_residual),
        norm(CG_error),
        norm(SOR_error)
    };
}

int main()
{
    const int max_iter = 100000;
    const double tol = 1e-8;
    const double omega = 1.8;
    const std::vector<int> grid_sizes = {32, 64, 128, 256};

    // Output the benchmark results.
    std::filesystem::create_directories("results");
    std::ofstream csv("results/benchmark.csv");

    csv << "N,cells,cg_iterations,cg_time,cg_residual,cg_error,"
        << "sor_iterations,sor_time,sor_residual,sor_error\n";

    std::cout << "CG Poisson Solver Project\n";

    for (int N : grid_sizes) {
        const BenchmarkResult result = simulation(N, max_iter, tol, omega);

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

        // Print the results.
        std::cout << "Size of the grid: " << result.N << " x " << result.N << "\n";
        std::cout << "=============================================" << "\n";
        std::cout << "CG iterations = " << result.cg_iterations << "\n";
        std::cout << "CG time = " << result.cg_time << " s\n";
        std::cout << "CG residual   = " << result.cg_residual << "\n";
        std::cout << "CG solution error = " << result.cg_error << "\n";
        std::cout << "=============================================" << "\n";
        std::cout << "SOR iterations = " << result.sor_iterations << "\n";
        std::cout << "SOR time = " << result.sor_time << " s\n";
        std::cout << "SOR residual   = " << result.sor_residual << "\n";
        std::cout << "SOR solution error = " << result.sor_error << "\n";
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << "\n";
    }

    std::cout << "Benchmark CSV written to results/benchmark.csv\n";

    return 0;
}
