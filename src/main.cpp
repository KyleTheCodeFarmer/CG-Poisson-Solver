#include <algorithm>
#include <cmath>
#include <iomanip>
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

            Ax[k] = (4.0 * x[k]
                     - x[idx(i + 1, j, Nx)]
                     - x[idx(i - 1, j, Nx)]
                     - x[idx(i, j + 1, Nx)]
                     - x[idx(i, j - 1, Nx)]) * inv_h2;
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

int main()
{
    const int Nx = 256;
    const int Ny = 256;
    const double pi = std::acos(-1.0);
    const double h = 1.0 / (Nx - 1);

    std::vector<double> phi_exact(Nx * Ny, 0.0);
    std::vector<double> rhs(Nx * Ny, 0.0);
    std::vector<double> phi(Nx * Ny, 0.0);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const double x = i * h;
            const double y = j * h;
            const double exact = std::sin(pi * x) * std::sin(pi * y);

            phi_exact[idx(i, j, Nx)] = exact;
            rhs[idx(i, j, Nx)] = 2.0 * pi * pi * exact;
        }
    }

    const int max_iter = 10000;
    const double tol = 1e-8;

    int cg_iterations = CG(phi, rhs, Nx, Ny, h, max_iter, tol);

    std::vector<double> Aphi(Nx * Ny, 0.0);
    std::vector<double> cg_residual(Nx * Ny, 0.0);

    compute_Ax(phi, Aphi, Nx, Ny, h);

    for (size_t k = 0; k < cg_residual.size(); ++k) {
        cg_residual[k] = rhs[k] - Aphi[k];
    }

    std::vector<double> CG_error(Nx * Ny, 0.0);

    for (size_t k = 0; k < CG_error.size(); ++k) {
        CG_error[k] = phi[k] - phi_exact[k];
    }


    //print the results
    std::cout << "CG Poisson Solver Project\n";
    
    std::cout << "CG iterations = " << cg_iterations << "\n";
    std::cout << "CG residual   = " << norm(cg_residual) << "\n";
    std::cout << "CG solution error = " << norm(CG_error) << "\n";

  

    return 0;
}
