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

int main()
{
    const int Nx = 256;
    const int Ny = 256;
    const double pi = std::acos(-1.0);
    const double h = 1.0 / (Nx - 1);

    std::vector<double> phi_exact(Nx * Ny, 0.0);
    std::vector<double> rhs(Nx * Ny, 0.0);
    std::vector<double> phi(Nx * Ny, 0.0);
    std::vector<double> Aphi_exact(Nx * Ny, 0.0);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const double x = i * h;
            const double y = j * h;
            const double exact = std::sin(pi * x) * std::sin(pi * y);

            phi_exact[idx(i, j, Nx)] = exact;
            rhs[idx(i, j, Nx)] = 2.0 * pi * pi * exact;
        }
    }

    compute_Ax(phi_exact, Aphi_exact, Nx, Ny, h);

    const int center_i = Nx / 2;
    const int center_j = Ny / 2;
    const int center = idx(center_i, center_j, Nx);

    std::cout << "CG Poisson Solver Project\n";
    std::cout << "phi_exact(center) = " << phi_exact[center] << "\n";
    std::cout << "rhs(center)       = " << rhs[center] << "\n";
    std::cout << "Aphi_exact(center)= " << Aphi_exact[center] << "\n";

    return 0;
}
