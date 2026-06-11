# CG-Poisson-Solver
2D Poisson Equation Solver using Conjugate Gradient and SOR Methods with OpenMP Parallelization

## Development Log

### Setup Grid and Matrix-Free Operator (`compute_Ax`)

- Created a 2D grid on the domain `[0, 1] x [0, 1]`.
- Used `idx(i, j, Nx)` to map 2D indices to 1D array indices.
- Implemented `compute_Ax()`.
- This computes `A*x` without explicitly constructing the matrix `A`.
- The operator corresponds to the finite-difference approximation of `laplacian`.
- To check this operator, I used the example `phi_exact = sin(pi x) sin(pi y)`.
- Verified the implementation by comparing `A*phi_exact` with `rhs`.

For the center-point check, I obtained:

```text
rhs(center)        = -19.7385
Aphi_exact(center) = -19.7382
```

### Global Residual Check

- To check the accuracy of `compute_Ax()` after finite-difference discretization, I computed the global residual:

```text
residual = rhs - A*phi_exact
```

- This checks how well the discrete operator approximates `laplacian(phi_exact)` over the whole grid.

- To compute the residual norm, I added two helper functions: `dot_product()` and `norm()`.

### Basic CG Solver

- Add a simple Conjugate Gradient solver for the linear system:

```text
A*phi = rhs
```

- To check the CG solver, I computed two quantities:

```text
CG residual = rhs - A*phi
phi_error   = phi - phi_exact
```

- The CG residual checks whether the numerical solution satisfies the discrete linear system.
- The phi error checks whether the numerical solution is close to the exact solution.

For the current test case, I obtained:

```text
CG iterations = 1
CG residual   = 4.14421e-09
CG solution error = 0.00161269
```

- The CG method converges in one iteration for this simple test case because `sin(pi x) sin(pi y)` is an eigenmode of the discrete Laplacian operator.

### Basic SOR Solver

- Add a simple SOR solver to compare with CG.
- I used `omega = 1.8` for the relaxation factor.
- To check the SOR solver, I computed the residual and solution error.

For `N = 64`, I obtained:

```text
SOR iterations = 1061
SOR residual   = 9.95666e-09
SOR solution error = 0.00652833
```

### Timing Test (step 1)

- Add wall-clock timing for both CG and SOR.
- I ran the simulation again using a `64 x 64` grid.

```text
CG iterations = 1
CG time = 2.7083e-05 s
CG residual   = 6.72534e-11
CG solution error = 0.00652833

SOR iterations = 1061
SOR time = 0.0372997 s
SOR residual   = 9.95666e-09
SOR solution error = 0.00652833
```

- Since the current test case uses a simple source term from `phi_exact = sin(pi x) sin(pi y)`, CG converges in only one iteration.
- Next, I tested a more complicated source term to produce a more meaningful comparison.

### Timing Test (step 2)

- I changed the exact solution to:

```text
phi_exact = sin(pi x) sin(pi y) + 0.5 sin(3pi x) sin(3pi y)
```

- This gives a more complicated `rhs`, so CG needs more iterations.
- I ran the simulation using a `64 x 64` grid.

```text
CG iterations = 4
CG time = 6.6375e-05 s
CG residual   = 7.67349e-09
CG solution error = 0.0301227

SOR iterations = 1061
SOR time = 0.0380466 s
SOR residual   = 9.96279e-09
SOR solution error = 0.0301227
```

- The CG and SOR solution errors are almost the same because both methods solve the same discrete system.

### Scaling Benchmark

- Add a scaling loop to run the solver with different grid sizes:

```text
N = 32, 64, 128, 256
```

- The program outputs the benchmark data to:

```text
results/benchmark.csv
```

- The CSV file includes iterations, runtime, residual, and solution error for both CG and SOR.
- I also added a Python plotting script:

```text
scripts/plot_results.py
```

- The script reads `results/benchmark.csv` and makes two scaling plots:

```text
results/runtime_scaling.png
results/iteration_scaling.png
```

- Both plots use log-log scale to show the scaling behavior more clearly.
- From these two plots, CG is much more efficient than SOR for this problem.

### OpenMP Thread Scaling

- I added OpenMP parallelization to the CG solver.
- The main parallelized parts are:
  - `compute_Ax()`
  - `dot_product()`
  - vector update loops
  - residual and solution error calculations

- I kept the previous grid scaling benchmark, which tests:

```text
N = 32, 64, 128, 256
```

- This benchmark is still used to study how the solver changes with different grid sizes.

- I also added a new thread scaling benchmark for CG.

- I ran the solver with different numbers of OpenMP threads:

```text
threads = 1, 2, 4, 8
```
- The program outputs the thread scaling result to:

```text
results/thread_scaling.csv
```

- The CSV file includes the number of threads, CG runtime, CG residual, solution error, and speedup.

- I updated the plotting script:

```text
scripts/plot_results.py
```

- It now also makes two thread scaling plots:

```text
results/thread_speedup.png
results/thread_runtime.png
```

- `thread_speedup.png` shows the CG speedup compared with the 1-thread runtime.

- `thread_runtime.png` shows how the CG runtime changes when using more threads.

- The speedup is computed as:

```text
speedup = time with 1 thread / time with N threads
```

- The CG iteration count does not change much with different thread numbers.

- This is expected because OpenMP changes the runtime of each iteration, not the mathematical convergence of the CG method.

- I also added a switch to control whether SOR is included in the benchmark because SOR takes much longer to run, especially for larger grids.
- If I only want to test the OpenMP performance for CG, I set:

```text
run_sor = false;
```

- When I want to compare CG and SOR, I can set:

```text
run_sor = true;
```

- The SOR core update is kept serial because the standard Gauss-Seidel/SOR update has data dependency between grid points.
- Therefore, I only parallelized the CG-related parts in this version.

### Double Gaussian Source and Larger Thread Test

- I also tested a harder manufactured problem using two Gaussian functions:

```text
phi_exact = x(1-x)y(1-y) * (gaussian1 - 0.8 gaussian2)
```

- The Gaussian centers are `(0.65, 0.24)` and `(0.13, 0.90)`.
- I used `alpha = 100`, so the Gaussian functions are sharper and more localized.
- The factor `x(1-x)y(1-y)` makes `phi_exact = 0` on the boundary.

- I generated `rhs` directly from `phi_exact` using `compute_Ax()` to avoid hand-deriving the Gaussian Laplacian.

- This gives a manufactured discrete problem where:

```text
laplacian(phi_exact) = rhs
```

For the double Gaussian test, I obtained:

```text
N=32   CG iterations = 105
N=64   CG iterations = 220
N=128  CG iterations = 454
N=256  CG iterations = 940
```

- The solution error stayed around `1e-12`, showing that CG recovered the manufactured solution accurately.

- I also compared CG with SOR using the same double Gaussian source:

```text
N=32   CG: 105 iterations, 0.000589 s    SOR: 139 iterations, 0.001163 s
N=64   CG: 220 iterations, 0.004768 s    SOR: 801 iterations, 0.028318 s
N=128  CG: 454 iterations, 0.033666 s    SOR: 3521 iterations, 0.502923 s
N=256  CG: 940 iterations, 0.313736 s    SOR: 14857 iterations, 8.92609 s
```

- The performance gap is smaller than in the simpler sine-mode test because the double Gaussian source requires many more CG iterations.
- However, CG is still much faster than SOR for all tested grid sizes.

- While testing this source, I found two important implementation details.

- First, the solver uses zero Dirichlet boundary conditions:

```text
phi = 0 on the boundary
```

- The boundary values are fixed and are not updated by `compute_Ax()`, CG, or SOR.
- Therefore, the boundary points should not be treated as unknown equations.
- In the problem setup, I set the boundary values of `phi_exact` to zero and apply the discrete equation only on the interior points.

- Second, the discrete Laplacian operator is negative definite.
- Standard CG is designed for symmetric positive definite systems.
- To use CG correctly, I keep the project convention as:

```text
laplacian(phi) = rhs
```

- But inside the CG solver, I solve the equivalent positive definite system:

```text
-laplacian(phi) = -rhs
```

- This does not change the solution `phi`.
- It only changes the sign of both sides so that the CG method satisfies its positive-definite requirement.
- The residual is still checked using `residual = rhs - laplacian(phi)`.

- For OpenMP thread scaling, I used a larger fixed grid `N = 1024`.
- This gives each thread more work and makes the speedup clearer.

The thread scaling result was:

```text
threads=1  time=19.5086 s  speedup=1.00
threads=2  time=11.9223 s  speedup=1.64
threads=4  time=8.55634 s  speedup=2.28
threads=8  time=8.03579 s  speedup=2.43
```

- The CG iteration count stayed the same for all thread counts:

```text
CG iterations = 3940
```

- The speedup improves up to 4 threads and then saturates at 8 threads.
- This is likely due to memory bandwidth limits and synchronization overhead in the CG dot-product reductions.

### Unknown Multi-Mode Source Test

- Finally, I tested an unknown source problem.
- In this test, I directly prescribed a source term with a smooth background field and two localized Gaussian sources:

```text
rhs = sine/cosine background + large Gaussian - small Gaussian
```

- Unlike the manufactured tests, there is no known `phi_exact`.
- Therefore, I checked the solution using the residual instead of solution error:

```text
residual = rhs - laplacian(phi)
```

- For `N = 1024`, I obtained:

```text
CG iterations = 3811
CG time       = 8.20142 s
CG residual   = 1.23027e-07
```

- I also saved the source and solution fields:

```text
results/unknown_source.csv
results/unknown_solution.csv
results/unknown_source.png
results/unknown_solution.png
```

- This is closer to a real Poisson problem, where the source term is given and the potential field `phi` is unknown.

### Final Summary

- Overall, the CG solver accurately solves the manufactured 2D Poisson problems and produces small residuals for the unknown source test.
- CG is much faster than SOR for the tested grids, and OpenMP improves the runtime for larger problems, although the speedup saturates because the solver needs frequent memory access and dot-product reductions.
