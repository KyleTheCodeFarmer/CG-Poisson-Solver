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
