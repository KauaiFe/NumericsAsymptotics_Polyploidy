# Validation of the corrected Figure 3

The regenerated figure contains 90 finite numerical thresholds: three values
of phi and 30 values of upsilon. Every numerical curve decreases with upsilon.
The largest absolute relative difference from the corrected leading-order
formula, measured relative to that formula, is 2.77%, 3.42% and 4.67% for
phi = 0.1, 0.3 and 0.5, respectively. This difference includes finite-upsilon
asymptotic error and numerical error; it is not an estimate of numerical accuracy.

## Regression and asymptotic checks

`python NumericsAsymptoticsPolyploid.py --self-test` passes checks of:

- The reaction calculation and the tridiagonal solver.
- Identical initial grid patches receiving identical outcomes. At phi = 0.1,
  upsilon = 0.02, dr = 0.2 and dt = 0.25, R0 = 21.605 and 21.795 both contract;
  R0 = 21.805 and 21.995 both expand.
- Insufficient integration time producing an unresolved result, not an
  arbitrary expansion/collapse decision.
- The corrected leading prefactor against the cubic wave-speed formula,
  evaluated using independently computed polynomial roots at upsilon = 1e-6.

## Resolution and domain checks

All radii below use sigma = 1 and phi = 0.1. Thresholds are reported to more
digits for reproducibility, not as a claim of comparable physical precision.
Bisection tolerance is 0.1 radius units. Different starting brackets can change
the reported midpoint by less than that tolerance.

| upsilon | Check | Estimated radius |
|---:|---|---:|
| 0.02 | dr = 0.20, dt = 0.25 | 21.79118 |
| 0.02 | dr = 0.10, dt = 0.10, using the coarse result as a bracket hint | 21.82523 |
| 0.001 | Default dr = 0.10, dt = 0.10 | 446.47552 |
| 0.001 | Finer dr = 0.05, dt = 0.05 | 446.60632 |
| 0.001 | Relaxation increased from 60 to 180 time units | 446.51912 |
| 0.001 | Outer padding increased from 80 to 160 radius units | 446.51912 |

The last three checks use the default result as a bracket hint. They can be
repeated using the script's public classes:

```python
from dataclasses import replace
from NumericsAsymptoticsPolyploid import DEFAULT_SOLVER_CONFIG, CriticalRadiusEstimator

checks = {
    "finer_grid": replace(DEFAULT_SOLVER_CONFIG, dr=0.05, dt=0.05),
    "longer_relaxation": replace(DEFAULT_SOLVER_CONFIG, min_decision_time=180.0),
    "larger_domain": replace(DEFAULT_SOLVER_CONFIG, domain_padding=160.0),
}
for name, config in checks.items():
    radius, metadata = CriticalRadiusEstimator(1.0, config).find_critical_radius(
        0.001, 0.1, radius_hint=446.4755183745723
    )
    print(name, radius, metadata)
```

These are targeted checks, not an exhaustive convergence study over all
parameters or a rigorous error bound for the continuum PDE.
