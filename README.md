# Numerical vs. Asymptotic Critical Radius for Fig. 3

This folder contains a self-contained script for the comparison described in
Section 5 of the manuscript: the numerical critical radius of the radial
continuous-time model versus the small-`\upsilon` asymptotic approximation.

In the revised manuscript source, this comparison appears as **Fig. 3**.
Older workspace files were named `Figure2.py` and `Figure2Truth.py`; this
folder replaces that split with a single script that is easier to read, run,
and upload to GitHub.

## Contents

- `NumericsAsymptoticsPolyploid.py`: computes the numerical threshold from the
  radial PDE, evaluates the asymptotic approximation, and plots the
  comparison.
- `Figure3.pdf` and `Figure3.png`: current rendered versions of Fig. 3.
- `Figure3_data.csv`: numerical and asymptotic values used to render Fig. 3.
- `README.md`: usage notes and a short description of the model.

## Model summary

The numerical experiment starts from the radial reaction-diffusion equation

$$
\frac{\partial y}{\partial t}
=
\frac{\sigma^2}{2}
\left(
\frac{\partial^2 y}{\partial r^2}
+
\frac{1}{r}\frac{\partial y}{\partial r}
\right)
+
f(y),
$$

where the local continuous-time reaction term is

$$
f(y)
=
\upsilon
+(\phi-2\upsilon-1)y
+3(1-\phi)y^2
+(2\phi+\upsilon-2)y^3.
$$

The initial condition is a top-hat patch,

$$
y(r,0)=
\begin{cases}
1, & r < R_0, \\
0, & r \ge R_0.
\end{cases}
$$

Here, `R0` is the radius of the initial patch of unreduced gametes. The
numerical critical radius `Rc` is the smallest initial radius for which the
patch expands rather than collapses.

The asymptotic comparison uses the small-`\upsilon` expression derived in the
manuscript,

$$
R_c
\sim
\frac{\sigma(1-\phi)}{\upsilon\sqrt{2(1-\phi)}}
=
\frac{\sigma}{\upsilon}\sqrt{\frac{1-\phi}{2}}.
$$

By default, the script reproduces the parameter sweep used for the manuscript
comparison:

- `sigma = 1.0`
- `phi in {0.1, 0.3, 0.5}`
- `upsilon` evenly spaced in `[0.001, 0.02]`

## Numerical Approach

The radial PDE is solved with a semi-implicit finite-difference scheme:

- diffusion is treated implicitly;
- the nonlinear reaction term is treated explicitly;
- radial symmetry is enforced at the origin through `y_r(0) = 0`;
- the outer edge of the computational domain uses a zero-flux truncation.

For each pair `(upsilon, phi)`, the code searches for the critical radius by:

1. bracketing the transition between collapse and expansion;
2. refining that bracket by bisection.

This is the numerical counterpart of the threshold argument in the manuscript:
patches below `Rc` contract, whereas patches above `Rc` generate outward spread.

## Requirements

- Python 3.11 or newer
- `numpy`
- `matplotlib`
- `scipy` (optional, but recommended)

If `scipy` is unavailable, the script falls back to a pure-NumPy tridiagonal
solver.

On systems where `python` is not mapped to Python 3, use `python3` in the
commands below.

## Usage

Run the manuscript-style comparison:

```bash
python NumericsAsymptoticsPolyploid.py --output Figure3.pdf
```

This computes the numerical critical radius for the radial PDE and plots
it against the small-`\upsilon` asymptotic approximation
`sigma * sqrt((1 - phi) / 2) / upsilon`.

Save the figure and the underlying table:

```bash
python NumericsAsymptoticsPolyploid.py \
  --output Figure3.pdf \
  --csv Figure3_data.csv
```

Run with timing diagnostics:

```bash
python NumericsAsymptoticsPolyploid.py \
  --output Figure3.pdf \
  --csv Figure3_data.csv \
  --profile
```

Run a smaller sweep as a quick test:

```bash
python NumericsAsymptoticsPolyploid.py \
  --output test.pdf \
  --upsilon-min 0.015 \
  --upsilon-max 0.02 \
  --upsilon-count 3 \
  --phis 0.1
```

Run lightweight validation checks without producing a figure:

```bash
python NumericsAsymptoticsPolyploid.py --self-test
```

Run a short convergence check comparing `dr = 0.20, dt = 0.25` with
`dr = 0.10, dt = 0.10` for one representative parameter point:

```bash
python NumericsAsymptoticsPolyploid.py --convergence-check
```

## Output

The script writes:

- a PDF figure with filled circles for the numerical critical radius and open
  circles for the asymptotic approximation;
- optionally, a CSV file with the columns
  `phi`, `upsilon`, `numerical_radius`, and `asymptotic_radius`.

## Notes

- The script is intentionally self-contained and does not import the older
  workspace figure scripts.
- It is the cleaned version of the code used for the manuscript comparison
  between the asymptotic approximation and the radial continuous-time model.
- If you want the same naming convention used in the manuscript, use
  `--output Figure3.pdf`.
- The checked-in `Figure3_data.csv`, `Figure3.pdf`, and `Figure3.png` have been
  refreshed with the current asymptotic expression above.
- Numerical thresholds near the expansion/collapse boundary can be sensitive to
  the final integration time and the grid. For manuscript-quality values,
  check convergence by reducing `dr` and `dt`, and increase `--max-time-chunks`
  for cases that remain close to the threshold.
