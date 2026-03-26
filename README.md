# Numerical vs. Asymptotic Critical Radius

This folder contains a self-contained script for the comparison described in
Section 5 of the manuscript: the numerical critical radius of the full radial
mixed-ploidy model versus the small-`\upsilon` asymptotic approximation.

In the current manuscript source, this comparison appears as **Figure 3**.
Older workspace files were named `Figure2.py` and `Figure2Truth.py`; this
folder replaces that split with a single script that is easier to read, run,
and upload to GitHub.

## Contents

- `NumericsAsymptoticsPolyploid.py`: computes the numerical threshold from the
  full radial PDE, evaluates the asymptotic approximation, and plots the
  comparison.
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

with top-hat initial data

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
R_c \sim \frac{\sigma\sqrt{1-\phi^2}}{2\upsilon}.
$$

By default, the script reproduces the parameter sweep used for the manuscript
comparison:

- `sigma = 1.0`
- `phi in {0.1, 0.3, 0.5}`
- `upsilon` evenly spaced in `[0.001, 0.02]`

## Numerical approach

The full radial PDE is solved with a semi-implicit finite-difference scheme:

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

## Usage

Run the manuscript-style comparison:

```bash
python NumericsAsymptoticsPolyploid.py --output Figure3.pdf
```

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
  --upsilon-count 5 \
  --phis 0.1 0.3
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
  between the asymptotic approximation and the full model.
- If you want the same naming convention used in the manuscript, use
  `--output Figure3.pdf`.
