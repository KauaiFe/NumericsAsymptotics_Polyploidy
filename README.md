# Numerical vs. Asymptotic Critical-Radius Comparison

This folder contains a cleaned, self-contained implementation of the numerical
experiment that compares the **full radial mixed-ploidy model** with the
**small-upsilon asymptotic approximation** for the critical radius required to
nucleate an expanding tetraploid patch.

It is aligned with the narrative in **Section 5** of the manuscript and with
the corresponding figure caption in the current manuscript source. In the
current `main.tex`, this comparison appears as **Figure 3**, even though older
workspace scripts were named `Figure2.py` and `Figure2Truth.py`. This folder
consolidates those exploratory scripts into one documented script that is ready
to upload to GitHub.

## Files

- `NumericsAsymptoticsPolyploid.py`
  - Main script. Computes the numerical critical radius from the full radial
    PDE and compares it against the manuscript asymptotic law.
- `README.md`
  - This documentation.

## What The Script Computes

The script compares two quantities:

1. **Numerical critical radius**
   - Obtained from the full radial reaction-diffusion model
     \[
     y_t = \frac{\sigma^2}{2}\left(y_{rr} + \frac{1}{r}y_r\right) + f(y),
     \]
     using a top-hat initial patch
     \[
     y(r,0) = 1 \text{ for } r < R_0,\qquad y(r,0) = 0 \text{ otherwise.}
     \]
   - The numerical critical radius \(R_c\) is the smallest \(R_0\) for which
     the localized patch expands instead of collapsing.

2. **Asymptotic critical radius**
   - Uses the small-upsilon approximation derived in the manuscript:
     \[
     R_c \sim \frac{\sigma\sqrt{1-\phi^2}}{2\upsilon}.
     \]

By default, the script reproduces the manuscript-style parameter sweep:

- `sigma = 1.0`
- `phi in {0.1, 0.3, 0.5}`
- `upsilon` evenly spaced in `[0.001, 0.02]`

## Numerical Method

The numerical critical radius is computed as follows:

1. The radial PDE is discretized with a **semi-implicit finite-difference**
   scheme.
   - Diffusion is treated implicitly.
   - The nonlinear reaction term is treated explicitly.
2. The origin uses the natural radial symmetry condition `y_r(0) = 0`.
3. The finite computational domain is truncated far from the front with a
   zero-flux outer boundary.
4. For each `(upsilon, phi)` pair, the code:
   - brackets the transition between collapse and expansion,
   - then refines the threshold by bisection.

This is the numerical counterpart of the critical-radius discussion in the
manuscript: patches smaller than \(R_c\) collapse, whereas patches larger than
\(R_c\) generate outward spread.

## Requirements

- Python 3.11 or newer
- `numpy`
- `matplotlib`
- `scipy` (optional but recommended; if unavailable, the script falls back to a
  pure-NumPy tridiagonal solver)

## Usage

Run with the manuscript-style defaults:

```bash
python NumericsAsymptoticsPolyploid.py --output Figure3.pdf
```

Save the comparison table as CSV as well:

```bash
python NumericsAsymptoticsPolyploid.py \
  --output Figure3.pdf \
  --csv Figure3_data.csv
```

Enable profiling information:

```bash
python NumericsAsymptoticsPolyploid.py \
  --output Figure3.pdf \
  --csv Figure3_data.csv \
  --profile
```

Use a smaller sweep for a quick smoke test:

```bash
python NumericsAsymptoticsPolyploid.py \
  --output test.pdf \
  --upsilon-count 5 \
  --phis 0.1 0.3
```

## Output

The script produces:

- a PDF figure with
  - **filled circles** for the numerical critical radius,
  - **open circles** for the asymptotic critical radius,
  - one color per `phi`;
- optionally, a CSV file with columns:
  - `phi`
  - `upsilon`
  - `numerical_radius`
  - `asymptotic_radius`

## Notes For Repository Use

- The script is intentionally self-contained and does **not** import the older
  workspace files.
- It supersedes the previous split between `Figure2.py` and `Figure2Truth.py`
  for the manuscript comparison between the asymptotic approximation and the
  full model.
- If you want the manuscript naming convention, use `--output Figure3.pdf`.
