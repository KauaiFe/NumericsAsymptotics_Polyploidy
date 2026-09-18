from __future__ import annotations

"""
Numerical comparison between the full radial mixed-ploidy model and the
small-upsilon asymptotic critical-radius approximation.

This script is aligned with the critical-radius comparison described in
Section 5 of the manuscript and with the corresponding figure caption in
the current LaTeX source (`main.tex`, Figure 3 in the manuscript build).

In manuscript notation, the numerical model solves the radial PDE

    y_t = (sigma^2 / 2) * (y_rr + (1 / r) y_r) + f(y),

with a top-hat initial condition representing a localized patch of
unreduced gametes,

    y(r, 0) = 1 for r < R0, and 0 otherwise.

The numerical critical radius R_c is the smallest R0 for which the patch
expands instead of collapsing. The asymptotic comparison curve is the
small-upsilon approximation

    R_c ~ (2 / 3) * sigma * sqrt((1 - phi) / 2) / upsilon.

The default parameters reproduce the sweep discussed in the manuscript:
phi in {0.1, 0.3, 0.5}, sigma = 1, and upsilon in [0.001, 0.02].
"""

import argparse
import csv
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    from scipy.linalg.lapack import get_lapack_funcs
except ImportError:  # pragma: no cover - pure-NumPy fallback remains available
    get_lapack_funcs = None


DEFAULT_SIGMA = 1.0
DEFAULT_PHIS = (0.1, 0.3, 0.5)
DEFAULT_UPSILONS = np.linspace(0.001, 0.02, 30)


@dataclass(frozen=True)
class SolverConfig:
    """
    Numerical settings for the radial PDE solve and the critical-radius search.

    The defaults were tuned to make the manuscript sweep practical while still
    resolving the expansion-versus-collapse threshold robustly.
    """

    dr: float = 0.10
    dt: float = 0.10
    radius_tolerance: float = 0.1
    max_bisect_iter: int = 30
    initial_radius: float = 2.0
    min_radius: float = 0.05
    max_radius: float = 3000.0
    growth_factor: float = 1.8
    max_halving_steps: int = 16
    min_domain_radius: float = 250.0
    domain_scale: float = 1.0
    domain_padding: float = 80.0
    chunk_time_floor: float = 300.0
    chunk_time_scale: float = 6.0
    asym_time_factor: float = 0.0
    max_time_chunks: int = 8
    min_decision_time: float = 60.0
    velocity_window: float = 40.0
    velocity_tolerance: float = 1e-9
    front_level: float = 0.5
    check_interval: float = 2.0
    profile: bool = False

    @property
    def check_stride(self) -> int:
        return max(1, int(round(self.check_interval / self.dt)))


DEFAULT_SOLVER_CONFIG = SolverConfig()


@dataclass(frozen=True)
class ReactionCoefficients:
    """Precomputed coefficients for the corrected local reaction term f(y)."""

    upsilon: float
    phi: float
    c3: float
    c2: float
    c1: float
    c0: float

    @classmethod
    def from_parameters(cls, upsilon: float, phi: float) -> "ReactionCoefficients":
        return cls(
            upsilon=upsilon,
            phi=phi,
            c3=2.0 * phi + upsilon - 2.0,
            c2=3.0 * (1.0 - phi),
            c1=phi - 2.0 * upsilon - 1.0,
            c0=upsilon,
        )


@dataclass(frozen=True)
class ComparisonCurve:
    """Numerical and asymptotic critical-radius values for one phi."""

    phi: float
    upsilons: np.ndarray
    numerical_radii: np.ndarray
    asymptotic_radii: np.ndarray


def full_reaction(
    y: np.ndarray | float,
    upsilon: float,
    phi: float,
) -> np.ndarray | float:
    """
    Corrected local reaction term f(y) used in the numerical comparison.
    """

    return (
        upsilon
        + (phi - 2.0 * upsilon - 1.0) * y
        + 3.0 * (1.0 - phi) * y * y
        + (2.0 * phi + upsilon - 2.0) * y * y * y
    )


def asymptotic_critical_radius(
    upsilon: float,
    phi: float,
    sigma: float,
) -> float:
    """
    Corrected small-upsilon asymptotic critical radius (see README).
    """

    if upsilon <= 0.0 or not (0.0 < phi < 1.0):
        return math.nan

    prefactor = (1.0 - phi) / 2.0
    if prefactor <= 0.0:
        return math.nan

    return (2.0 / 3.0) * sigma * math.sqrt(prefactor) / upsilon


def bistable_equilibria(upsilon: float, phi: float) -> tuple[float, float]:
    """Return the lower stable and unstable roots, with a stable quadratic formula."""

    a = 1.0 - phi
    if not (0.0 <= phi < 1.0 and 0.0 < upsilon < a / 5.0):
        raise ValueError("The radial threshold calculation requires 0 < upsilon < (1-phi)/5.")
    discriminant = math.sqrt((a - upsilon) * (a - 5.0 * upsilon))
    root_sum = a + upsilon + discriminant
    return 2.0 * upsilon / root_sum, root_sum / (2.0 * (2.0 * a - upsilon))


class TridiagonalSolver:
    """
    Reusable tridiagonal linear solver.

    If SciPy is available we factor once with LAPACK; otherwise a Thomas-method
    fallback is prepared once and reused.
    """

    def __init__(
        self,
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        *,
        use_scipy: bool = True,
    ):
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.diagonal = np.array(diagonal, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)

        self._gttrs = None
        if use_scipy and get_lapack_funcs is not None:
            gttrf = get_lapack_funcs("gttrf", dtype=np.float64)
            self._gttrs = get_lapack_funcs("gttrs", dtype=np.float64)
            dl, d, du, du2, ipiv, info = gttrf(
                self.lower.copy(),
                self.diagonal.copy(),
                self.upper.copy(),
                overwrite_dl=True,
                overwrite_d=True,
                overwrite_du=True,
            )
            if info != 0:
                raise RuntimeError(f"dgttrf failed with info={info}")
            self._dl = dl
            self._d = d
            self._du = du
            self._du2 = du2
            self._ipiv = ipiv
        else:
            self._prepare_thomas()

    def _prepare_thomas(self) -> None:
        size = self.diagonal.size
        self._inv_pivots = np.empty(size, dtype=np.float64)
        self._modified_upper = np.empty(size - 1, dtype=np.float64)

        pivot = self.diagonal[0]
        self._inv_pivots[0] = 1.0 / pivot
        if size > 1:
            self._modified_upper[0] = self.upper[0] / pivot

        for index in range(1, size - 1):
            pivot = (
                self.diagonal[index]
                - self.lower[index - 1] * self._modified_upper[index - 1]
            )
            self._inv_pivots[index] = 1.0 / pivot
            self._modified_upper[index] = self.upper[index] * self._inv_pivots[index]

        if size > 1:
            pivot = self.diagonal[-1] - self.lower[-1] * self._modified_upper[-1]
            self._inv_pivots[-1] = 1.0 / pivot

    def solve(self, rhs: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Solve A x = rhs into `out` without allocating new arrays."""

        if self._gttrs is not None:
            np.copyto(out, rhs)
            solution, info = self._gttrs(
                self._dl,
                self._d,
                self._du,
                self._du2,
                self._ipiv,
                out,
                overwrite_b=True,
            )
            if info != 0:
                raise RuntimeError(f"dgttrs failed with info={info}")
            if solution is not out:
                np.copyto(out, solution)
            return out

        out[0] = rhs[0] * self._inv_pivots[0]
        for index in range(1, rhs.size):
            out[index] = (
                rhs[index] - self.lower[index - 1] * out[index - 1]
            ) * self._inv_pivots[index]
        for index in range(rhs.size - 2, -1, -1):
            out[index] -= self._modified_upper[index] * out[index + 1]
        return out


class RadialSemiImplicitSolver:
    """
    Semi-implicit finite-difference solver for the radial full PDE.

    Diffusion is treated implicitly and the nonlinear reaction explicitly.
    A reflective condition is enforced at r = 0, and the outer boundary is
    truncated far away with a zero-flux condition.
    """

    def __init__(self, radius_max: float, sigma: float, config: SolverConfig):
        self.config = config
        self.sigma = sigma
        self.radius_max = radius_max
        self.dr = config.dr
        self.r = self.dr * np.arange(
            int(round(radius_max / self.dr)) + 1,
            dtype=np.float64,
        )
        self.linear_solver = TridiagonalSolver(*self._build_system())

        size = self.r.size
        self.state = np.zeros(size, dtype=np.float64)
        self.next_state = np.zeros(size, dtype=np.float64)
        self.rhs = np.zeros(size, dtype=np.float64)
        self.reaction_buffer = np.zeros(size, dtype=np.float64)
        self.work1 = np.zeros(size, dtype=np.float64)
        self.work2 = np.zeros(size, dtype=np.float64)

    def _build_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the implicit diffusion operator in radial coordinates."""

        size = self.r.size
        theta = 0.5 * self.sigma * self.sigma * self.config.dt
        inv_dr2 = 1.0 / (self.dr * self.dr)

        lower = np.empty(size - 1, dtype=np.float64)
        diagonal = np.full(size, 1.0 + 2.0 * theta * inv_dr2, dtype=np.float64)
        upper = np.empty(size - 1, dtype=np.float64)

        # Symmetry at r = 0 gives y_r(0) = 0.
        diagonal[0] = 1.0 + 4.0 * theta * inv_dr2
        upper[0] = -4.0 * theta * inv_dr2

        if size > 2:
            interior_r = self.r[1:-1]
            lower[:-1] = -theta * (inv_dr2 - 0.5 / (interior_r * self.dr))
            upper[1:] = -theta * (inv_dr2 + 0.5 / (interior_r * self.dr))

        # Zero-flux truncation at the artificial outer boundary.
        lower[-1] = -2.0 * theta * inv_dr2
        return lower, diagonal, upper

    def _reaction_inplace(
        self,
        state: np.ndarray,
        coeffs: ReactionCoefficients,
        out: np.ndarray,
    ) -> None:
        np.multiply(state, state, out=self.work1)
        np.multiply(self.work1, state, out=self.work2)
        np.copyto(out, self.work2)
        out *= coeffs.c3
        out += coeffs.c2 * self.work1
        out += coeffs.c1 * state
        out += coeffs.c0

    def occupied_radius(self, state: np.ndarray) -> float:
        """
        Interpolate the radius where the profile crosses the front level.
        """

        mask = state >= self.config.front_level
        if not np.any(mask):
            return 0.0

        last_above = mask.size - 1 - int(np.argmax(mask[::-1]))
        if last_above >= state.size - 1:
            return self.r[-1]

        y_left = state[last_above]
        y_right = state[last_above + 1]
        if y_right == y_left:
            return self.r[last_above]

        fraction = (self.config.front_level - y_left) / (y_right - y_left)
        fraction = min(1.0, max(0.0, fraction))
        return self.r[last_above] + fraction * self.dr

    def simulate(
        self,
        radius0: float,
        coeffs: ReactionCoefficients,
        chunk_time: float,
        max_chunks: int,
    ) -> tuple[bool | None, dict[str, float | int | str]]:
        """Classify sustained front motion after relaxation; None means unresolved.

        Decisions use three successive late-time velocity windows, never the
        displacement from the requested (unrounded) top-hat radius. This makes
        identical grid states receive identical decisions on a fixed domain.
        A consistent velocity must also cease rapid transient decay. Borderline
        cases continue until the integration budget is exhausted, then remain
        unresolved rather than being assigned an arbitrary drift sign.
        """

        _, unstable_root = bistable_equilibria(coeffs.upsilon, coeffs.phi)
        state, next_state, rhs = self.state, self.next_state, self.rhs
        state.fill(0.0)
        hot_cells = np.searchsorted(self.r, radius0, side="left")
        state[:hot_cells] = 1.0
        initial_extent = float(self.occupied_radius(state))
        steps_per_chunk = int(np.ceil(chunk_time / self.config.dt))
        total_steps = steps_per_chunk * max_chunks
        check_stride = self.config.check_stride
        check_time = check_stride * self.config.dt
        window_checks = max(1, int(round(self.config.velocity_window / check_time)))
        history: deque[tuple[float, float]] = deque(maxlen=3 * window_checks + 1)
        history.append((0.0, initial_extent))
        run_start = time.perf_counter()
        max_extent = final_extent = initial_extent
        velocity = math.nan

        def result(outcome: bool | None, reason: str, step: int):
            diagnostics = {
                "reason": reason,
                "initial_extent": initial_extent,
                "max_extent": max_extent,
                "final_extent": final_extent,
                "radius_max": self.radius_max,
                "t_max": total_steps * self.config.dt,
                "decision_time": step * self.config.dt,
                "front_velocity": velocity,
                "steps": step,
            }
            if self.config.profile:
                diagnostics["elapsed_s"] = time.perf_counter() - run_start
            return outcome, diagnostics

        for step in range(1, total_steps + 1):
            self._reaction_inplace(state, coeffs, self.reaction_buffer)
            np.copyto(rhs, state)
            rhs += self.config.dt * self.reaction_buffer
            self.linear_solver.solve(rhs, next_state)
            np.clip(next_state, 0.0, 1.0, out=next_state)
            state, next_state = next_state, state
            if step % check_stride != 0 and step != total_steps:
                continue

            peak = float(np.max(state))
            final_extent = float(self.occupied_radius(state))
            max_extent = max(max_extent, final_extent)
            elapsed_time = step * self.config.dt

            # Once the entire population lies below the unstable equilibrium,
            # the maximum principle rules out invasion of the upper state.
            if peak < unstable_root - 1e-10:
                return result(False, "below_unstable_equilibrium", step)

            if final_extent >= self.r[-1] - 10.0 * self.sigma:
                return result(None, "front_near_outer_boundary", step)

            history.append((elapsed_time, final_extent))
            if len(history) < history.maxlen:
                continue
            if history[0][0] < self.config.min_decision_time:
                continue
            samples = [history[i * window_checks] for i in range(4)]
            velocities = np.diff([p[1] for p in samples]) / np.diff([p[0] for p in samples])
            velocity = float(velocities[-1])
            tolerance = self.config.velocity_tolerance * self.sigma
            settled = abs(velocities[-1]) >= 0.8 * abs(velocities[0])
            if settled and np.all(velocities > tolerance):
                return result(True, "sustained_positive_velocity", step)
            if settled and np.all(velocities < -tolerance):
                return result(False, "sustained_negative_velocity", step)

        return result(None, "unresolved_at_time_limit", total_steps)


class CriticalRadiusEstimator:
    """
    Find the numerical critical radius by bracketing and bisection.
    """

    def __init__(self, sigma: float, config: SolverConfig = DEFAULT_SOLVER_CONFIG):
        self.sigma = sigma
        self.config = config
        self._solver_cache: dict[tuple[float, float, int], RadialSemiImplicitSolver] = {}
        self.total_pde_solves = 0

    def _required_domain_radius(self, radius0: float) -> float:
        return max(
            self.config.min_domain_radius,
            self.config.domain_scale * radius0 + self.config.domain_padding,
        )

    def _effective_config(self, radius0: float) -> SolverConfig:
        # Keep the requested resolution for every radius in the sweep.
        return self.config

    def _solver_for_radius(
        self,
        radius0: float,
    ) -> tuple[RadialSemiImplicitSolver, SolverConfig]:
        effective = self._effective_config(radius0)
        intervals = int(np.ceil(self._required_domain_radius(radius0) / effective.dr))
        key = (round(effective.dr, 12), round(effective.dt, 12), intervals)
        solver = self._solver_cache.get(key)
        if solver is None:
            solver = RadialSemiImplicitSolver(
                intervals * effective.dr,
                self.sigma,
                effective,
            )
            self._solver_cache[key] = solver
        return solver, effective

    def _starting_radius(self, asymptotic_radius: float, radius_hint: float) -> float:
        if np.isfinite(radius_hint) and radius_hint > 0.0:
            return max(self.config.initial_radius, radius_hint)
        if np.isfinite(asymptotic_radius) and asymptotic_radius > 0.0:
            return max(self.config.initial_radius, 0.5 * asymptotic_radius)
        return self.config.initial_radius

    def _chunk_time(self, radius0: float, asymptotic_radius: float) -> float:
        horizon = max(
            self.config.chunk_time_floor,
            self.config.chunk_time_scale * radius0,
        )
        if (
            self.config.asym_time_factor > 0.0
            and np.isfinite(asymptotic_radius)
            and asymptotic_radius > 0.0
        ):
            horizon = max(horizon, self.config.asym_time_factor * asymptotic_radius)
        return horizon

    def _max_radius(self, asymptotic_radius: float) -> float:
        if np.isfinite(asymptotic_radius) and asymptotic_radius > 0.0:
            return max(self.config.max_radius, 4.0 * asymptotic_radius)
        return self.config.max_radius

    def find_bracket(
        self,
        upsilon: float,
        phi: float,
        asymptotic_radius_value: float,
        radius_hint: float,
    ) -> tuple[float, float, object]:
        coeffs = ReactionCoefficients.from_parameters(upsilon, phi)
        trial_cache: dict[float, tuple[bool | None, dict[str, float | int | str]]] = {}
        max_radius = self._max_radius(asymptotic_radius_value)

        def classify(radius0: float) -> tuple[bool, dict[str, float | int | str]]:
            radius0 = float(radius0)
            if radius0 not in trial_cache:
                solver, effective = self._solver_for_radius(radius0)
                trial_cache[radius0] = solver.simulate(
                    radius0,
                    coeffs,
                    chunk_time=self._chunk_time(radius0, asymptotic_radius_value),
                    max_chunks=effective.max_time_chunks,
                )
                self.total_pde_solves += 1
            outcome, diagnostics = trial_cache[radius0]
            if outcome is None:
                raise RuntimeError(
                    f"Unresolved radius {radius0:.6g}: {diagnostics['reason']}. "
                    "Increase max_time_chunks or enlarge the domain; no threshold "
                    "will be reported from an unresolved classification."
                )
            return outcome, diagnostics

        low = max(
            self.config.min_radius,
            self._starting_radius(asymptotic_radius_value, radius_hint),
        )
        expands, _ = classify(low)

        if expands:
            test = low
            for _ in range(self.config.max_halving_steps):
                test *= 0.5
                if test < self.config.min_radius:
                    break
                expands_test, _ = classify(test)
                if not expands_test:
                    return test, low, classify
                low = test
            raise RuntimeError("Could not find a subcritical lower bracket.")

        high = low * self.config.growth_factor
        while high <= max_radius:
            expands, _ = classify(high)
            if expands:
                return low, high, classify
            low = high
            high *= self.config.growth_factor

        raise RuntimeError(
            "Could not find a supercritical upper bracket up to "
            f"R={max_radius:.1f} for upsilon={upsilon:.5f}, phi={phi:.3f}."
        )

    def find_critical_radius(
        self,
        upsilon: float,
        phi: float,
        radius_hint: float = math.nan,
    ) -> tuple[float, dict[str, float | int]]:
        asymptotic_radius_value = asymptotic_critical_radius(upsilon, phi, self.sigma)
        solve_start = self.total_pde_solves
        wall_start = time.perf_counter()

        low, high, classify = self.find_bracket(
            upsilon,
            phi,
            asymptotic_radius_value=asymptotic_radius_value,
            radius_hint=radius_hint,
        )

        for _ in range(self.config.max_bisect_iter):
            mid = 0.5 * (low + high)
            expands, _ = classify(mid)
            if expands:
                high = mid
            else:
                low = mid

            _, effective = self._solver_for_radius(mid)
            radius_tolerance = max(
                self.config.radius_tolerance,
                0.5 * effective.dr,
            )
            if high - low < radius_tolerance:
                break

        critical_radius = 0.5 * (low + high)
        metadata = {
            "pde_solves": self.total_pde_solves - solve_start,
            "elapsed_s": time.perf_counter() - wall_start,
            "lower_radius": low,
            "upper_radius": high,
        }
        return critical_radius, metadata


def format_value(value: float, digits: int = 4) -> str:
    """Consistent scalar formatting for console output."""

    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def compute_comparison_curves(
    sigma: float,
    phis: tuple[float, ...],
    upsilons: np.ndarray,
    config: SolverConfig,
) -> list[ComparisonCurve]:
    """
    Compute numerical and asymptotic critical radii for all requested curves.
    """

    estimator = CriticalRadiusEstimator(sigma, config)
    curves: list[ComparisonCurve] = []
    total_start = time.perf_counter()

    for phi in phis:
        print(f"\nphi = {phi}")
        numerical_radii: list[float] = []
        asymptotic_radii: list[float] = []
        previous_numeric = math.nan
        previous_asymptotic = math.nan

        for upsilon in upsilons:
            print(f"  upsilon = {upsilon:.5f}")
            asymptotic_radius_value = asymptotic_critical_radius(upsilon, phi, sigma)

            radius_hint = math.nan
            if (
                np.isfinite(previous_numeric)
                and previous_numeric > 0.0
                and np.isfinite(previous_asymptotic)
                and previous_asymptotic > 0.0
                and np.isfinite(asymptotic_radius_value)
                and asymptotic_radius_value > 0.0
            ):
                # Transport the previous numerical threshold by the local
                # asymptotic scaling so the next bracket starts near the new Rc.
                radius_hint = previous_numeric * (
                    asymptotic_radius_value / previous_asymptotic
                )

            try:
                numerical_radius, metadata = estimator.find_critical_radius(
                    upsilon,
                    phi,
                    radius_hint=radius_hint,
                )
                print(f"    Rc numerical   = {format_value(numerical_radius)}")
                if config.profile:
                    print(
                        "    PDE solves     = "
                        f"{metadata['pde_solves']} ({metadata['elapsed_s']:.2f} s)"
                    )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Sweep stopped at phi={phi}, upsilon={upsilon}: {exc}"
                ) from exc

            print(f"    Rc asymptotic  = {format_value(asymptotic_radius_value)}")

            previous_numeric = numerical_radius
            previous_asymptotic = asymptotic_radius_value
            numerical_radii.append(numerical_radius)
            asymptotic_radii.append(asymptotic_radius_value)

        curves.append(
            ComparisonCurve(
                phi=phi,
                upsilons=np.array(upsilons, dtype=float),
                numerical_radii=np.array(numerical_radii, dtype=float),
                asymptotic_radii=np.array(asymptotic_radii, dtype=float),
            )
        )

    if config.profile:
        elapsed = time.perf_counter() - total_start
        print(
            f"\nTotal runtime: {elapsed:.2f} s over {estimator.total_pde_solves} PDE solves"
        )

    return curves


def write_curves_csv(curves: list[ComparisonCurve], csv_path: Path) -> None:
    """Write the sweep results to a tidy CSV file."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["phi", "upsilon", "numerical_radius", "asymptotic_radius"]
        )
        for curve in curves:
            for upsilon, numerical_radius, asymptotic_radius_value in zip(
                curve.upsilons,
                curve.numerical_radii,
                curve.asymptotic_radii,
            ):
                writer.writerow(
                    [
                        f"{curve.phi:.6f}",
                        f"{upsilon:.8f}",
                        f"{numerical_radius:.12g}",
                        f"{asymptotic_radius_value:.12g}",
                    ]
                )


def plot_curves(curves: list[ComparisonCurve], output_path: Path) -> None:
    """
    Plot the manuscript-style comparison:
    filled circles for numerical Rc, open circles for asymptotic Rc.
    """

    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(tempfile.gettempdir()) / "polyploid-matplotlib-cache"
        try:
            mpl_cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        else:
            os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = Path(tempfile.gettempdir()) / "polyploid-cache"
        try:
            xdg_cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        else:
            os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    color_cycle = {
        0.1: "tab:blue",
        0.3: "tab:orange",
        0.5: "tab:green",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for curve in curves:
        color = color_cycle.get(curve.phi, None)
        ax.plot(curve.upsilons, curve.numerical_radii, "o", color=color, ms=4)
        ax.plot(
            curve.upsilons,
            curve.asymptotic_radii,
            "o",
            mec=color,
            mfc="none",
            ms=7,
            mew=1.5,
        )

    ax.set_xlabel(r"Rate of unreduced gametes $\upsilon$")
    ax.set_ylabel(r"Critical radius $R_c$")
    ax.set_xlim(0.0, 0.0204)
    ax.set_xticks(np.linspace(0.0, 0.02, 6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="major", length=6, labelsize=10)
    ax.margins(y=0.05)

    phi_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_cycle.get(curve.phi, "k"),
            linestyle="None",
            markerfacecolor=color_cycle.get(curve.phi, "k"),
            markersize=6,
            label=fr"$\phi={curve.phi}$",
        )
        for curve in curves
    ]
    marker_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linestyle="None",
            markersize=6,
            label=r"Numerical $R_c$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linestyle="None",
            markerfacecolor="none",
            markersize=7,
            label=r"Corrected asymptotic $R_c$",
        ),
    ]
    phi_handle = ax.legend(handles=phi_legend, loc="upper right")
    ax.add_artist(phi_handle)
    ax.legend(handles=marker_legend, loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight")
    for extension in (".png", ".svg"):
        if output_path.suffix.lower() != extension:
            fig.savefig(output_path.with_suffix(extension), dpi=200, bbox_inches="tight")

    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def validate_reaction_implementation() -> None:
    """
    Check that the in-place vectorized reaction matches the scalar formula.
    """

    config = replace(DEFAULT_SOLVER_CONFIG, dr=0.1, dt=0.05)
    solver = RadialSemiImplicitSolver(radius_max=1.0, sigma=1.0, config=config)
    state = np.linspace(0.0, 1.0, solver.r.size, dtype=np.float64)
    out = np.empty_like(state)

    for upsilon, phi in ((0.005, 0.1), (0.02, 0.3), (0.05, 0.5)):
        coeffs = ReactionCoefficients.from_parameters(upsilon, phi)
        solver._reaction_inplace(state, coeffs, out)
        expected = np.array(
            [full_reaction(float(value), upsilon, phi) for value in state],
            dtype=np.float64,
        )
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def validate_numpy_tridiagonal_solver() -> None:
    """
    Check the pure-NumPy tridiagonal fallback against a dense solve.
    """

    lower = np.array([-0.7, -0.5, -0.2], dtype=np.float64)
    diagonal = np.array([2.0, 2.4, 2.1, 1.8], dtype=np.float64)
    upper = np.array([-0.3, -0.4, -0.6], dtype=np.float64)
    rhs = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64)

    dense = np.diag(diagonal) + np.diag(upper, 1) + np.diag(lower, -1)
    expected = np.linalg.solve(dense, rhs)
    out = np.empty_like(rhs)

    solver = TridiagonalSolver(lower, diagonal, upper, use_scipy=False)
    solver.solve(rhs, out)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def run_self_tests() -> None:
    """Run lightweight deterministic validation checks."""

    validate_reaction_implementation()
    validate_numpy_tridiagonal_solver()
    validate_threshold_classification()
    validate_asymptotic_prefactor()
    print("Self-tests passed: reaction, linear solver, threshold decisions, and asymptotic limit.")


def validate_threshold_classification() -> None:
    """Regression for identical patches with different requested radii."""

    config = replace(DEFAULT_SOLVER_CONFIG, dr=0.2, dt=0.25)
    solver = RadialSemiImplicitSolver(250.0, 1.0, config)
    coefficients = ReactionCoefficients.from_parameters(0.02, 0.1)
    for radius, expected in ((21.605, False), (21.795, False), (21.805, True), (21.995, True)):
        outcome, _ = solver.simulate(radius, coefficients, 300.0, 1)
        assert outcome is expected, (radius, outcome, expected)
    outcome, diagnostics = solver.simulate(21.605, coefficients, 20.0, 1)
    assert outcome is None and diagnostics["reason"] == "unresolved_at_time_limit"


def validate_asymptotic_prefactor() -> None:
    """Compare the leading law with the independently evaluated cubic roots."""

    upsilon = 1e-6
    for phi in DEFAULT_PHIS:
        coefficients = ReactionCoefficients.from_parameters(upsilon, phi)
        stable, unstable, upper = np.sort(np.roots([
            coefficients.c3, coefficients.c2, coefficients.c1, coefficients.c0,
        ]))
        speed = math.sqrt(-coefficients.c3) * (upper + stable - 2 * unstable) / 2
        np.testing.assert_allclose(
            asymptotic_critical_radius(upsilon, phi, 1.0), 0.5 / speed, rtol=1e-5,
        )


def run_convergence_check(
    sigma: float,
    phi: float,
    upsilon: float,
    config: SolverConfig,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    """
    Compare one critical radius on coarse and finer space-time grids.

    This is a short validation check, not a full convergence study.
    """

    print(
        "Convergence check for "
        f"phi={phi:.3f}, upsilon={upsilon:.5f}, sigma={sigma:.3f}"
    )

    coarse_config = replace(config, dr=0.20, dt=0.25, profile=False)
    fine_config = replace(config, dr=0.10, dt=0.10, profile=False)

    coarse_estimator = CriticalRadiusEstimator(sigma, coarse_config)
    coarse_radius, coarse_metadata = coarse_estimator.find_critical_radius(
        upsilon,
        phi,
    )

    fine_estimator = CriticalRadiusEstimator(sigma, fine_config)
    fine_radius, fine_metadata = fine_estimator.find_critical_radius(
        upsilon,
        phi,
        radius_hint=coarse_radius,
    )

    difference = abs(fine_radius - coarse_radius)
    scale = max(abs(fine_radius), 1.0)
    relative_difference = difference / scale
    allowed_difference = max(absolute_tolerance, relative_tolerance * scale)

    print(
        "  dr=0.20, dt=0.25: "
        f"Rc={coarse_radius:.6f} "
        f"({coarse_metadata['pde_solves']} PDE solves)"
    )
    print(
        "  dr=0.10, dt=0.10: "
        f"Rc={fine_radius:.6f} "
        f"({fine_metadata['pde_solves']} PDE solves)"
    )
    print(
        "  difference: "
        f"{difference:.6f} ({100.0 * relative_difference:.2f}% of fine-grid Rc)"
    )

    if not (np.isfinite(coarse_radius) and np.isfinite(fine_radius)):
        raise AssertionError("Convergence check failed: non-finite critical radius.")

    if difference > allowed_difference:
        raise AssertionError(
            "Convergence check failed: coarse/fine critical radii differ by "
            f"{difference:.6f}, above the allowed {allowed_difference:.6f}. "
            "Near-threshold results should be rerun with smaller dr/dt and "
            "longer integration time."
        )

    print(
        "Convergence check passed within "
        f"max({absolute_tolerance:.3f}, {relative_tolerance:.1%} of fine-grid Rc)."
    )


def require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise SystemExit(f"{name} must be positive.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the numerical critical radius of the full radial PDE with "
            "the small-upsilon asymptotic approximation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("Figure3.pdf"),
        help="Output path for the comparison figure.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV file for the numerical and asymptotic curves.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="Dispersal scale sigma used in both the PDE and the asymptotic law.",
    )
    parser.add_argument(
        "--phis",
        type=float,
        nargs="+",
        default=list(DEFAULT_PHIS),
        help="Triploid fitness values to compare.",
    )
    parser.add_argument(
        "--upsilon-min",
        type=float,
        default=float(DEFAULT_UPSILONS[0]),
        help="Minimum upsilon value in the sweep.",
    )
    parser.add_argument(
        "--upsilon-max",
        type=float,
        default=float(DEFAULT_UPSILONS[-1]),
        help="Maximum upsilon value in the sweep.",
    )
    parser.add_argument(
        "--upsilon-count",
        type=int,
        default=int(DEFAULT_UPSILONS.size),
        help="Number of upsilon points in the sweep.",
    )
    parser.add_argument(
        "--radius-tolerance",
        type=float,
        default=DEFAULT_SOLVER_CONFIG.radius_tolerance,
        help="Target bisection tolerance for the critical radius.",
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=DEFAULT_SOLVER_CONFIG.dr,
        help="Radial grid spacing for the PDE solver.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_SOLVER_CONFIG.dt,
        help="Time step for the semi-implicit PDE solver.",
    )
    parser.add_argument(
        "--max-time-chunks",
        type=int,
        default=DEFAULT_SOLVER_CONFIG.max_time_chunks,
        help=(
            "Maximum integration budget in chunks. Unresolved cases are "
            "reported without assigning an expansion/collapse label."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print timing and PDE-solve diagnostics during the sweep.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run lightweight internal validation tests and exit.",
    )
    parser.add_argument(
        "--convergence-check",
        action="store_true",
        help=(
            "Run a short coarse/fine critical-radius comparison with "
            "dr=0.20, dt=0.25 versus dr=0.10, dt=0.10, then exit."
        ),
    )
    parser.add_argument(
        "--convergence-phi",
        type=float,
        default=0.1,
        help="Phi value used by --convergence-check.",
    )
    parser.add_argument(
        "--convergence-upsilon",
        type=float,
        default=0.02,
        help="Upsilon value used by --convergence-check.",
    )
    parser.add_argument(
        "--convergence-rtol",
        type=float,
        default=0.01,
        help="Relative tolerance for --convergence-check.",
    )
    parser.add_argument(
        "--convergence-atol",
        type=float,
        default=0.2,
        help="Absolute tolerance for --convergence-check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    require_positive("--sigma", float(args.sigma))
    require_positive("--dr", float(args.dr))
    require_positive("--dt", float(args.dt))
    require_positive("--radius-tolerance", float(args.radius_tolerance))
    require_positive("--convergence-rtol", float(args.convergence_rtol))
    require_positive("--convergence-atol", float(args.convergence_atol))
    if args.upsilon_count < 1:
        raise SystemExit("--upsilon-count must be at least 1.")
    if args.max_time_chunks < 1:
        raise SystemExit("--max-time-chunks must be at least 1.")

    if args.self_test:
        run_self_tests()
        if not args.convergence_check:
            return

    upsilons = np.linspace(args.upsilon_min, args.upsilon_max, args.upsilon_count)
    phis = tuple(float(value) for value in args.phis)
    config = replace(
        DEFAULT_SOLVER_CONFIG,
        radius_tolerance=args.radius_tolerance,
        dr=args.dr,
        dt=args.dt,
        max_time_chunks=args.max_time_chunks,
        profile=args.profile,
    )

    if args.convergence_check:
        run_convergence_check(
            sigma=float(args.sigma),
            phi=float(args.convergence_phi),
            upsilon=float(args.convergence_upsilon),
            config=config,
            relative_tolerance=float(args.convergence_rtol),
            absolute_tolerance=float(args.convergence_atol),
        )
        return

    curves = compute_comparison_curves(
        sigma=float(args.sigma),
        phis=phis,
        upsilons=upsilons,
        config=config,
    )
    plot_curves(curves, args.output.expanduser())

    if args.csv is not None:
        write_curves_csv(curves, args.csv.expanduser())
        print(f"\nSaved table to: {args.csv.expanduser()}")

    print(f"\nSaved figure to: {args.output.expanduser()}")
    print(
        "\nNote: near-threshold classifications can depend on integration "
        "time and grid resolution; validate important points by reducing "
        "--dr and --dt or by running --convergence-check.",
    )
    print("\nEstimated critical radii:\n")
    for curve in curves:
        print(f"phi = {curve.phi}")
        print("upsilon       numerical_Rc     asymptotic_Rc")
        for upsilon, numerical_radius, asymptotic_radius_value in zip(
            curve.upsilons,
            curve.numerical_radii,
            curve.asymptotic_radii,
        ):
            print(
                f"{upsilon:8.5f}   "
                f"{numerical_radius:12.6f}   "
                f"{asymptotic_radius_value:14.6f}"
            )
        print()


if __name__ == "__main__":
    main()
