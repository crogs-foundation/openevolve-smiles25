import importlib.util
import numpy as np
import time
import traceback
from scipy.spatial import ConvexHull, distance
from itertools import combinations
import concurrent.futures

# Default global settings
additional_dim = 10

base_dim = 3
n_dim = base_dim + additional_dim
k_points = base_dim + additional_dim + 1
radius = 0.5
tol = 1e-6


class TimeoutError(Exception):
    pass


def validate_packing(points: np.ndarray, *, radius: float, tol: float) -> bool:
    """
    Validate that all points lie on the sphere of given radius (within tolerance).
    """
    # squared distances to avoid repeated sqrt
    sq_norms = np.sum(points**2, axis=1)
    target_sq = radius**2
    if not np.all(np.abs(sq_norms - target_sq) <= tol):
        return False
    return True


def build_cones(points: np.ndarray) -> np.ndarray:
    """Return facets of the convex hull as index simplices."""
    return ConvexHull(points).simplices


def max_cone_diameter(
    points: np.ndarray, cones: np.ndarray, *, radius: float
) -> tuple[float, list[float]]:
    """
    Compute the maximum diameter among all cones: the largest distance between any two vertices,
    but at least the sphere diameter (2*radius).
    """
    max_diam = radius
    diams = []
    for cone in cones:
        pts = points[cone]
        # pairwise distances
        dists = [distance.euclidean(p1, p2) for p1, p2 in combinations(pts, 2)]
        cone_max = max(dists, default=0.0)
        cone_max = max(cone_max, radius)
        diams.append(cone_max)
        if cone_max > max_diam:
            max_diam = cone_max
    return max_diam, diams


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds: float = 5):
    """
    Execute a callable with a timeout using ThreadPoolExecutor.
    """
    if kwargs is None:
        kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def evaluate(
    program_path: str,
    n_dim: int = n_dim,
    k_points: int = k_points,
    radius: float = radius,
    tol: float = tol,
    timeout_seconds: float = 120,
) -> dict:
    """
    Evaluate an external packing program.

    Args:
        program_path: path to Python file implementing run_packing(...).
        n_dim: dimension of the sphere.
        k_points: number of points to pack.
        radius: radius of the sphere.
        tol: tolerance for validation.
        timeout_seconds: max runtime for user code.
    Returns:
        Metrics including max_diam, target_ratio, validity, eval_time, combined_score.
    """
    target_val = 1
    teorethical_target = radius * np.sqrt(2 * (n_dim + 1) / n_dim)
    start = time.time()

    try:
        # dynamically load user program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "run_packing"):
            return {"error": "Missing run_packing function"}

        # pass globals as named args to user's function
        points = run_with_timeout(
            program.run_packing,
            kwargs={"n_dim": n_dim, "k_points": k_points, "radius": radius},
            timeout_seconds=timeout_seconds,
        )

        duration = time.time() - start

        # convert to numpy array
        pts = np.array(points, dtype=float)

        # shape check
        valid = pts.shape == (k_points, n_dim)

        # validate on-sphere
        if valid and not validate_packing(pts, radius=radius, tol=tol):
            valid = False

        # check if origin lies in convex hull of pts
        center = np.zeros(n_dim)
        hull = ConvexHull(pts)
        A, b = hull.equations[:, :-1], hull.equations[:, -1]
        inside = np.all(np.dot(A, center) + b <= tol)

        # compute max diameter
        if inside and valid:
            cones = hull.simplices
            max_diam, _ = max_cone_diameter(pts, cones, radius=radius)
        else:
            max_diam = 2 * radius
            valid = False

        target_ratio = (target_val / max_diam) if valid else 0.0
        validity = 1.0 if valid else 0.0
        combined = validity * target_ratio

        print(
            f"Evaluation: valid={valid}, max_diam={max_diam:.6f}, "
            f"target={target_val:.6f}, ratio={target_ratio:.6f}, time={duration:.2f}s"
        )

        return {
            "max_diam": float(max_diam),
            "target_ratio": float(target_ratio),
            "validity": validity,
            # "eval_time": duration,
            "combined_score": combined,
            "theoretical": teorethical_target,
            "dim": n_dim,
        }

    except Exception:
        traceback.print_exc()
        return {
            "max_diam": 2 * radius,
            "target_ratio": 0.0,
            "validity": 0.0,
            # "eval_time": 0.0,
            "combined_score": 0.0,
            "theoretical": teorethical_target,
            "dim": n_dim,
        }
