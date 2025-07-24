"""
Evaluator for sphere partitions into parts of smaller diameter
"""

import importlib.util
import numpy as np
import time
import traceback
from scipy.spatial import ConvexHull, distance
from itertools import combinations
import concurrent.futures

n_dim = 3
k_points = 4
radius = 0.5
tol = 1e-6


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(points, radius=0.5, tol=1e-6):
    """
    Validate that points are correct

    Args:
        points: np.array of shape (num_points, n_dim) with points coordinates
        radius: the radius of the sphere

    Returns:
        True if valid, False otherwise
    """

    # check that points are located on the sphere with radius r
    if not np.all(
        np.abs(np.linalg.norm(points, axis=1, keepdims=True) - radius * radius <= tol)
    ):
        return False

    return True


def build_cones(points):
    """Возвращает грани выпуклой оболочки как конусы."""
    return ConvexHull(points).simplices


def max_cone_diameter(points, cones, radius=0.5):
    """Вычисляет максимальный диаметр конусов, учитывая расстояние от центра до сферы."""
    max_diameter = 0  # Начинаем с диаметра шара

    diams = []
    for cone in cones:
        cone_points = points[cone]
        # Расстояния между точками конуса
        pairwise_dist = [
            distance.euclidean(p1, p2) for p1, p2 in combinations(cone_points, 2)
        ]

        # Вычисляем диаметр конуса
        current_max = max(max(pairwise_dist), radius)
        diams.append(current_max)
        if current_max > max_diameter:
            max_diameter = current_max

    return max_diameter, diams


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def evaluate(program_path, n_dim=3, k_points=4, radius=0.5):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    TARGET_VALUE = radius * np.sqrt(2 * (n_dim + 1) / n_dim)  # SPHERE DIAMETER

    start_time = time.time()

    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_packing"):
            print("Error: program does not have 'run_packing' function")
            return {
                "error": "Missing run_packing function",
            }

        points = run_with_timeout(program.run_packing, timeout_seconds=120)

        end_time = time.time()
        eval_time = end_time - start_time

        # Ensure points is a numpy array
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # Validate solution

        # Check shape
        shape_valid = points.shape == (k_points, n_dim)
        if not shape_valid:
            print(f"Invalid shapes: {points.shape}, expected ({k_points}, {n_dim})")
            valid = False
        else:
            valid = True

        # Validate packing
        if not validate_packing(points):
            valid = False

        # checking for the center of sphere
        center = np.zeros(points.shape[1])
        hull = ConvexHull(points)
        A = hull.equations[:, :-1]
        b = hull.equations[:, -1]

        center_in_convex_hull = np.all(np.dot(A, center) + b <= tol)
        max_diam = 2 * radius

        # compute maximal diameter among all partition items
        if center_in_convex_hull:
            cones = ConvexHull(points).simplices
            max_diam, _ = max_cone_diameter(points, cones, radius)

        # Target ratio (how close we are to the target)
        target_ratio = TARGET_VALUE / max_diam if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = validity * target_ratio

        print(
            f"Evaluation: valid={valid}, max_diameter={max_diam:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        return {
            "max_diam": float(max_diam),
            "target_ratio": float(target_ratio),
            "validity": float(validity),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "max_diam": 2 * radius,
            "target_ratio": 0.0,
            "validity": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }
