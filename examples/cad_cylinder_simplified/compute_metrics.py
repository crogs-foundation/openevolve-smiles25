from pathlib import Path
from typing import Callable, Optional

import numpy as np
import ot
import trimesh
from scipy.spatial import KDTree

# This script assumes a CAD library like 'cadquery' or 'build123d' is installed,
# as it executes code that calls methods like .val() and .tessellate().
# The executed script is expected to place the final CAD object in a variable named 'r'.
OUTPUT_NAME = "r"
ALIGN_MESH = True  # TODO: if true - errors


def eval_wrapper(error_value: float = 0.0, precision: int = 5, debug: bool = False):
    """Decorator to handle exceptions, round results, and provide a default error value."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return round(float(result), precision)
            except Exception as e:
                if debug:
                    raise
                print(f"Error in {func.__name__}: {e}")
                return error_value

        return wrapper

    return decorator


# --- Helper Functions ---


def sample_surface_points(
    obj: trimesh.Trimesh, num_samples: int, seed: int = 420
) -> np.ndarray:
    """Samples points from a mesh surface."""
    return trimesh.sample.sample_surface(obj, num_samples, seed=seed)[0]


def get_vertices(obj: trimesh.Trimesh, max_points: Optional[int] = None) -> np.ndarray:
    """Gets (or subsamples) vertices from a mesh."""
    vertices = obj.vertices
    if max_points and len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        return vertices[indices]
    return vertices


def _get_point_distances(
    points1: np.ndarray, points2: np.ndarray
) -> tuple[float, float]:
    """Helper to compute Chamfer and Hausdorff distances between point clouds."""
    tree1, tree2 = KDTree(points1), KDTree(points2)
    dist1, _ = tree1.query(points2, k=1)
    dist2, _ = tree2.query(points1, k=1)
    chamfer_dist = np.mean(np.square(dist1)) + np.mean(np.square(dist2))
    hausdorff_dist = max(np.max(dist1), np.max(dist2))
    return chamfer_dist, hausdorff_dist


def _scalar_similarity(val1: float, val2: float) -> float:
    """Helper to compute similarity between two scalar values."""
    maximum = max(val1, val2)
    return 1.0 - abs(val1 - val2) / maximum if maximum > 0 else 1.0


# --- Metric Functions ---


@eval_wrapper()
def iou(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    intersection = obj1.intersection(obj2, check_volume=False).volume
    union = obj1.union(obj2, check_volume=False).volume
    return intersection / union if union > 0 else 0.0


@eval_wrapper()
def voxel_iou(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, resolution: int = 32
) -> float:
    v1 = obj1.voxelized(pitch=obj1.scale / resolution).matrix
    v2 = obj2.voxelized(pitch=obj2.scale / resolution).matrix

    shape1, shape2 = np.array(v1.shape), np.array(v2.shape)
    max_shape = np.maximum(shape1, shape2)

    v1_padded = np.zeros(max_shape, dtype=bool)
    v1_padded[tuple(map(slice, shape1))] = v1

    v2_padded = np.zeros(max_shape, dtype=bool)
    v2_padded[tuple(map(slice, shape2))] = v2

    intersection = np.logical_and(v1_padded, v2_padded).sum()
    union = np.logical_or(v1_padded, v2_padded).sum()
    return intersection / union if union > 0 else 0.0


@eval_wrapper()
def inverse_chamfer_distance(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, num_samples: int = 5000
) -> float:
    points1 = sample_surface_points(obj1, num_samples)
    points2 = sample_surface_points(obj2, num_samples)
    chamfer, _ = _get_point_distances(points1, points2)
    return 1.0 - chamfer


@eval_wrapper()
def inverse_chamfer_distance_vertices(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, max_points: int = 5000
) -> float:
    points1 = get_vertices(obj1, max_points)
    points2 = get_vertices(obj2, max_points)
    chamfer, _ = _get_point_distances(points1, points2)
    return 1.0 - chamfer


@eval_wrapper()
def inverse_hausdorff_distance(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, num_samples: int = 5000
) -> float:
    points1 = sample_surface_points(obj1, num_samples)
    points2 = sample_surface_points(obj2, num_samples)
    _, hausdorff = _get_point_distances(points1, points2)
    return 1.0 - hausdorff


@eval_wrapper()
def inverse_hausdorff_distance_vertices(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, max_points: int = 5000
) -> float:
    points1 = get_vertices(obj1, max_points)
    points2 = get_vertices(obj2, max_points)
    _, hausdorff = _get_point_distances(points1, points2)
    return 1.0 - hausdorff


@eval_wrapper()
def inverse_wasserstein_distance(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, num_samples: int = 1000
) -> float:
    points1 = sample_surface_points(obj1, num_samples)
    points2 = sample_surface_points(obj2, num_samples)
    a = b = np.ones((num_samples,)) / num_samples
    cost_matrix = ot.dist(points1, points2, metric="sqeuclidean")
    emd2 = ot.emd2(a, b, cost_matrix)
    return 1.0 - emd2  # type: ignore


@eval_wrapper()
def volume_similarity(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    if not obj1.is_watertight or not obj2.is_watertight:
        return 0.0
    return _scalar_similarity(obj1.volume, obj2.volume)


@eval_wrapper()
def area_similarity(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    return _scalar_similarity(obj1.area, obj2.area)


@eval_wrapper()
def inverse_centroid_distance(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    distance = np.linalg.norm(obj1.centroid - obj2.centroid)
    return float(1.0 - distance)


@eval_wrapper()
def inertia_similarity(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    i1, i2 = obj1.moment_inertia, obj2.moment_inertia
    norm = np.linalg.norm(i1) + np.linalg.norm(i2)
    if norm == 0:
        return 1.0
    return float(1.0 - np.linalg.norm(i1 - i2) / norm)


METRICS_DICT: dict[str, Callable] = {
    "iou": iou,
    "viou": voxel_iou,
    "cd": inverse_chamfer_distance,
    "cdv": inverse_chamfer_distance_vertices,
    "hd": inverse_hausdorff_distance,
    "hdv": inverse_hausdorff_distance_vertices,
    "wd": inverse_wasserstein_distance,
    "vs": volume_similarity,
    "as": area_similarity,
    "ctd": inverse_centroid_distance,
    "is": inertia_similarity,
}


def align_mesh(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    n_samples: int = 5000,
    icp_iterations: int = 50,
) -> trimesh.Trimesh:
    source_points = source_mesh.sample(n_samples)
    target_points = target_mesh.sample(n_samples)

    # ----- Step 1: Procrustes Alignment -----
    matrix_procrustes, _, _ = trimesh.registration.procrustes(
        source_points, target_points, reflection=True, scale=False, translation=False
    )

    source_mesh_procrustes = source_mesh.copy()
    source_mesh_procrustes.apply_transform(matrix_procrustes)

    source_points_aligned = source_mesh_procrustes.sample(n_samples)

    # ----- Step 2: ICP Fine Alignment -----
    matrix_icp, _, _ = trimesh.registration.icp(
        source_points_aligned,
        target_points,
        max_iterations=icp_iterations,
    )

    # Combine both transformations
    matrix_total = matrix_icp @ matrix_procrustes

    source_mesh_aligned = source_mesh.copy()
    source_mesh_aligned.apply_transform(matrix_total)

    return source_mesh_aligned


def transform(obj: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalizes a mesh to be centered and fit within a unit cube."""
    center = obj.bounds.mean(axis=0)
    obj.apply_translation(-center)
    scale = obj.extents.max()
    if scale > 1e-7:
        obj.apply_scale(1.0 / scale)
    return obj.apply_transform(trimesh.transformations.translation_matrix([0, 0, 0]))


def evaluate(
    target_obj: trimesh.Trimesh, predicted_obj: trimesh.Trimesh
) -> dict[str, float]:
    """Computes all metrics for two (normalized) meshes."""
    target_obj = transform(target_obj.copy())

    predicted_obj = transform(predicted_obj.copy())
    if ALIGN_MESH:
        # Align the predicted mesh to the target mesh
        predicted_obj = align_mesh(predicted_obj, target_obj)

    return {
        name: metric_fn(target_obj, predicted_obj)
        for name, metric_fn in METRICS_DICT.items()
    }


def run_compute(gt_mesh_path: Path, mesh: trimesh.Trimesh) -> dict[str, float]:
    """Main function to run the full evaluation pipeline."""
    error_metrics = {name: 0.0 for name in METRICS_DICT}

    try:
        pred_mesh = mesh.copy()
        gt_mesh = trimesh.load_mesh(gt_mesh_path)
        metrics = evaluate(pred_mesh, gt_mesh)
    except Exception as e:
        print(f"Error during mesh loading or evaluation: {e}")
        return error_metrics

    print(metrics)
    return metrics
