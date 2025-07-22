from multiprocessing import Process
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

# --- Core Logic ---


def py_script_to_mesh_file(py_path: Path, mesh_path: Path):
    """Executes a python script to generate a CAD object and saves it as a mesh."""

    # print(py_path)
    # print(mesh_path)

    with open(py_path, "r") as f:
        py_string = f.read()

    # print(py_string)

    context = {}
    exec(py_string, context)

    print("LOG 1")

    compound = context[OUTPUT_NAME].val()
    vertices, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh(vertices=[(v.x, v.y, v.z) for v in vertices], faces=faces)
    print("LOG 2")

    if len(mesh.faces) < 3:
        raise ValueError("Generated mesh has too few faces.")
    mesh.export(str(mesh_path))


def _convert_script_to_mesh_safe(
    py_path: Path, mesh_path: Path, timeout: int = 15
) -> bool:
    """Runs script-to-mesh conversion in a separate process with a timeout."""
    process = Process(target=py_script_to_mesh_file, args=(py_path, mesh_path))

    process.start()
    process.join(timeout)

    if process.is_alive():
        print(f"Process for {py_path.name} timed out. Terminating.")
        process.terminate()
        process.join()
        return False

    if process.exitcode != 0:
        print(f"Process for {py_path.name} failed with exit code {process.exitcode}.")
        return False

    return mesh_path.exists() and mesh_path.stat().st_size > 0


def transform(obj: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalizes a mesh to be centered and fit within a unit cube."""
    center = obj.bounds.mean(axis=0)
    obj.apply_translation(-center)
    scale = obj.extents.max()
    if scale > 1e-7:
        obj.apply_scale(1.0 / scale)
    return obj.apply_transform(
        trimesh.transformations.translation_matrix([0.5, 0.5, 0.5])
    )


def evaluate(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> dict[str, float]:
    """Computes all metrics for two (normalized) meshes."""
    t_obj1 = transform(obj1.copy())
    t_obj2 = transform(obj2.copy())
    return {name: metric_fn(t_obj1, t_obj2) for name, metric_fn in METRICS_DICT.items()}


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


# if __name__ == "__main__":
# parser = ArgumentParser(description="Evaluate 3D mesh similarity.")
# parser.add_argument(
#     "-b",
#     "--baseline-mesh-path",
#     type=Path,
#     required=True,
#     help="Path to the ground truth mesh file (e.g., .stl, .obj).",
# )
# parser.add_argument(
#     "-c",
#     "--py-code-path",
#     type=Path,
#     required=True,
#     help="Path to the Python script that generates the predicted mesh.",
# )
# args = parser.parse_args()

# if not args.baseline_mesh_path.is_file():
#     print(f"Error: Baseline mesh file not found at {args.baseline_mesh_path}")
#     exit(1)
# if not args.py_code_path.is_file():
#     print(f"Error: Python code file not found at {args.py_code_path}")
#     exit(1)


# mesh = trimesh.load_mesh(temp_file.name)

# run_compute(Path(args.baseline_mesh_path), Path(args.py_code_path))
