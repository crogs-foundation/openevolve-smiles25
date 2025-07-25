from itertools import permutations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import ot
import trimesh
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

ALIGN_MESH = True


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


def get_pca_axes(mesh: trimesh.Trimesh, n_components: int = 3):
    pca = PCA(n_components=n_components)
    pca.fit(mesh.vertices)
    return pca.components_  # (3, 3), строки — оси


def best_pca_alignment_score(axes1, axes2):
    best_score = 0.0
    for perm in permutations(range(3)):
        permuted_axes2 = axes2[list(perm)]
        cosines = np.abs(np.sum(axes1 * permuted_axes2, axis=1))  # cosine similarity
        score = np.mean(cosines)
        best_score = max(best_score, score)
    return best_score


@eval_wrapper()
def orientation_similarity_pca_invariant(
    mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh
):
    axes1 = get_pca_axes(mesh1)
    axes2 = get_pca_axes(mesh2)
    return best_pca_alignment_score(axes1, axes2)


@eval_wrapper()
def orientation_similarity_faces(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh):
    if len(mesh1.face_normals) != len(mesh2.face_normals):
        return 0
    n1 = mesh1.face_normals
    n2 = mesh2.face_normals

    dot = np.abs(np.sum(n1 * n2, axis=1))
    score = np.mean(dot)
    return score


@eval_wrapper()
def orientation_similarity_vertices(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh):
    if mesh1.vertices.shape != mesh2.vertices.shape:
        return 0

    # Center the meshes at the origin
    v1 = mesh1.vertices - mesh1.vertices.mean(axis=0)
    v2 = mesh2.vertices - mesh2.vertices.mean(axis=0)

    v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

    cosines = np.sum(v1_norm * v2_norm, axis=1)  # dot products
    cosines = np.clip(cosines, -1.0, 1.0)  # numerical safety

    score = np.mean(cosines)

    normalized_score = (score + 1) / 2
    return normalized_score


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


ORIENT_METRICS_DICT: dict[str, Callable] = {
    "osi": orientation_similarity_pca_invariant,
    "osf": orientation_similarity_faces,
    "osv": orientation_similarity_vertices,
}


def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Get the centroid of the mesh
    centroid = mesh.centroid

    # Create a translation matrix
    T = np.eye(4)
    T[:3, 3] = -centroid  # translate by negative centroid

    # Apply transformation
    centered = mesh.copy()
    centered.apply_transform(T)
    return centered


def transform(obj: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalizes a mesh to be centered and fit within a unit cube."""
    center = obj.bounds.mean(axis=0)
    obj.apply_translation(-center)
    scale = obj.extents.max()
    if scale > 1e-7:
        # if scale > 1:
        obj.apply_scale(1.0 / scale)
    return center_mesh(obj)


def align_rot(
    source_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh
) -> trimesh.Trimesh:
    # 1. Center both meshes
    def center(mesh):
        return mesh.vertices - mesh.centroid

    source_centered = center(source_mesh)
    target_centered = center(target_mesh)

    source_centered = source_centered[: min(len(source_centered), len(target_centered))]
    target_centered = target_centered[: min(len(source_centered), len(target_centered))]

    # 2. Solve for best orthogonal matrix (rotation or reflection)
    U, _, Vt = np.linalg.svd(target_centered.T @ source_centered)
    R_opt = U @ Vt

    # 3. If it's a reflection (det < 0), try both R and reflected R
    if np.linalg.det(R_opt) < 0:
        # Try reflection
        R_reflect = U @ np.diag([1, 1, -1]) @ Vt
        error_original = np.linalg.norm(source_centered @ R_opt.T - target_centered)
        error_reflect = np.linalg.norm(source_centered @ R_reflect.T - target_centered)

        # Choose the better one
        if error_reflect < error_original:
            R_opt = R_reflect

    # 4. Apply transform
    T = np.eye(4)
    T[:3, :3] = R_opt

    mesh_a_aligned = source_mesh.copy()
    mesh_a_aligned.apply_transform(T)

    # 5. Optional: Check alignment
    return mesh_a_aligned


def evaluate(
    target_obj: trimesh.Trimesh, predicted_obj: trimesh.Trimesh
) -> dict[str, float]:
    """Computes all metrics for two (normalized) meshes."""
    target_obj = transform(target_obj.copy())

    predicted_obj = transform(predicted_obj.copy())
    aligned_obj = predicted_obj.copy()
    if ALIGN_MESH:
        aligned_obj = transform(align_rot(aligned_obj, target_obj))

    return {
        **{
            name: metric_fn(target_obj, aligned_obj)
            for name, metric_fn in METRICS_DICT.items()
        },
        **{
            name: metric_fn(target_obj, predicted_obj)
            for name, metric_fn in ORIENT_METRICS_DICT.items()
        },
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
