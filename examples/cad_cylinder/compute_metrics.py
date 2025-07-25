from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d
import ot
import trimesh
from scipy.spatial import KDTree

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
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, resolution: int = 64
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
    # "cdv": inverse_chamfer_distance_vertices,
    "hd": inverse_hausdorff_distance,
    # "hdv": inverse_hausdorff_distance_vertices,
    "wd": inverse_wasserstein_distance,
    "vs": volume_similarity,
    "as": area_similarity,
    # "ctd": inverse_centroid_distance,
    "is": inertia_similarity,
}


ORIENT_METRICS_DICT: dict[str, Callable] = {
    # "osi": orientation_similarity_pca_invariant,
    # "osf": orientation_similarity_faces,
    # "osv": orientation_similarity_vertices,
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


def tri_to_o(trimesh_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(trimesh_mesh.vertices)
    triangles = np.asarray(trimesh_mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return o3d_mesh


def o_to_tri(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def align_rot(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    n_points: int = 20000,
    # voxel_size: float = 0.5,
    voxel_size: float = 0.01,
) -> trimesh.Trimesh:
    # Convert to Open3D triangle meshes and sample point clouds
    target_pcd = tri_to_o(target_mesh).sample_points_uniformly(n_points)
    source_pcd = tri_to_o(source_mesh).sample_points_uniformly(n_points)
    # Preprocess point clouds
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    # Register point clouds
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )

    # Transform original Open3D mesh and convert back to trimesh
    source_o3d = tri_to_o(source_mesh)
    source_o3d.transform(result_ransac.transformation)

    return o_to_tri(source_o3d)


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
