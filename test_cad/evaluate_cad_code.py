import os
from argparse import ArgumentParser
from multiprocessing import Process
from typing import Callable, Optional

import numpy as np
import ot
import trimesh
from scipy.spatial import KDTree

OUTPUT_NAME = "r"


def sample_surface_points(
    obj: trimesh.Trimesh, num_samples: int = 1000, seed: int = 420
) -> np.ndarray:
    return trimesh.sample.sample_surface(obj, num_samples, seed=seed)[0]


def get_vertices(obj: trimesh.Trimesh, max_points: Optional[int] = None) -> np.ndarray:
    vertices = obj.vertices
    if max_points is not None and max_points > 0 and len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        return vertices[indices]
    return vertices


def eval_wrapper(error_value: float = 0.0, precision: int = 5, debug: bool = True):
    def decorator(function):
        def inner_wrapper(*args, **kwargs):
            try:
                return float(np.round(function(*args, **kwargs), decimals=precision))
            except Exception as e:
                if debug:
                    raise e
                return error_value

        return inner_wrapper

    return decorator


@eval_wrapper()
def iou(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    # Aim: Maximization
    # Range: 0-1
    intersection_volume = trimesh.boolean.intersection(
        [obj1, obj2], check_volume=False
    ).volume
    union_volume = trimesh.boolean.union([obj1, obj2], check_volume=False).volume

    if intersection_volume == 0 or union_volume == 0:
        return 0.0
    return intersection_volume / union_volume


@eval_wrapper()
def voxel_iou(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, resolution: int = 32):
    # Aim: Maximization
    # Range: 0-1
    v1 = obj1.voxelized(pitch=obj1.scale / resolution).matrix
    v2 = obj2.voxelized(pitch=obj2.scale / resolution).matrix

    min_shape = np.minimum(v1.shape, v2.shape)
    v1 = v1[: min_shape[0], : min_shape[1], : min_shape[2]]
    v2 = v2[: min_shape[0], : min_shape[1], : min_shape[2]]

    intersection = np.logical_and(v1, v2).sum()
    union = np.logical_or(v1, v2).sum()
    return intersection / union if union != 0 else 0.0


@eval_wrapper()
def inverse_chamfer_distance(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, num_samples: int = 5000
):
    # Aim: Maximization
    # Range: 0-1 (ONLY when scaling to unit scaled meshes)
    points1 = sample_surface_points(obj1, num_samples)
    points2 = sample_surface_points(obj2, num_samples)

    gt_distance, _ = KDTree(points1).query(points2, k=1)
    pred_distance, _ = KDTree(points2).query(points1, k=1)

    actual_distance = np.mean(np.square(gt_distance)) + np.mean(
        np.square(pred_distance)
    )

    # Following is only true on unit scaled meshes
    inverse_distance = 1.0 - actual_distance

    return inverse_distance

    # prox_query2 = trimesh.proximity.ProximityQuery(obj1)
    # prox_query1 = trimesh.proximity.ProximityQuery(obj2)

    # dist_1_to_2 = prox_query2.signed_distance(points1)
    # dist_2_to_1 = prox_query1.signed_distance(points2)
    # chamfer_dist = np.mean(dist_1_to_2**2) + np.mean(dist_2_to_1**2)

    # return chamfer_dist


@eval_wrapper()
def inverse_chamfer_distance_vertices(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, max_points: int = 5000
):
    # Aim: Maximization
    # Range: 0-1 (ONLY when scaling to unit scaled meshes)
    points1 = get_vertices(obj1, max_points)
    points2 = get_vertices(obj2, max_points)

    gt_distance, _ = KDTree(points1).query(points2, k=1)
    pred_distance, _ = KDTree(points2).query(points1, k=1)

    actual_distance = np.mean(np.square(gt_distance)) + np.mean(
        np.square(pred_distance)
    )

    # Following is only true on unit scaled meshes
    inverse_distance = 1.0 - actual_distance
    return inverse_distance


@eval_wrapper()
def inverse_hausdorff_distance(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, num_samples: int = 5000
):
    # Aim: Maximization
    # Range: 0-1 (ONLY when scaling to unit scaled meshes)
    points1 = sample_surface_points(obj1, num_samples)
    points2 = sample_surface_points(obj2, num_samples)

    gt_distance, _ = KDTree(points1).query(points2, k=1)
    pred_distance, _ = KDTree(points2).query(points1, k=1)

    actual_distance = max(np.max(gt_distance), np.max(pred_distance))
    # Following is only true on unit scaled meshes
    inverse_distance = 1.0 - actual_distance
    return inverse_distance


@eval_wrapper()
def inverse_hausdorff_distance_vertices(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, max_points: int = 5000
):
    # Aim: Maximization
    # Range: 0-1 (ONLY when scaling to unit scaled meshes)
    points1 = get_vertices(obj1, max_points)
    points2 = get_vertices(obj2, max_points)

    gt_distance, _ = KDTree(points1).query(points2, k=1)
    pred_distance, _ = KDTree(points2).query(points1, k=1)

    actual_distance = max(np.max(gt_distance), np.max(pred_distance))
    # Following is only true on unit scaled meshes
    inverse_distance = 1.0 - actual_distance
    return inverse_distance


@eval_wrapper()
def inverse_wasserstain_distance(
    obj1: trimesh.Trimesh, obj2: trimesh.Trimesh, num_samples: int = 1000
) -> float:
    # Aim: Maximization
    # Range: 0-1 (ONLY when scaling to unit scaled meshes)
    points1 = sample_surface_points(obj1, num_samples)
    points2 = sample_surface_points(obj2, num_samples)

    a = np.ones((num_samples,)) / num_samples
    b = np.ones((num_samples,)) / num_samples

    cost_matrix = ot.dist(points1, points2, metric="sqeuclidean")

    actual_distance: float = ot.emd2(a, b, cost_matrix)  # type: ignore
    # Following is only true on unit scaled meshes
    inverse_distance = 1.0 - actual_distance
    return inverse_distance


@eval_wrapper()
def volume_similarity(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    # Aim: Maximization
    # Range: 0-1
    # TODO: twice bigger and twice smaller give the same metric

    if not obj1.is_watertight or not obj2.is_watertight:
        return 0.0

    vol1 = float(obj1.volume)
    vol2 = float(obj2.volume)

    if vol1 + vol2 == 0:
        return 1.0
    return 1.0 - abs(vol1 - vol2) / max(vol1, vol2)


@eval_wrapper()
def area_similarity(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    # Aim: Maximization
    # Range: 0-1
    # TODO: twice bigger and twice smaller give the smae metric
    if not obj1.is_watertight or not obj2.is_watertight:
        return 0.0

    area1 = float(obj1.area)
    area2 = float(obj2.area)

    if max(area1, area2) == 0:
        return 1.0

    return 1.0 - abs(area1 - area2) / max(area1, area2)


@eval_wrapper()
def inverse_centroid_distance(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    # Aim: Maximization
    # Range: 0-1 (ONLY when scaling to unit scaled meshes)
    actual_distance = float(np.linalg.norm(obj1.centroid - obj2.centroid))
    # Following is only true on unit scaled meshes
    inverse_distance = 1.0 - actual_distance
    return inverse_distance


@eval_wrapper()
def inertia_similarity(obj1: trimesh.Trimesh, obj2: trimesh.Trimesh) -> float:
    # Aim: Maximization
    # Range: 0-1
    i1 = obj1.moment_inertia
    i2 = obj2.moment_inertia
    diff = np.linalg.norm(i1 - i2)
    norm = np.linalg.norm(i1) + np.linalg.norm(i2)
    return float(1.0 - diff / norm if norm != 0 else 1.0)


METRICS_DICT: dict[str, Callable[[trimesh.Trimesh, trimesh.Trimesh], float]] = {
    "iou": iou,
    # "giou": giou,
    "viou": voxel_iou,
    "cd": inverse_chamfer_distance,
    "cdv": inverse_chamfer_distance_vertices,
    "hd": inverse_hausdorff_distance,
    "hdv": inverse_hausdorff_distance_vertices,
    "wd": inverse_wasserstain_distance,
    "vs": volume_similarity,
    "as": area_similarity,
    # "bbiou": bbiou,
    "ctd": inverse_centroid_distance,
    "is": inertia_similarity,
}


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def py_file_to_mesh_and_brep_files(py_path, mesh_path):
    try:
        with open(py_path, OUTPUT_NAME) as f:
            py_string = f.read()
        exec(py_string, globals())
        compound = globals()[OUTPUT_NAME].val()
        mesh = compound_to_mesh(compound)
        assert len(mesh.faces) > 2
        mesh.export(mesh_path)
    except Exception as e:
        print(e)


def transform(obj: trimesh.Trimesh) -> trimesh.Trimesh:
    center = (obj.bounds[0] + obj.bounds[1]) / 2.0
    obj = obj.apply_translation(-center)
    extent = np.max(obj.extents)
    if extent > 1e-7:
        obj = obj.apply_scale(1.0 / extent)
    return obj.apply_transform(
        trimesh.transformations.translation_matrix([0.5, 0.5, 0.5])
    )


def py_file_to_mesh_and_brep_files_safe(py_path, mesh_path):
    process = Process(target=py_file_to_mesh_and_brep_files, args=(py_path, mesh_path))
    process.start()
    process.join(3)

    if process.is_alive():
        print("process alive:", py_path)
        process.terminate()
        process.join()


def evaluate(
    obj1: trimesh.Trimesh,
    obj2: trimesh.Trimesh,
    metrics_dict: dict[str, Callable[[trimesh.Trimesh, trimesh.Trimesh], float]],
) -> dict[str, float]:
    transformed_obj1 = transform(obj1.copy())
    transformed_obj2 = transform(obj2.copy())
    return {
        name: metric_fn(transformed_obj1, transformed_obj2)
        for name, metric_fn in metrics_dict.items()
    }


def run_cd_single(pred_py_path, pred_mesh_path, gt_mesh_path) -> dict:
    py_file_to_mesh_and_brep_files_safe(pred_py_path, pred_mesh_path)

    metrics = {name: 0 for name in METRICS_DICT}
    try:  # Apply transformations
        pred_mesh = transform(trimesh.load_mesh(pred_mesh_path))

        gt_mesh = transform(trimesh.load_mesh(os.path.join(gt_mesh_path)))
        return evaluate(pred_mesh, gt_mesh, metrics_dict=METRICS_DICT)
    except Exception as e:
        print(e)

    return metrics


def run_evaluator(gt_mesh_path, pred_py_path) -> dict:
    pred_mesh_path = os.path.join(os.path.dirname(pred_py_path), "tmp_mesh.stl")

    try:
        os.remove(pred_mesh_path)
    except FileNotFoundError:
        pass

    metrics = run_cd_single(
        pred_py_path=pred_py_path,
        pred_mesh_path=pred_mesh_path,
        gt_mesh_path=gt_mesh_path,
    )

    try:
        os.remove(pred_mesh_path)
    except FileNotFoundError:
        pass

    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-b", "--baseline-mesh-path", type=str, default="./baseline.stl"
    )
    parser.add_argument("-c", "--py-code-path", type=str, default="./code.py")
    args = parser.parse_args()
    run_evaluator(args.baseline_mesh_path, args.py_code_path)
