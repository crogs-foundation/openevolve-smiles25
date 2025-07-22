"""
Evaluator for the function minimization example
"""

import concurrent.futures
import importlib.util
import tempfile
import traceback
from pathlib import Path

import cadquery as cq
import trimesh
from compute_metrics import run_compute

# PRED_MESH_PATH = "pred.stl"
GROUND_TRUTH_MESH_PATH = Path("./test_cad/holed_box.stl")


def run_with_timeout(func, *args, timeout_seconds=5, **kwargs):
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


def safe_float(value):
    """Convert a value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} of type {type(value)} to float")
        return 0.0


def cq_to_trimesh(workplane: cq.Workplane) -> trimesh.Trimesh:
    """
    Converts a CadQuery Workplane object into a trimesh.Trimesh object.
    """
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
        workplane.export(temp_file.name)
        mesh = trimesh.load_mesh(temp_file.name)
    return mesh


def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        assert spec is not None and spec.loader is not None, (
            "Failed to load program module"
        )
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "build_3d_figure"):
            print("Error: program does not have the main 'build_3d_figure' function")
            return {"combined_score": 0.0} | {
                "error": "Missing main 'build_3d_figure' function"
            }

        predicted_mesh = cq_to_trimesh(program.build_3d_figure())

        metrics = run_with_timeout(
            run_compute,
            GROUND_TRUTH_MESH_PATH,
            predicted_mesh,
            timeout_seconds=120,
        )

        return {"combined_score": sum(v for v in metrics.values()), **metrics}

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {"combined_score": 0.0, "error": str(e)}
