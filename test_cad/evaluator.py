"""
Evaluator for the function minimization example
"""

import concurrent.futures
import importlib.util
import traceback
from pathlib import Path

import trimesh
from evaluate_cad_code import METRICS_DICT, run_evaluation2

# PRED_MESH_PATH = "pred.stl"
GROUND_TRUTH_MESH_PATH = Path("./test_cad/holed_box.stl")


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


def safe_float(value):
    """Convert a value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} of type {type(value)} to float")
        return 0.0


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
        assert spec is not None
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "r"):
            print("Error: program does not have 'r' variable")
            return {name: 0.0 for name in METRICS_DICT} | {
                "error": "Missing 'r' variable"
            }

        program.r.export("jopa.stl")

        metrics = run_with_timeout(
            run_evaluation2,
            (GROUND_TRUTH_MESH_PATH, trimesh.load_mesh("jopa.stl")),
            timeout_seconds=120,
        )

        return metrics

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {name: 0.0 for name in METRICS_DICT} | {"error": str(e)}
