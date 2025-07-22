"""
Evaluator for the function minimization example
"""

import importlib.util
import concurrent.futures
import traceback
from evaluate_cad_code import METRICS_DICT, run_evaluation

# PRED_MESH_PATH = "pred.stl"
GROUND_TRUTH_MESH_PATH = "test_cad/a.stl"


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
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "save_evolve_block"):
            print("Error: program does not have 'save_evolve_block' function")
            return {name: 0.0 for name in METRICS_DICT} | {
                "error": "Missing run_build function"
            }

        pred_py_path = run_with_timeout(program.save_evolve_block, timeout_seconds=120)

        metrics = run_evaluation(GROUND_TRUTH_MESH_PATH, pred_py_path)

        return metrics

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {name: 0.0 for name in METRICS_DICT} | {"error": str(e)}
