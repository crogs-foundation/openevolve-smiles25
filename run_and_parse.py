import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path


def get_command(experiment_path: str | Path) -> list[str]:
    return [
        "uv",
        "run",
        "openevolve-run.py",
        f"{experiment_path}/initial_program.py",
        f"{experiment_path}/evaluator.py",
        "--config",
        f"{experiment_path}/config.yaml",
    ]


# The number of times to run the command.
NUM_RUNS = 5

# The base name for the output folders created by your command.
# The script will look for folders starting with this prefix.
FOLDER_PREFIX = "openevolve_output_"

# The name of the final aggregated CSV file.
OUTPUT_CSV_PREFIX = "aggregated_results"


def run_command(command, n_times, verbose: bool):
    """
    Executes the given command n_times.
    """
    print(f"--- 2. EXECUTING COMMAND {n_times} TIMES ---")
    for i in range(n_times):
        print(f"[*] Starting run {i + 1} of {n_times}...")
        try:
            values = {}
            if not verbose:
                values = dict(capture_output=True, text=True)
            subprocess.run(command, check=True, **values)  # type: ignore
            print(f"[+] Finished run {i + 1} of {n_times}.")
        except subprocess.CalledProcessError as e:
            print(
                f"[!] ERROR on run {i + 1}: Command failed with exit code {e.returncode}"
            )
            print(f"    STDOUT: {e.stdout}")
            print(f"    STDERR: {e.stderr}")

    print("-" * 20 + "\n")


def parse_results(
    n_to_parse: int, folder_prefix: str, working_dir: Path, output_csv_path: Path
):
    """
    Finds the latest N folders in the working_dir, parses them, and writes to a CSV.
    """
    print(f"--- 3. PARSING RESULTS FROM '{working_dir}' ---")
    try:
        # Use Path.glob() to find matching folders
        all_folders = list(working_dir.glob(f"{folder_prefix}*"))

        if not all_folders:
            print(
                f"[!] No folders found in '{working_dir}' with prefix '{folder_prefix}'. Exiting."
            )
            return

        # Sort folders by name (which includes the timestamp)
        all_folders.sort(key=lambda p: p.name, reverse=True)

        latest_folders = all_folders[:n_to_parse]
        print(
            f"[*] Found {len(all_folders)} total folders. Processing the latest {len(latest_folders)}."
        )

    except Exception as e:
        print(f"[!] Error finding folders: {e}")
        return

    all_results = []
    for folder_path in latest_folders:
        print(f"  -> Processing folder: {folder_path.name}")  # Use .name attribute
        try:
            json_path = folder_path / "best" / "best_program_info.json"
            code_path = folder_path / "best" / "best_program.py"

            # Use .read_text() for cleaner file reading
            info_data = json.loads(json_path.read_text())
            code_content = code_path.read_text()

            run_data = {"code": code_content, **info_data.get("metrics", {})}
            all_results.append(run_data)

        except FileNotFoundError as e:
            print(
                f"    [!] Warning: Missing file in {folder_path.name}: {e.filename}. Skipping."
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"    [!] Warning: Could not process data in {folder_path.name} due to {type(e).__name__}. Skipping."
            )

    if not all_results:
        print("[!] No data was successfully parsed. CSV file will not be created.")
        return

    try:
        headers = list(all_results[0].keys())
        # Use Path.open() to write the CSV
        with output_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_results)

        print(
            f"\n[+] Success! Aggregated results from {len(all_results)} runs into '{output_csv_path}'."
        )

    except Exception as e:
        print(f"[!] An error occurred while writing the CSV file: {e}")


if __name__ == "__main__":
    # --- NEW: Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Run an experiment N times and parse the results into a single CSV."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Experiment's directory",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=10,
        help="The number of times to execute the command. Defaults to 10.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="print execution information (default: False)",
    )

    args = parser.parse_args()

    num_runs_from_cli = args.num_runs
    experiment_path = Path(f"./experiments/{args.directory}")
    assert experiment_path.exists()

    # Step 1: Run the experiments N times
    run_command(get_command(experiment_path), num_runs_from_cli, args.verbose)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{OUTPUT_CSV_PREFIX}_{timestamp_str}.csv"
    full_output_path = experiment_path / output_filename

    # Step 2: Parse the N latest results and generate the CSV
    parse_results(num_runs_from_cli, FOLDER_PREFIX, experiment_path, full_output_path)
