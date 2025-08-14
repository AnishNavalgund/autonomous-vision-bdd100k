"""
BDD100K data processing pipeline.
Runs parser and COCO converter in parallel with basic error handling.
"""

import multiprocessing as mp
import time

from autonomous_vision.config import Config
from autonomous_vision.data_parser.bdd_to_coco import main as coco_main
from autonomous_vision.data_parser.parser_core import main as parser_main


def run_parser():
    """Run the parser script."""
    try:
        print("Starting parser...")

        parser_main()
        print("Parser completed successfully")
        return True
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Parser failed: {e}")
        return False


def run_coco_converter():
    """Run the COCO converter script."""
    try:
        print("Starting COCO converter...")
        coco_main()
        print("COCO converter completed successfully")
        return True
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"COCO converter failed: {e}")
        return False


def main():
    """Run both processes in parallel."""
    print("Starting BDD100K data parsing pipeline...")

    # make sure output directories exist
    Config.parsed_data.mkdir(parents=True, exist_ok=True)
    Config.coco_data.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Create and start parallel processes
    parser_process = mp.Process(target=run_parser, name="Parser")
    coco_process = mp.Process(target=run_coco_converter, name="COCO-Converter")

    parser_process.start()
    coco_process.start()

    print("Waiting for both processes to complete...")
    parser_process.join()
    coco_process.join()

    # Check results
    parser_success = parser_process.exitcode == 0
    coco_success = coco_process.exitcode == 0

    duration = time.time() - start_time
    print(f"\nPipeline completed in {duration:.2f} seconds")
    print("=" * 50)

    if parser_success and coco_success:
        print("Both processes completed successfully!")
    elif parser_success:
        print("Parser completed, but COCO converter failed")
    elif coco_success:
        print("COCO converter completed, but Parser failed")
    else:
        print("Both processes failed!")

    print("=" * 50)
    print("Output directories:")
    print(f"  Parsed data: {Config.parsed_data}")
    print(f"  COCO data: {Config.coco_data}")

    # if failed, run manually
    if not parser_success or not coco_success:
        print("\nRestart manually:")

        if not parser_success:
            print(
                "Run parser: uv run python -m "
                "autonomous_vision.data_parser.parser_core"
            )
        if not coco_success:
            print(
                "Run COCO: uv run python -m "
                "autonomous_vision.data_parser.bdd_to_coco"
            )

        print(
            "Run both: uv run python -m "
            "autonomous_vision.data_parser.parsing_logic"
        )


if __name__ == "__main__":
    main()

# After parser and coco converter, run sanity check
# to make sure parsing and coco conversion went well
# uv run python -m scripts.sanity_check
# The numbers were verified from the notebook/EDA_RawData.ipynb
