"""CLI Script to download and verify the research datasets."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import (
    download_all_datasets,
    download_dosepi,
    download_sensor_defects,
    check_pressure_occlusion_availability,
    check_force_failure_availability,
    check_smart_pump_logs_availability,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("download_datasets")


def main():
    parser = argparse.ArgumentParser(description="Download and verify research datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "dosepi", "pressure_occlusion", "force_failure", "smart_pump_logs", "sensor_defects"],
        default="all",
        help="Specify which dataset to download/verify (default: all)",
    )
    args = parser.parse_args()

    logger.info(f"Starting acquisition check for target: {args.dataset}")

    results = {}
    if args.dataset in ["all", "dosepi"]:
        logger.info("--- DATASET A: DosePI (Figshare) ---")
        results["dosepi"] = download_dosepi()

    if args.dataset in ["all", "pressure_occlusion"]:
        logger.info("--- DATASET B: Pressure Occlusion (PMC8555317) ---")
        results["pressure_occlusion"] = check_pressure_occlusion_availability()

    if args.dataset in ["all", "force_failure"]:
        logger.info("--- DATASET C: Force Failure (PMC11535327) ---")
        results["force_failure"] = check_force_failure_availability()

    if args.dataset in ["all", "smart_pump_logs"]:
        logger.info("--- DATASET D: Smart Pump Logs (JMIR AI e48628) ---")
        results["smart_pump_logs"] = check_smart_pump_logs_availability()

    if args.dataset in ["all", "sensor_defects"]:
        logger.info("--- DATASET E: Typical Sensor Defects (Zenodo 56358) ---")
        results["sensor_defects"] = download_sensor_defects()

    print("\n" + "=" * 80)
    print("DATASET ACQUISITION SUMMARY")
    print("=" * 80)
    for ds_id, res in results.items():
        print(f"[{res.status.upper()}] Dataset: {ds_id}")
        print(f"  Message: {res.message}")
        print(f"  Files: {len(res.downloaded_files)} files ({res.total_bytes / (1024*1024):.2f} MB)")
        if res.manual_instructions:
            print(f"  Manual Instructions:\n    {res.manual_instructions.replace(chr(10), chr(10) + '    ')}")
        print("-" * 80)


if __name__ == "__main__":
    main()

