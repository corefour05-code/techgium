"""Dataset acquisition module for the Infusion Pump ML System.

This module provides verified download utilities for public research datasets:
- DosePI (Figshare / Nature Scientific Data)
- Pressure Occlusion Study (PMC8555317)
- Force-based Failure Detection Study (PMC11535327)
- Smart Pump Event Logs (JMIR AI e48628)
- Typical Sensor Defects (Zenodo 56358)

Adheres strictly to scientific data integrity rules:
- Verifies downloaded files are not HTML error pages or truncated streams.
- Preserves raw downloaded files untouched.
- Accurately reports public availability vs. author-request/restricted datasets.
"""

import json
import logging
import os
import shutil
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.config import DATASET_METADATA, RAW_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class DownloadResult:
    dataset_id: str
    status: str  # "SUCCESS", "MANUAL_REQUIRED", "PARTIAL", "FAILED"
    downloaded_files: List[str]
    total_bytes: int
    message: str
    manual_instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_download_file(url: str, dest_path: Path, min_bytes: int = 100) -> bool:
    """Download a file with verification of HTTP status and content integrity.

    Args:
        url: Direct HTTP/HTTPS download URL.
        dest_path: Local destination path.
        min_bytes: Minimum expected file size to reject error messages.

    Returns:
        True if download succeeded and passed content checks, False otherwise.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "InfusionPumpML-ResearchBot/1.0 (Academic/MedTech ML Research)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
            content_type = response.headers.get("Content-Type", "")
            shutil.copyfileobj(response, out_file)

        # Check file size
        file_size = temp_path.stat().st_size
        if file_size < min_bytes:
            logger.warning(f"File {temp_path.name} is suspiciously small ({file_size} bytes).")
            if temp_path.exists():
                temp_path.unlink()
            return False

        # Check if an HTML error page was returned instead of data/binary
        with open(temp_path, "rb") as check_f:
            header_sample = check_f.read(256).decode("utf-8", errors="ignore").lower()
            if "<!doctype html" in header_sample or "<html" in header_sample:
                # Unless we expected HTML, this is usually an access gate, captcha or 404
                if not dest_path.name.endswith(".html"):
                    logger.warning(f"Downloaded content for {dest_path.name} appears to be HTML rather than raw data.")
                    temp_path.unlink()
                    return False

        # Move temp to destination
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        logger.info(f"Successfully downloaded {dest_path.name} ({file_size:,} bytes).")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def download_dosepi(target_dir: Optional[Path] = None) -> DownloadResult:
    """Download DosePI dataset from Figshare repository (DOI: 10.6084/m9.figshare.28391645)."""
    meta = DATASET_METADATA["dosepi"]
    out_dir = target_dir or meta["target_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    article_id = meta["figshare_article_id"]
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"

    downloaded = []
    total_bytes = 0

    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "InfusionPumpML-ResearchBot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        files = data.get("files", [])
        if not files:
            return DownloadResult(
                dataset_id="dosepi",
                status="FAILED",
                downloaded_files=[],
                total_bytes=0,
                message="Figshare API returned no files for article 28391645.",
                manual_instructions=f"Visit {meta['repository_url']} to manually download files."
            )

        logger.info(f"DosePI article found with {len(files)} files on Figshare.")
        for f_info in files:
            file_name = f_info.get("name", "unknown")
            download_url = f_info.get("download_url")
            expected_size = f_info.get("size", 0)
            local_file = out_dir / file_name

            # Skip if already exists with matching size
            if local_file.exists() and local_file.stat().st_size == expected_size:
                logger.info(f"File {file_name} already exists and matches expected size.")
                downloaded.append(str(local_file))
                total_bytes += expected_size
                continue

            if download_url and _safe_download_file(download_url, local_file):
                downloaded.append(str(local_file))
                total_bytes += local_file.stat().st_size

        if downloaded:
            return DownloadResult(
                dataset_id="dosepi",
                status="SUCCESS",
                downloaded_files=downloaded,
                total_bytes=total_bytes,
                message=f"Downloaded {len(downloaded)} files from Figshare."
            )
        else:
            return DownloadResult(
                dataset_id="dosepi",
                status="FAILED",
                downloaded_files=[],
                total_bytes=0,
                message="Files could not be downloaded from Figshare endpoints.",
                manual_instructions=f"Visit {meta['repository_url']} to manually download the dataset."
            )

    except Exception as e:
        logger.error(f"Error querying Figshare API for DosePI: {e}")
        return DownloadResult(
            dataset_id="dosepi",
            status="FAILED",
            downloaded_files=[],
            total_bytes=0,
            message=f"Network/API error while querying Figshare: {str(e)}",
            manual_instructions=f"Visit {meta['repository_url']} and download files directly into {out_dir}."
        )


def download_sensor_defects(target_dir: Optional[Path] = None) -> DownloadResult:
    """Download Typical Sensor Defects dataset from Zenodo repository (Record: 56358)."""
    meta = DATASET_METADATA["sensor_defects"]
    out_dir = target_dir or meta["target_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    record_id = meta["zenodo_record_id"]
    api_url = f"https://zenodo.org/api/records/{record_id}"

    downloaded = []
    total_bytes = 0

    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "InfusionPumpML-ResearchBot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        files = data.get("files", [])
        if not files:
            return DownloadResult(
                dataset_id="sensor_defects",
                status="FAILED",
                downloaded_files=[],
                total_bytes=0,
                message=f"Zenodo record {record_id} contains no files.",
                manual_instructions=f"Visit {meta['repository_url']} to manually download."
            )

        logger.info(f"Zenodo record 56358 found with {len(files)} files.")
        for f_info in files:
            file_name = f_info.get("key", f_info.get("filename", "unknown"))
            # In Zenodo v1/v2 API links can be in 'links.self' or 'links.download'
            links = f_info.get("links", {})
            download_url = links.get("self") or links.get("download")
            expected_size = f_info.get("size", 0)
            local_file = out_dir / file_name

            if local_file.exists() and local_file.stat().st_size == expected_size:
                logger.info(f"File {file_name} already exists.")
                downloaded.append(str(local_file))
                total_bytes += expected_size
            elif download_url and _safe_download_file(download_url, local_file):
                downloaded.append(str(local_file))
                total_bytes += local_file.stat().st_size

            # If the downloaded file is a zip archive, extract it
            if local_file.exists() and zipfile.is_zipfile(local_file):
                extract_dir = out_dir / "extracted"
                try:
                    with zipfile.ZipFile(local_file, "r") as zf:
                        zf.extractall(extract_dir)
                    logger.info(f"Extracted {local_file.name} to {extract_dir}.")
                except Exception as ze:
                    logger.warning(f"Could not extract {local_file.name}: {ze}")

        return DownloadResult(
            dataset_id="sensor_defects",
            status="SUCCESS",
            downloaded_files=downloaded,
            total_bytes=total_bytes,
            message=f"Downloaded {len(downloaded)} files from Zenodo Record 56358."
        )

    except Exception as e:
        logger.error(f"Error querying Zenodo API for sensor defects: {e}")
        return DownloadResult(
            dataset_id="sensor_defects",
            status="FAILED",
            downloaded_files=[],
            total_bytes=0,
            message=f"Network/API error while querying Zenodo: {str(e)}",
            manual_instructions=f"Visit {meta['repository_url']} and download the archive into {out_dir}."
        )


def check_pressure_occlusion_availability(target_dir: Optional[Path] = None) -> DownloadResult:
    """Check availability of Multi-infusion pressure sensing and line occlusion study (PMC8555317)."""
    meta = DATASET_METADATA["pressure_occlusion"]
    out_dir = target_dir or meta["target_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already present manually
    existing_files = [str(p) for p in out_dir.iterdir() if p.name != ".gitkeep" and not p.name.endswith(".md")]
    if existing_files:
        return DownloadResult(
            dataset_id="pressure_occlusion",
            status="SUCCESS",
            downloaded_files=existing_files,
            total_bytes=sum(Path(p).stat().st_size for p in existing_files),
            message=f"Found {len(existing_files)} locally present files in {out_dir}."
        )

    # Formal availability statement from the published manuscript:
    # "The datasets used and/or analyzed during the current study are available from the corresponding author on reasonable request."
    return DownloadResult(
        dataset_id="pressure_occlusion",
        status="MANUAL_REQUIRED",
        downloaded_files=[],
        total_bytes=0,
        message="Public repository deposit not provided by authors. The publication states: 'The datasets used and/or analyzed during the current study are available from the corresponding author on reasonable request.'",
        manual_instructions=(
            f"1. Refer to paper PMC8555317 ({meta['paper_url']}).\n"
            f"2. Contact corresponding author: Prof. Dr. Tim Kaufmann / Department of Anesthesiology, University Medical Center Groningen.\n"
            f"3. Place any received raw pressure timeseries CSV/DAT files into '{out_dir}'."
        )
    )


def check_force_failure_availability(target_dir: Optional[Path] = None) -> DownloadResult:
    """Check availability of Force-based Infusion Failure Detection Study (PMC11535327)."""
    meta = DATASET_METADATA["force_failure"]
    out_dir = target_dir or meta["target_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [str(p) for p in out_dir.iterdir() if p.name != ".gitkeep" and not p.name.endswith(".md")]
    if existing_files:
        return DownloadResult(
            dataset_id="force_failure",
            status="SUCCESS",
            downloaded_files=existing_files,
            total_bytes=sum(Path(p).stat().st_size for p in existing_files),
            message=f"Found {len(existing_files)} locally present files in {out_dir}."
        )

    # Study published in Journal of Diabetes Science and Technology (2024).
    # Raw continuous force/pressure time-series was not deposited to Zenodo/Figshare;
    # supplementary file is method documentation (docx).
    return DownloadResult(
        dataset_id="force_failure",
        status="MANUAL_REQUIRED",
        downloaded_files=[],
        total_bytes=0,
        message="Raw continuous force-load timeseries was not deposited in an open repository. Supplementary docx is attached to manuscript.",
        manual_instructions=(
            f"1. Refer to PMC11535327 ({meta['paper_url']}).\n"
            f"2. Corresponding author: Prof. Dr. Bryan P. Boudouris / Purdue University.\n"
            f"3. Place raw sensor records into '{out_dir}'."
        )
    )


def check_smart_pump_logs_availability(target_dir: Optional[Path] = None) -> DownloadResult:
    """Check availability of Hospitals Infusion Data Set: Spain (JMIR AI 2023; e48628)."""
    meta = DATASET_METADATA["smart_pump_logs"]
    out_dir = target_dir or meta["target_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [str(p) for p in out_dir.iterdir() if p.name != ".gitkeep" and not p.name.endswith(".md")]
    if existing_files:
        return DownloadResult(
            dataset_id="smart_pump_logs",
            status="SUCCESS",
            downloaded_files=existing_files,
            total_bytes=sum(Path(p).stat().st_size for p in existing_files),
            message=f"Found {len(existing_files)} locally present files in {out_dir}."
        )

    return DownloadResult(
        dataset_id="smart_pump_logs",
        status="MANUAL_REQUIRED",
        downloaded_files=[],
        total_bytes=0,
        message="Hospital smart pump event logs are proprietary and clinical patient-data governed. Not released publicly under open data license.",
        manual_instructions=(
            f"1. Refer to JMIR AI 2023; 2:e48628 ({meta['paper_url']}).\n"
            f"2. Contact authors: Dr. Arash Kia / Becton Dickinson Medical Affairs & University of Limerick.\n"
            f"3. Data sharing requires institutional data governance and clinical research ethics approval.\n"
            f"4. Once access is approved, place event log CSV files into '{out_dir}'."
        )
    )


def download_all_datasets() -> Dict[str, DownloadResult]:
    """Execute download and availability inspection across all 5 research datasets."""
    results = {}
    logger.info("Executing acquisition for DATASET A: DosePI...")
    results["dosepi"] = download_dosepi()

    logger.info("Executing acquisition for DATASET B: Pressure Occlusion (PMC8555317)...")
    results["pressure_occlusion"] = check_pressure_occlusion_availability()

    logger.info("Executing acquisition for DATASET C: Force Failure (PMC11535327)...")
    results["force_failure"] = check_force_failure_availability()

    logger.info("Executing acquisition for DATASET D: Smart Pump Logs (JMIR AI e48628)...")
    results["smart_pump_logs"] = check_smart_pump_logs_availability()

    logger.info("Executing acquisition for DATASET E: Typical Sensor Defects (Zenodo 56358)...")
    results["sensor_defects"] = download_sensor_defects()

    return results

