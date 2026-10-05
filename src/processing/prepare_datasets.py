"""
PCB Dataset Preparation and Extraction Utility.

Extracts, filters, and standardizes real-world PCB defect datasets:
1. DeepPCB (Tang et al., 2019) for Broken Tracks (Open Circuits) and Template Matching.
2. Formats dataset metadata (CSV & JSON) mapping test images, golden templates, and annotations.
3. Structures data/raw/, data/processed/, and data/test/ partitions.
"""

import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# DeepPCB defect type mapping
DEEPPCB_DEFECT_MAP = {
    1: "open",           # Broken / Open Copper Track
    2: "short",          # Copper Trace Short Circuit
    3: "mousebite",      # Edge Notch / Necking Defect
    4: "spur",           # Unwanted Copper Spur
    5: "spurious_copper",# Isolated Copper Particle
    6: "pin_hole",       # Hole / Void in Copper
}


def extract_deeppcb_subset(
    repo_root: Optional[str] = None,
    output_raw_dir: Optional[str] = None,
    output_processed_dir: Optional[str] = None,
    filter_open_only: bool = False,
) -> Dict[str, Any]:
    """
    Extract and organize DeepPCB dataset pairs and annotations.

    Args:
        repo_root: Path to cloned DeepPCB repository (default: data/raw/DeepPCB_repo).
        output_raw_dir: Destination for organized raw broken tracks data.
        output_processed_dir: Destination for processed metadata CSV/JSON.
        filter_open_only: If True, copies only pairs containing at least one 'open' defect.

    Returns:
        Dict of extraction summary statistics.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    src_repo = Path(repo_root or (project_root / "data" / "raw" / "DeepPCB_repo"))
    pcb_data_dir = src_repo / "PCBData"

    if not pcb_data_dir.is_dir():
        raise FileNotFoundError(
            f"DeepPCB repository not found at {src_repo}. Please ensure it is cloned."
        )

    # Output directory layout
    raw_dest = Path(output_raw_dir or (project_root / "data" / "raw" / "broken_tracks" / "deeppcb"))
    images_dir = raw_dest / "images"
    templates_dir = raw_dest / "templates"
    annotations_dir = raw_dest / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    proc_dest = Path(output_processed_dir or (project_root / "data" / "processed"))
    proc_dest.mkdir(parents=True, exist_ok=True)

    # Read official trainval and test splits if available
    trainval_file = pcb_data_dir / "trainval.txt"
    test_file = pcb_data_dir / "test.txt"

    trainval_set = set()
    if trainval_file.is_file():
        with open(trainval_file, "r") as f:
            trainval_set = {line.strip().split()[0] for line in f if line.strip()}

    test_set = set()
    if test_file.is_file():
        with open(test_file, "r") as f:
            test_set = {line.strip().split()[0] for line in f if line.strip()}

    # Scan all group folders
    groups = [g for g in pcb_data_dir.iterdir() if g.is_dir() and g.name.startswith("group")]
    groups.sort(key=lambda x: x.name)

    metadata_rows: List[Dict[str, Any]] = []

    total_scanned_pairs = 0
    total_open_defects = 0
    pairs_with_open = 0
    defect_type_counts = {k: 0 for k in DEEPPCB_DEFECT_MAP.keys()}

    for group in groups:
        group_id = group.name.replace("group", "")
        img_folder = group / group_id
        not_folder = group / f"{group_id}_not"

        if not img_folder.is_dir() or not not_folder.is_dir():
            continue

        for ann_file in sorted(not_folder.glob("*.txt")):
            base_id = ann_file.stem
            test_img_file = img_folder / f"{base_id}_test.jpg"
            temp_img_file = img_folder / f"{base_id}_temp.jpg"

            if not test_img_file.is_file() or not temp_img_file.is_file():
                continue

            total_scanned_pairs += 1

            # Parse annotations
            boxes_all = []
            boxes_open = []
            defect_types_in_img = set()

            with open(ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        x1, y1, x2, y2, dtype_val = map(int, parts)
                        boxes_all.append({
                            "bbox": [x1, y1, x2, y2],
                            "type_id": dtype_val,
                            "type_name": DEEPPCB_DEFECT_MAP.get(dtype_val, "unknown"),
                        })
                        defect_type_counts[dtype_val] = defect_type_counts.get(dtype_val, 0) + 1
                        defect_types_in_img.add(DEEPPCB_DEFECT_MAP.get(dtype_val, "unknown"))

                        if dtype_val == 1:
                            boxes_open.append([x1, y1, x2, y2])
                            total_open_defects += 1

            has_open = len(boxes_open) > 0
            if has_open:
                pairs_with_open += 1

            if filter_open_only and not has_open:
                continue

            # Determine split (default to trainval if not found in test)
            rel_key = f"{group.name}/{group_id}/{base_id}_test.jpg"
            if base_id in test_set or any(base_id in k for k in test_set):
                split_name = "test"
            elif base_id in trainval_set or any(base_id in k for k in trainval_set):
                split_name = "trainval"
            else:
                # 1000 trainval / 500 test canonical split
                split_name = "test" if len(metadata_rows) % 3 == 0 else "trainval"

            # Copy files to organized raw structure
            dest_test = images_dir / f"{base_id}_test.jpg"
            dest_temp = templates_dir / f"{base_id}_temp.jpg"
            dest_ann = annotations_dir / f"{base_id}.txt"

            if not dest_test.is_file():
                shutil.copy2(test_img_file, dest_test)
            if not dest_temp.is_file():
                shutil.copy2(temp_img_file, dest_temp)
            if not dest_ann.is_file():
                shutil.copy2(ann_file, dest_ann)

            record = {
                "image_id": base_id,
                "group_id": group_id,
                "split": split_name,
                "test_image_filename": f"{base_id}_test.jpg",
                "template_image_filename": f"{base_id}_temp.jpg",
                "annotation_filename": f"{base_id}.txt",
                "relative_test_image_path": str(dest_test.relative_to(project_root)).replace("\\", "/"),
                "relative_template_path": str(dest_temp.relative_to(project_root)).replace("\\", "/"),
                "relative_annotation_path": str(dest_ann.relative_to(project_root)).replace("\\", "/"),
                "total_defects": len(boxes_all),
                "open_defects_count": len(boxes_open),
                "has_open_defect": has_open,
                "defect_types": sorted(list(defect_types_in_img)),
                "open_bounding_boxes": boxes_open,
                "all_annotations": boxes_all,
                "source_dataset": "DeepPCB",
            }
            metadata_rows.append(record)

    # Write processed metadata JSON
    json_path = proc_dest / "deeppcb_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_rows, f, indent=2)

    # Write processed metadata CSV (flatted for easy table inspection)
    csv_path = proc_dest / "deeppcb_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "group_id",
            "split",
            "test_image_filename",
            "template_image_filename",
            "annotation_filename",
            "total_defects",
            "open_defects_count",
            "has_open_defect",
            "defect_types",
            "source_dataset",
        ])
        for r in metadata_rows:
            writer.writerow([
                r["image_id"],
                r["group_id"],
                r["split"],
                r["test_image_filename"],
                r["template_image_filename"],
                r["annotation_filename"],
                r["total_defects"],
                r["open_defects_count"],
                r["has_open_defect"],
                ";".join(r["defect_types"]),
                r["source_dataset"],
            ])

    # Summary report
    summary = {
        "dataset_name": "DeepPCB",
        "official_url": "https://github.com/tangsanli5201/DeepPCB",
        "license": "Research Use Only",
        "total_image_pairs_scanned": total_scanned_pairs,
        "total_image_pairs_extracted": len(metadata_rows),
        "pairs_with_open_defects": pairs_with_open,
        "total_open_defect_annotations": total_open_defects,
        "defect_type_breakdown": {
            DEEPPCB_DEFECT_MAP[k]: v for k, v in defect_type_counts.items()
        },
        "trainval_count": sum(1 for r in metadata_rows if r["split"] == "trainval"),
        "test_count": sum(1 for r in metadata_rows if r["split"] == "test"),
        "raw_storage_dir": str(raw_dest),
        "metadata_json_path": str(json_path),
        "metadata_csv_path": str(csv_path),
    }

    return summary


def init_dataset_directories() -> Dict[str, str]:
    """Initialize standard project dataset directories with .gitkeep files."""
    project_root = Path(__file__).resolve().parent.parent.parent

    dirs = {
        "raw_broken_tracks": project_root / "data" / "raw" / "broken_tracks",
        "raw_missing_components_images": project_root / "data" / "raw" / "missing_components" / "images",
        "raw_missing_components_refs": project_root / "data" / "raw" / "missing_components" / "references",
        "raw_missing_components_anns": project_root / "data" / "raw" / "missing_components" / "annotations",
        "processed": project_root / "data" / "processed",
        "test": project_root / "data" / "test",
    }

    for name, p in dirs.items():
        p.mkdir(parents=True, exist_ok=True)
        gitkeep = p / ".gitkeep"
        if not gitkeep.is_file():
            gitkeep.touch()

    return {k: str(v) for k, v in dirs.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Initializing Dataset Directories ===")
    dirs = init_dataset_directories()
    for k, v in dirs.items():
        print(f"  {k}: {v}")

    print("\n=== Extracting DeepPCB Broken Track Dataset ===")
    summary = extract_deeppcb_subset()
    print(json.dumps(summary, indent=2))
