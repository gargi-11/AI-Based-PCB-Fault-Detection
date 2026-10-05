"""
Processing Package for AI-Based PCB Fault Detection.

Provides computer vision modules for:
- Image loading and validation (image_loader.py)
- LAB/CLAHE contrast and bilateral denoising (enhancement.py, preprocess.py)
- Golden reference ORB/SIFT homography alignment (alignment.py)
- Differential comparison and noise suppression (comparison.py)
- Missing component detection (missing_component_detector.py, component_detection.py)
- Broken copper track skeletonization & gap detection (track_defect_detector.py, track_detection.py)
- Fault deduplication and verdict aggregation (fault_aggregation.py)
- Master end-to-end inspection pipeline (pipeline.py)
"""

from src.processing.image_loader import (
    validate_and_load_image,
    save_image,
)
from src.processing.enhancement import (
    resize_preserve_aspect,
    enhance_pcb_illumination,
    reduce_pcb_noise,
    to_grayscale,
    enhance_pcb_pipeline,
)
from src.processing.preprocess import preprocess_pcb_image
from src.processing.alignment import align_images_orb
from src.processing.comparison import compute_difference_map
from src.processing.missing_component_detector import (
    BaseMissingComponentDetector,
    CVMissingComponentDetector,
    DeepLearningMissingComponentDetector,
    get_missing_component_detector,
    compute_ssim_map,
)
from src.processing.component_detection import detect_component_defects
from src.processing.track_defect_detector import (
    CopperTrackDefectDetector,
    segment_conductive_traces,
    skeletonize_medial_axis,
    extract_skeleton_endpoints,
)
from src.processing.track_detection import (
    detect_track_defects,
    segment_copper_traces,
    skeletonize_traces,
    find_skeleton_endpoints,
)
from src.processing.fault_aggregation import (
    aggregate_faults,
    deduplicate_faults,
    compute_iou,
)
from src.processing.pipeline import (
    PCBInspectionPipeline,
    inspect_pcb,
)

__all__ = [
    "validate_and_load_image",
    "save_image",
    "resize_preserve_aspect",
    "enhance_pcb_illumination",
    "reduce_pcb_noise",
    "to_grayscale",
    "enhance_pcb_pipeline",
    "preprocess_pcb_image",
    "align_images_orb",
    "compute_difference_map",
    "BaseMissingComponentDetector",
    "CVMissingComponentDetector",
    "DeepLearningMissingComponentDetector",
    "get_missing_component_detector",
    "compute_ssim_map",
    "detect_component_defects",
    "CopperTrackDefectDetector",
    "segment_conductive_traces",
    "skeletonize_medial_axis",
    "extract_skeleton_endpoints",
    "detect_track_defects",
    "segment_copper_traces",
    "skeletonize_traces",
    "find_skeleton_endpoints",
    "aggregate_faults",
    "deduplicate_faults",
    "compute_iou",
    "PCBInspectionPipeline",
    "inspect_pcb",
]
