# Missing Component Detection Specification & ML Roadmap

## 1. Executive Summary

The **Missing Component Detection Module** is an authoritative visual inspection subsystem designed to identify absent, dislocated, or unpopulated electronic components (ICs, chip resistors, capacitors, inductors, diodes, connectors) on Printed Circuit Boards.

In accordance with the project's core architectural principle, **Computer Vision & Reference Differential Comparison serves as the authoritative primary detector**, while Gemini and RAG layers perform high-level diagnostic reasoning.

---

## 2. Computer Vision Algorithmic Pipeline

```
[Test PCB] + [Golden Reference PCB]
              │
              ├── 1. ORB/SIFT Geometric Registration (Homography Warp)
              │
              ├── 2. Structural Similarity (Local SSIM) & Absolute Intensity Differential
              │
              ├── 3. Morphological Noise Filtering (Eliminates sub-pixel jitter & lighting glare)
              │
              ├── 4. Component Candidate Footprint Extraction (Contours & Aspect Ratio)
              │
              ├── 5. Physical Presence Verification (Texture Variance, Canny Edge Density, Solder Pad Reflection)
              │
              └── 6. Missing Component Schema Output & Multi-Panel Visualization
```

### 2.1 Presence vs Absence Verification
For each candidate component region, the detector evaluates:
1. **Reference Board Presence ($P_{\text{ref}}$)**:
   - High Canny edge density ($> 0.04$).
   - High pixel intensity standard deviation / texture ($\sigma_{\text{ref}} > 18.0$).
2. **Test Board Absence ($A_{\text{test}}$)**:
   - Sharp edge density reduction ($\Delta \text{edge} > 0.03$).
   - Texture drop ($\Delta \sigma > 8.0$).
   - Local SSIM degradation ($\text{SSIM}_{\text{local}} < 0.65$).
   - High metallic reflectivity on bare terminal solder pads with vacant dark center.

### 2.2 False-Positive Suppression
- **Luminance Normalization**: LAB-CLAHE equalizes non-uniform illumination and reflections.
- **Morphological Opening/Closing**: Strips single-pixel noise and lighting boundary artifacts.
- **Aspect Ratio Gating**: Excludes narrow, elongated contours (aspect ratio $> 5.5$) that correspond to traces rather than discrete SMD/TH packages.

---

## 3. Output Schema Specification

Every detected missing component generates a structured JSON record:

```json
{
  "component_id": "MC-001",
  "fault_id": "FLT-MC-001",
  "fault_type": "missing_component",
  "location": {
    "bbox": [120, 220, 40, 25],
    "area_px": 1000,
    "aspect_ratio": 1.60
  },
  "confidence": 0.9450,
  "severity": "critical_open",
  "evidence": [
    "Reference PCB exhibits populated component structure (edge density: 0.125, std: 42.1).",
    "Test PCB shows absent component body (edge density: 0.012, local SSIM: 0.412, intensity delta: 64.5).",
    "High solder pad reflectivity detected on unpopulated terminal pads."
  ],
  "reference_present": true,
  "test_present": false,
  "possible_cause": "Pick-and-place feeder misfeed, missing tape pocket component, or desoldering detachment.",
  "impact": "Open circuit on affected branch leading to total or partial board subsystem failure.",
  "recommended_action": [
    "Inspect solder pads for oxidation or residual solder bridging.",
    "Place correct specification component and reflow/solder per IPC standards."
  ]
}
```

---

## 4. Multi-Panel Diagnostic Visualization

Diagnostic comparison images are automatically rendered and saved to:
`results/detected_components/missing_components_{analysis_id}.png`

- **Left Panel (Golden Reference)**: Displays the expected component in green outline with label `EXPECTED: MC-XXX`.
- **Right Panel (Test PCB)**: Displays the missing component area in red highlight with label `MISSING MC-XXX (Confidence %)`.

---

## 5. Machine Learning & Deep Learning Roadmap

While the classical CV & differential comparison baseline provides immediate, deterministic inspection without requiring millions of labelled images, a future Deep Learning model can be plugged directly into `BaseMissingComponentDetector`.

### 5.1 Dataset Requirements for Future ML Training
To train an object detection model (e.g. YOLOv8 / Faster R-CNN / RT-DETR), the following dataset must be collected:
1. **Annotated Classes**:
   - `populated_resistor_0805`, `populated_resistor_0603`, `populated_resistor_0402`
   - `populated_capacitor_smd`, `populated_capacitor_electrolytic`
   - `populated_ic_soic`, `populated_ic_qfp`, `populated_ic_dip`, `populated_ic_qfn`
   - `vacant_solder_pad_pair`
   - `missing_component_anomaly`
2. **Annotation Format**: YOLO TXT / COCO JSON bounding boxes (`class_id x_center y_center width height`).
3. **Volume**: Recommended minimum 1,500 real PCB inspection photos with diverse lighting conditions, angles, and soldermask colors (green, blue, black, red).

### 5.2 Drop-in Deep Learning Integration
When trained weights (`.pt` or `.onnx`) are available, instantiate via:
```python
from src.processing.missing_component_detector import DeepLearningMissingComponentDetector

detector = DeepLearningMissingComponentDetector(model_weights_path="models/missing_component_detector/yolov8_pcb_best.pt")
```
The architecture seamlessly routes inference through the neural network without modifying downstream RAG or LLM reasoning modules.
