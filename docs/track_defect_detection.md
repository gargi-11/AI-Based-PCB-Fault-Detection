# Copper Track Defect Detection Specification & Limitations

## 1. Executive Summary

The **Copper Track Defect Detection Module** provides automated computer vision inspection for conductive copper traces on Printed Circuit Boards.

Target physical defect categories:
1. **Broken Copper Tracks**: Complete physical trace fractures / open circuits causing signal/power path disconnection.
2. **Track Discontinuities & Gap Fractures**: Facing collinear trace endpoints with missing copper gaps.
3. **Missing Track Segments**: Entire trace segments present in the golden reference layout that are absent or un-etched in the test board.
4. **Abnormal Track Necking & Mousebites**: Localized reduction in trace width exceeding standard tolerances.

---

## 2. Algorithmic Pipeline Architecture

```
[Test PCB] + [Golden Reference PCB]
        │
        ├── 1. Color Normalization & Contrast Enhancement (LAB-CLAHE)
        │
        ├── 2. ORB/SIFT Geometric Registration to Reference Space
        │
        ├── 3. Multi-Space Conductive Trace Segmentation:
        │      - LAB Luminance Adaptive Thresholding
        │      - HSV Green/Soldermask Suppression
        │      - Otsu Binarization
        │
        ├── 4. Morphological Graph Cleanup (Opening & Bridge Preservation)
        │
        ├── 5. Medial Axis Topological Skeletonization (1-pixel Thinning)
        │
        ├── 6. Skeleton Graph Analysis (Branch Points & 8-Neighbor Endpoints)
        │
        ├── 7. Facing Collinear Endpoint Gap Estimation
        │
        ├── 8. Reference vs Test Trace Volume Differential & Subtraction
        │
        └── 9. Structured Output Schema & Multi-Panel Visualization
```

### 2.1 Trace Skeletonization & Discontinuity Graph
1. **Topological Medial Axis**: Iteratively thins segmented binary traces down to a 1-pixel wide continuous skeleton preserving connectivity.
2. **Endpoint Detection**: Convolves the binary skeleton with an 8-neighbor kernel. Pixels with neighbor count $= 1$ are flagged as trace terminals.
3. **Collinear Gap Matching**: For every pair of endpoints facing each other within the critical gap window ($3\text{px} \le d \le 40\text{px}$), the algorithm measures the linear distance and checks trace density across the gap to confirm a physical fracture.

### 2.2 Reference-Guided Differential Volume Analysis
When a golden template is supplied:
1. The detector computes the differential trace mask: $\text{Mask}_{\text{missing}} = \text{Trace}_{\text{ref}} \setminus \text{Trace}_{\text{test}}$.
2. Regions with volume loss exceeding $60\%$ are flagged as broken or missing tracks with severe electrical impact.

---

## 3. Output Schema Specification

Every detected track fault generates a structured record containing differential evidence:

```json
{
  "fault_id": "FLT-BT-001",
  "fault_type": "broken_copper_track",
  "location": {
    "bbox": [220, 115, 25, 12],
    "gap_distance_px": 20.0,
    "endpoints": [[220, 120], [240, 120]],
    "orientation": "horizontal"
  },
  "confidence": 0.9250,
  "severity": "critical_open",
  "evidence": [
    "Topological skeleton discontinuity detected with physical gap distance of 20.0px.",
    "Facing collinear conductive endpoints identified at (220, 120) and (240, 120).",
    "Continuous conductive copper trace present in reference PCB is absent in test board at [220, 115, 25, 12]."
  ],
  "reference_difference": {
    "ref_trace_volume_px": 120,
    "test_trace_volume_px": 0,
    "volume_loss_pct": 100.0,
    "gap_distance_px": 20.0
  },
  "possible_cause": "Mechanical surface scratch, copper etching over-etch, or trace thermal blowout.",
  "impact": "Complete electrical open circuit on affected signal or power trace.",
  "recommended_action": [
    "Clean fracture area with isopropyl alcohol.",
    "Bridge open trace with 30 AWG insulated kynar jumper wire per IPC-7721 Procedure 4.2.3.",
    "Seal with UV-curable solder mask."
  ]
}
```

---

## 4. Multi-Panel Diagnostic Visualization

Diagnostic visualization images are automatically generated and saved to:
`results/detected_tracks/track_defects_{analysis_id}.png`

- **Left Panel (Golden Reference PCB)**: Outlines continuous conductive signal tracks in green labeled `CONTINUOUS TRACE`.
- **Right Panel (Test PCB)**: Highlights detected track fractures in amber with red endpoint markers and bridge connection lines labeled `FLT-BT-XXX: BREAK (Confidence %)`.

---

## 5. Technical Limitations & Mitigation Strategies

| Limitation | Impact | Mitigation / System Handling |
|---|---|---|
| **Multi-Layer Internal Traces** | Inner copper layers in 4+ layer boards cannot be inspected visually. | The system inspects top and bottom outer layers (AVI standard scope); internal layer faults require electrical/X-ray testing. |
| **Sub-Pixel Micro-Cracks ($< 2\mu m$)** | Hairline fractures narrower than optical camera resolution may evade thresholding. | Preprocessing applies LAB-CLAHE gradient enhancement; future zoom ROI cameras can be integrated. |
| **Matte / Dark Soldermask Finishes** | Matte black or dark purple soldermasks reduce color contrast against copper. | Multi-space segmentation combines HSV color suppression with Otsu luminance adaptation. |
| **Silkscreen Legend Occlusion** | White silkscreen text printed directly over traces alters local pixel brightness. | Morphological closing bridges small text interruptions; golden reference subtraction eliminates static silkscreen differences. |
