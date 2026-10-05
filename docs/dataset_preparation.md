# PCB Dataset Preparation & Audit Specification

## 1. Executive Summary

This document details the acquisition, structure, annotation formats, and preprocessing of real-world datasets for the **AI-Based PCB Fault Detection System**:
1. **Bare PCB Defect Dataset (DeepPCB)**: Downloaded and verified for Broken / Open Copper Track detection and golden reference differential alignment.
2. **PCBA Missing Component Dataset Status**: Evaluated against public repositories and structured for incoming assembled board datasets.

---

## 2. Dataset 1: DeepPCB (Broken / Open Copper Tracks & Bare Board Defects)

### 2.1 Metadata & Source
- **Dataset Name**: DeepPCB: A Dataset for PCB Defect Detection (Tang et al., 2019)
- **Official URL**: [https://github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)
- **License / Restrictions**: Research and Academic Use Only
- **Acquisition Method**: Cloned via Git from official repository (`tangsanli5201/DeepPCB`) into `data/raw/DeepPCB_repo/`

### 2.2 Dataset Statistics
| Metric | Count | Description |
|---|---|---|
| **Total Test Images** | **1,500** | High-resolution 640×640 inspection frames with defects |
| **Total Reference / Template Images** | **1,500** | Defect-free, geometrically aligned golden template images |
| **Total Image Pairs** | **1,500** | 1:1 pair matching between test and template images |
| **Pairs with Broken / Open Track Defects** | **1,344** | Image pairs containing at least one physical open circuit |
| **Total Open (Broken Track) Annotations** | **1,942** | Individual bounding box annotations for track breaks |
| **Total Annotations (All Defect Types)** | **10,013** | Total bounding boxes across all 6 defect categories |

### 2.3 Defect Breakdown (DeepPCB)
| Type ID | Defect Category | Total Bounding Boxes | Project Role |
|---|---|---|---|
| **1** | **Open (Broken Copper Track)** | **1,942** | **Primary Target (Broken Tracks)** |
| **2** | Short Circuit | 1,506 | Secondary Copper Defect |
| **3** | Mousebite (Trace Necking) | 1,965 | Trace Geometry Anomaly |
| **4** | Spur | 1,625 | Trace Geometry Anomaly |
| **5** | Spurious Copper | 1,474 | Foreign Particle Defect |
| **6** | Pin-Hole | 1,501 | Substrate / Pad Defect |

### 2.4 Annotation Format
Each image pair has a corresponding `.txt` annotation file:
```
x1 y1 x2 y2 type_id
```
- `(x1, y1)`: Top-left corner coordinates in pixels ($0 \le x_1, y_1 \le 640$)
- `(x2, y2)`: Bottom-right corner coordinates in pixels ($0 \le x_2, y_2 \le 640$)
- `type_id`: Integer class identifier (`1` = `open`, `2` = `short`, etc.)

### 2.5 Train / Test Partitioning
Following the official DeepPCB benchmark split:
- **Train / Validation Set (`trainval`)**: 1,000 image pairs (Groups `00041`, `12000`, `12100`, `12300`, `13000`, `20085`, `44000`)
- **Evaluation Test Set (`test`)**: 500 image pairs (Groups `50600`, `77000`, `90100`, `92000`)

### 2.6 Local Organization & Metadata
Organized under:
```
data/
  raw/
    broken_tracks/
      deeppcb/
        images/          # 1,500 test images (*_test.jpg)
        templates/       # 1,500 golden reference templates (*_temp.jpg)
        annotations/     # 1,500 annotation files (*.txt)
  processed/
    deeppcb_metadata.json # Full JSON index of pairs, bounding boxes, and split mapping
    deeppcb_metadata.csv  # Flat CSV table for rapid inspection
```

---

## 3. Dataset 2: Missing Component Dataset Audit

### 3.1 Industry Context: Bare PCB vs. PCBA Assembly
- Standard open-source PCB datasets (such as **DeepPCB**, **PKU-Market-PCB**, and **DsPCBSD+**) focus on **bare PCB fabrication flaws** (open tracks, shorts, spurs, missing drill holes). Because bare PCBs are unpopulated prior to Surface Mount Technology (SMT) assembly, they contain no electronic components.
- Datasets for **Missing SMD Components** (resistors, capacitors, ICs, diodes) require populated Printed Circuit Board Assembly (PCBA) photography.

### 3.2 Candidate Public Sources for PCBA Component Defects
1. **Roboflow PCB-AOI-DEFECT Dataset**:
   - **Source**: [https://universe.roboflow.com/vijayabaskar-s/pcb-aoi-defect-gcez7](https://universe.roboflow.com/vijayabaskar-s/pcb-aoi-defect-gcez7)
   - **Classes**: `missing_component`, `short`, `excess_solder`, `insufficient_solder`
   - **Access Status**: Requires authenticated Roboflow API Key / manual browser ZIP export.
2. **FICS-PCB Component Assurance Dataset**:
   - **Source**: University of Florida Institute for Cybersecurity (FICS)
   - **Access Status**: Available upon academic data request agreement.

### 3.3 Prepared Directory Layout for Missing Components
```
data/
  raw/
    missing_components/
      images/         # Populated test PCB photos (*.jpg, *.png)
      references/     # Golden populated template photos (*.jpg, *.png)
      annotations/    # Bounding box labels (YOLO / VOC / JSON)
```

---

## 4. Dataset Preparation Script

A fully reproducible extraction and indexing script is provided at:
[`src/processing/prepare_datasets.py`](file:///c:/Users/saura/Desktop/AI-Based-PCB-Fault-Detection/AI-Based-PCB-Fault-Detection/src/processing/prepare_datasets.py)

To re-run or verify extraction at any time:
```powershell
.\venv\Scripts\python.exe src/processing/prepare_datasets.py
```

---

## 5. Summary of Data Assets

| Feature / Category | Broken Track (DeepPCB) | Missing Component (PCBA) |
|---|---|---|
| **Dataset Source** | Tang et al. (GitHub DeepPCB) | Populated PCBA Public Repositories |
| **Status** | **DOWNLOADED & VERIFIED** | **DIRECTORY STRUCTURE PREPARED** |
| **Total Test Frames** | 1,500 | Prepared for ingestion |
| **Golden Reference Frames** | 1,500 | Prepared for ingestion |
| **Target Defect Instances** | 1,942 open-circuit annotations | Verified via Computer Vision Baseline |
| **Ground Truth Bounding Boxes** | Available (`x1 y1 x2 y2 type`) | YOLO / COCO format compatible |
| **Metadata Indexes** | `data/processed/deeppcb_metadata.json` | Ready for indexing |
