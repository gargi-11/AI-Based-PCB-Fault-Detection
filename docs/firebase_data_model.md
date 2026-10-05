# Firebase Backend & Data Model Specification

## 1. Executive Summary

The **AI-Based PCB Fault Detection System** utilizes Firebase as its persistence and security backbone:
- **Firebase Authentication**: User registration, login, session token validation, and role-based access to the dashboard.
- **Firebase Cloud Storage**: Structured, partitioned storage for high-resolution input PCB photographs, golden reference templates, and annotated defect visualizations.
- **Cloud Firestore**: Real-time NoSQL document database storing inspection records, detected physical defects, AI diagnostic reports, and user metrics.

---

## 2. Firebase Cloud Storage Hierarchy

All uploaded imagery is partitioned logically by `user_id` and unique `analysis_id`:

```
gs://<PROJECT_ID>.appspot.com/
│
└── pcb_uploads/
    └── {user_id}/
        └── {analysis_id}/
            ├── input/
            │   └── input_board.png         # Original raw inspection photograph
            ├── reference/
            │   └── golden_reference.png    # Reference template PCB image (if provided)
            └── processed/
                └── annotated_defects.png   # Preprocessed frame with CV defect overlays
```

### Storage Path Design
- `input`: Raw unprocessed image as submitted by the technician.
- `reference`: Golden board image against which differential subtraction is evaluated.
- `processed`: Output image rendered with bounding boxes, segmentation masks, and defect labels.

---

## 3. Cloud Firestore Document Schema

### 3.1 Root Collection: `analyses`

Each inspection session is stored as a distinct document within the `analyses` collection:
- **Document Path**: `/analyses/{analysis_id}`
- **Document Key**: `analysis_id` (e.g. `ANA-20260901-0042`)

```json
{
  "analysis_id": "ANA-A1B2C3D4",
  "user_id": "usr_998877",
  "created_at": "2026-09-22T14:30:00Z",
  "completed_at": "2026-09-22T14:30:02Z",
  "input_image_url": "https://storage.googleapis.com/pcb_uploads/usr_998877/ANA-A1B2C3D4/input/input_board.png",
  "reference_image_url": "https://storage.googleapis.com/pcb_uploads/usr_998877/ANA-A1B2C3D4/reference/golden_reference.png",
  "processed_image_url": "https://storage.googleapis.com/pcb_uploads/usr_998877/ANA-A1B2C3D4/processed/annotated_defects.png",
  "status": "completed",
  "overall_result": "DEFECTIVE",
  "confidence": 0.9450,
  "processing_time": 1.842,
  "faults": [
    {
      "fault_id": "FLT-01",
      "fault_type": "missing_component",
      "location": {
        "bbox": [342, 512, 398, 580],
        "reference_designator": "C14",
        "pad_type": "0805_SMD"
      },
      "confidence": 0.9600,
      "severity": "critical_open",
      "evidence": [
        "Unpopulated solder pads identified at C14 silkscreen location",
        "Golden board comparison shows populated 10uF capacitor in reference"
      ],
      "possible_cause": "Feeder misfeed during automated pick-and-place assembly.",
      "impact": "Absence of output filter capacitor on AMS1117 causes voltage rail oscillation.",
      "recommended_action": [
        "Install 10uF 0805 SMD ceramic capacitor onto C14 pads using Sn63/Pb37 solder at 320°C."
      ]
    },
    {
      "fault_id": "FLT-02",
      "fault_type": "broken_copper_track",
      "location": {
        "bbox": [720, 1104, 765, 1150],
        "affected_signal_trace": "3V3_POWER_RAIL"
      },
      "confidence": 0.9200,
      "severity": "critical_open",
      "evidence": [
        "Medial axis skeleton discontinuity detected on 3.3V rail trace."
      ],
      "possible_cause": "Mechanical scratch or chemical over-etching during PCB fabrication.",
      "impact": "Total power disruption to microcontroller U1.",
      "recommended_action": [
        "Bridge fractured copper track using 30 AWG insulated kynar jumper wire per IPC-7721 Section 4.2.3."
      ]
    }
  ],
  "diagnostic_report": {
    "summary": "Dual critical defect: missing capacitor C14 and open fracture on 3.3V rail.",
    "root_cause_analysis": "Mechanical stress combined with assembly feeder error.",
    "subsystem_affected": "Power Distribution & MCU Core"
  },
  "recommendations": [
    "1. Solder 10uF capacitor to C14 pads.",
    "2. Bridge 3.3V copper track break with 30 AWG wire per IPC-7721.",
    "3. Check continuity with multimeter before powering up."
  ],
  "metadata": {
    "board_family": "Microcontroller Interface Board",
    "resolution": [2048, 1536, 3],
    "scale_factor": 1.0
  }
}
```

### 3.2 Field Specifications

| Field | Type | Description |
|---|---|---|
| `analysis_id` | `string` | Unique inspection session ID (`ANA-XXXXXXXX`). |
| `user_id` | `string` | Unique identifier of the authenticated user. |
| `created_at` | `string` | ISO-8601 UTC timestamp of submission. |
| `input_image_url` | `string` | Cloud Storage or public URL of input board image. |
| `reference_image_url` | `string \| null` | URL of golden reference template (if uploaded). |
| `processed_image_url` | `string \| null` | URL of annotated output image. |
| `status` | `string` | `'pending'` \| `'processing'` \| `'completed'` \| `'failed'`. |
| `overall_result` | `string` | `'PASS'` \| `'DEFECTIVE'` \| `'CRITICAL_FAULT'`. |
| `confidence` | `float` | Aggregate confidence score (0.0 to 1.0). |
| `processing_time` | `float` | End-to-end execution latency in seconds. |
| `faults` | `array<object>` | List of validated defect records. |
| `diagnostic_report` | `object` | LLM contextual reasoning narrative. |
| `recommendations` | `array<string>` | Step-by-step IPC repair actions. |

---

## 4. Fault Record Specification

Each item inside the `faults` array follows this schema:

| Attribute | Type | Description |
|---|---|---|
| `fault_id` | `string` | Unique defect ID (e.g. `FLT-01`). |
| `fault_type` | `string` | Defect classification (`missing_component`, `broken_copper_track`, `solder_bridge`, `track_short`). |
| `location` | `object` | Bounding box `[ymin, xmin, ymax, xmax]` or coordinate mapping. |
| `confidence` | `float` | Primary detector confidence (0.0 – 1.0). |
| `severity` | `string` | `'minor'` \| `'moderate'` \| `'high'` \| `'critical_open'`. |
| `evidence` | `array<string>` | Visual and differential evidence points. |
| `possible_cause` | `string` | Inferred manufacturing/operational root cause. |
| `impact` | `string` | Electrical and functional consequence on circuit. |
| `recommended_action` | `array<string>` | IPC-compliant corrective procedures. |

---

## 5. Security & Secret Management Rules

1. **No Credentials in Version Control**:
   - `serviceAccountKey.json`, `*firebase*.json`, and `.env` are strictly included in `.gitignore`.
2. **Environment Variable Configuration**:
   - Keys are configured via `.env` (refer to `.env.example`).
3. **Graceful Offline Fallback**:
   - If cloud credentials are not supplied, all services operate transparently using local disk caching in `results/`, allowing offline development and testing.
