# Domain Knowledge Base Requirements & Source Blueprint

## 1. Executive Overview

In Phase 6 (Retrieval-Augmented Generation & Cognitive Diagnostic Reasoning), the system will cross-reference verified Computer Vision physical detections with technical literature to:
1. Explain the **root cause** of detected faults.
2. Determine the **circuit-level electrical impact** on components and signal nets.
3. Prescribe **standardized IPC corrective repair procedures**.
4. Assess **safety hazards and test procedures** before re-powering.

To prevent hallucinations, the Knowledge Base (`knowledge_base/`) must be constructed from authoritative public electronics and engineering documentation.

---

## 2. Required Document Categories & Recommended Legitimate Public Sources

```
knowledge_base/
├── datasheets/             # Component technical specs, pinouts, and application circuits
├── pcb_faults/             # FMEA tables, manufacturing root-cause analyses, failure modes
├── repair_guides/          # IPC-7711/7721 rework procedures, jumper wiring, solder mask curing
└── component_reference/    # Package dimensions, SMD footprint codes, reference designators
```

---

### Category 1: Electronic Component Datasheets (`knowledge_base/datasheets/`)

#### Purpose in Diagnosis
Provides nominal operating voltages, maximum ratings, pin functions (VDD, GND, OUT, EN), and required passive filtering components (e.g., input/output bypass capacitors). When an output capacitor (like `C14`) is missing near a voltage regulator (`AMS1117`), the datasheet provides the mathematical explanation of control-loop instability and ripple voltage.

#### Recommended Target Documents & Official Sources
1. **Low-Dropout Voltage Regulators (LDOs)**:
   - `AMS1117` / `LM1117` Datasheet (Advanced Monolithic Systems / Texas Instruments)
   - `LM7805` / `LM317` Voltage Regulator Datasheets (Texas Instruments / STMicroelectronics)
   - *Official Sources*: [ti.com](https://www.ti.com), [st.com](https://www.st.com)
2. **Standard Microcontrollers & ICs**:
   - `ATmega328P` Datasheet (Microchip Technology) — VCC, AVCC, decoupling recommendations
   - `NE555` Precision Timer Datasheet (Texas Instruments)
   - `STM32F103C8T6` ARM Cortex-M3 Reference Manual (STMicroelectronics)
   - *Official Sources*: [microchip.com](https://www.microchip.com), [st.com](https://www.st.com)
3. **Passive Component Standards**:
   - SMD Ceramic Capacitor Spec Sheets (Yageo / KEMET / Murata)
   - SMD Thick Film Resistor 0805 / 0603 Spec Sheets (Vishay / Panasonic)
   - *Official Sources*: [murata.com](https://www.murata.com), [vishay.com](https://www.vishay.com)

---

### Category 2: PCB Fault & Troubleshooting Information (`knowledge_base/pcb_faults/`)

#### Purpose in Diagnosis
Maps physical visual symptoms (open track, missing part, solder bridge) to manufacturing/assembly root causes and electrical consequences using empirical Failure Mode and Effects Analysis (FMEA).

#### Recommended Target Documents & Public Sources
1. **PCB Assembly Fault Matrices**:
   - *SMT Defect Troubleshooting Guide* (SMTA - Surface Mount Technology Association)
   - *PCB Etching and Photolithography Process Defects* (IPC Quality Resources)
2. **Failure Mode and Effects Analysis (FMEA)**:
   - Circuit Board Failure Analysis Guides (NASA Technical Reports Server / IEEE Xplore Open)
   - *Public Sources*: [ntrs.nasa.gov](https://ntrs.nasa.gov), [smta.org](https://www.smta.org)

---

### Category 3: Copper Track & Trace Repair Information (`knowledge_base/repair_guides/`)

#### Purpose in Diagnosis
Provides step-by-step IPC-standard rework procedures for repairing severed copper tracks, fractured conductors, lifted solder pads, and jumper wire installations.

#### Recommended Target Documents & Public Standards
1. **IPC-7711/7721 Standard: Rework, Modification, and Repair of Electronic Assemblies**:
   - *Procedure 4.2.3*: Jumper Wire Method for Conductor Repair (wire gauge selection, insulation, route guidelines)
   - *Procedure 4.3.1*: Foil and Trace Replacement using Copper Tape & Epoxy
   - *Procedure 3.5.1*: Lifted Pad and Land Restoration
2. **Soldering & Surface Prep Guidelines**:
   - IPC J-STD-001 (Requirements for Soldered Electrical and Electronic Assemblies)
   - *Official Sources*: [ipc.org](https://www.ipc.org), Manufacturer Application Notes (CircuitMedic, Chemtronics)

---

### Category 4: Missing Component & SMT Failure Information (`knowledge_base/pcb_faults/`)

#### Purpose in Diagnosis
Explains the failure mechanisms that lead to missing, unpopulated, or tombstoned components during assembly (feeder misfeed, solder paste volume deficiency, thermal imbalance) and their functional consequences.

#### Recommended Target Documents & Public Sources
1. **SMT Pick-and-Place & Reflow Failure Analysis**:
   - Pick-and-Place Feeder Misfeed and Tape Pocket Fault Guides (IPC / SMTA Publications)
   - Tombstoning and De-wetting Root Cause Diagnostics (Indium Corporation Tech Papers)
   - *Official Sources*: [indium.com/technical-documents](https://www.indium.com), [smta.org](https://www.smta.org)

---

### Category 5: PCB Inspection & Quality Acceptance Guidelines (`knowledge_base/component_reference/`)

#### Purpose in Diagnosis
Provides standard criteria for defect classification, minimum conductor width reduction thresholds (necking tolerances), and visual acceptance criteria.

#### Recommended Target Documents & Public Sources
1. **IPC-A-610: Acceptability of Electronic Assemblies**:
   - Section 6: Component Installation & Solder Criteria
   - Section 10: Discrete Wiring & Jumper Acceptability
2. **Component Package & Footprint Standards**:
   - IPC-7351: Generic Requirements for Surface Mount Design and Land Pattern Standard (0402, 0603, 0805, 1206, SOT-23, SOIC, QFP dimensions)
   - *Official Sources*: [ipc.org](https://www.ipc.org)

---

## 3. Knowledge Base Ingestion Architecture for Phase 6

During Phase 6, documents added to these folders will be:
1. **Parsed and Chunked**: Converted to structured semantic chunks preserving circuit tables, pinouts, and repair procedures.
2. **Indexed**: Vector embeddings generated via Gemini / TF-IDF hybrid search in `src/services/rag_service.py`.
3. **Queried Contextually**: When the CV detector flags a broken track or missing component, relevant technical excerpts are retrieved and provided to Gemini for root-cause synthesis.
