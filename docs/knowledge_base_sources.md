# Knowledge Base Sources & Technical Literature Catalog

## 1. Executive Summary

This document provides complete, verified metadata for all technical engineering literature acquired for the **AI-Based PCB Fault Detection RAG Subsystem** (`knowledge_base/`). 

In compliance with project safety and copyright integrity:
- All documents are obtained from official, publicly accessible manufacturer repositories (Texas Instruments, Microchip Technology).
- Proprietary commercial standards (such as IPC-7711/7721 and IPC-A-610) are documented via authoritative engineering reference summaries rather than unauthorized copies.

---

## 2. Ingested Technical Documents Catalog

### Category 1: Manufacturer Datasheets (`knowledge_base/datasheets/`)

#### 1. `ti_lm1117_datasheet.pdf`
- **Title**: LM1117 800-mA Low-Dropout Linear Regulator Datasheet
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/ds/symlink/lm1117.pdf](https://www.ti.com/lit/ds/symlink/lm1117.pdf)
- **Document Type**: Official Component Datasheet (SNOS412N)
- **Publication Date**: 2000 (Revised 2020)
- **File Size**: 2,791,217 bytes (~2.79 MB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Provides exact ESR and minimum output capacitance (10 µF tantalum or ceramic with series resistor) required to prevent LDO control loop oscillation when output filter capacitors (like C14) are missing.

#### 2. `ti_ne555_datasheet.pdf`
- **Title**: NE555 Precision Timers Datasheet
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/ds/symlink/ne555.pdf](https://www.ti.com/lit/ds/symlink/ne555.pdf)
- **Document Type**: Official Component Datasheet (SLFS022I)
- **Publication Date**: 1973 (Revised 2022)
- **File Size**: 2,258,820 bytes (~2.25 MB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Details pinout, threshold/trigger resistor networks, and decoupling capacitor requirements (0.01 µF on CONT pin).

#### 3. `ti_lm7805_datasheet.pdf`
- **Title**: LM340, LM340A, and LM7805 Series 3-Terminal Positive Regulators
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/ds/symlink/lm340.pdf](https://www.ti.com/lit/ds/symlink/lm340.pdf)
- **Document Type**: Official Component Datasheet (SNOSBD0J)
- **Publication Date**: 2000 (Revised 2016)
- **File Size**: 2,583,681 bytes (~2.58 MB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Specifies input bypass capacitor (0.33 µF) and output transient response capacitor (0.1 µF) to prevent parasitic oscillations on power distribution buses.

---

### Category 2: PCB Faults & Troubleshooting (`knowledge_base/pcb_faults/`)

#### 4. `ti_snva558_thermal_pcb_design.pdf`
- **Title**: AN-2020 Thermal Design By Insight, Not Hindsight
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/an/snva558b/snva558b.pdf](https://www.ti.com/lit/an/snva558b/snva558b.pdf)
- **Document Type**: Application Report (SNVA558B)
- **Publication Date**: 2013
- **File Size**: 188,185 bytes (~188 KB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Explains current-carrying capacity of copper traces, temperature rise, thermal overstress, and trace blowout/fusing failure mechanisms.

#### 5. `ti_slva951_power_layout_faults.pdf`
- **Title**: Layout Guidelines for Sound and Vibration Power Supplies
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/an/slva951/slva951.pdf](https://www.ti.com/lit/an/slva951/slva951.pdf)
- **Document Type**: Application Report (SLVA951)
- **Publication Date**: 2018
- **File Size**: 164,697 bytes (~164 KB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Analyzes trace parasitic inductance, grounding return discontinuities, and signal integrity degradation caused by open/fractured signal nets.

#### 6. `ti_snva021_pcb_layout_guidelines.pdf`
- **Title**: AN-1149 Layout Guidelines for Switching Regulators
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/an/snva021c/snva021c.pdf](https://www.ti.com/lit/an/snva021c/snva021c.pdf)
- **Document Type**: Application Report (SNVA021C)
- **Publication Date**: 2013
- **File Size**: 84,701 bytes (~84 KB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Identifies high dl/dt loop routing faults, noise injection, and ground plane separation symptoms.

---

### Category 3: PCB Repair & Rework Guides (`knowledge_base/repair_guides/`)

#### 7. `ti_snoa405_smt_rework_guidelines.pdf`
- **Title**: AN-1187 Leadless Package User's Guide and Rework Guidelines
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/an/snoa405a/snoa405a.pdf](https://www.ti.com/lit/an/snoa405a/snoa405a.pdf)
- **Document Type**: Application Report (SNOA405A)
- **Publication Date**: 2013
- **File Size**: 594,357 bytes (~594 KB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Provides temperature profiles, solder paste replenishment, pad re-tinning, and flux removal procedures for SMD IC rework.

#### 8. `ipc_7721_procedure_4_2_3_jumper_wire_reference.md`
- **Title**: IPC-7721 Procedure 4.2.3: Jumper Wire Modification & Conductor Repair Reference
- **Publisher / Organization**: IPC (Association Connecting Electronics Industries) Engineering Reference
- **Document Type**: Engineering Standard Procedure Summary
- **Publication Date**: Standard IPC-7711/7721 Rework Specification
- **File Size**: 1,826 bytes
- **License / Access**: Educational / Engineering Procedure Summary
- **Diagnostic Role**: Step-by-step instructions for repairing severed copper tracks using 30 AWG insulated Kynar jumper wire, conductor scraping (1.5 mm exposure), concave solder fillets, and UV-curable solder mask staking.

---

### Category 4: Component Reference & Failure Modes (`knowledge_base/component_reference/`)

#### 9. `microchip_an2519_hardware_design.pdf`
- **Title**: AN2519: AVR Microcontroller Hardware Design Considerations
- **Publisher / Organization**: Microchip Technology Incorporated
- **Source URL**: [https://ww1.microchip.com/downloads/en/Appnotes/AN2519-AVR-Microcontroller-Hardware-Design-Considerations-00002519B.pdf](https://ww1.microchip.com/downloads/en/Appnotes/AN2519-AVR-Microcontroller-Hardware-Design-Considerations-00002519B.pdf)
- **Document Type**: Application Note (DS00002519B)
- **Publication Date**: 2017
- **File Size**: 1,288,414 bytes (~1.28 MB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Explains power routing guidelines, VCC/AVCC decoupling capacitor placement, RESET line pull-up requirements, and oscillator ground shielding.

#### 10. `ti_slva079_ldo_basics.pdf`
- **Title**: Understanding the Terms and Definitions of LDO Voltage Regulators
- **Publisher / Organization**: Texas Instruments Incorporated
- **Source URL**: [https://www.ti.com/lit/an/slva079/slva079.pdf](https://www.ti.com/lit/an/slva079/slva079.pdf)
- **Document Type**: Application Report (SLVA079)
- **Publication Date**: 1999
- **File Size**: 202,342 bytes (~202 KB)
- **License / Access**: Public Manufacturer Technical Literature
- **Diagnostic Role**: Covers equivalent series resistance (ESR) stability boundaries for ceramic vs. tantalum capacitors and circuit consequences of missing output bypass elements.

#### 11. `ipc_a_610_component_acceptance_reference.md`
- **Title**: IPC-A-610: Electronic Assembly Inspection & Acceptance Criteria Reference
- **Publisher / Organization**: IPC Engineering Reference Summary
- **Document Type**: Engineering Standard Summary
- **Publication Date**: Standard IPC-A-610 Inspection Criteria
- **File Size**: 1,179 bytes
- **License / Access**: Educational / Engineering Reference Summary
- **Diagnostic Role**: Class 1, 2, and 3 criteria for missing components, solder joint fillet percentages, and maximum allowable conductor necking (mousebites).

---

## 3. Summary of Knowledge Base Assets

| Category | Filename | Publisher | Size | Status |
|---|---|---|---|---|
| **Datasheet** | `ti_lm1117_datasheet.pdf` | Texas Instruments | 2.79 MB | Verified on Disk |
| **Datasheet** | `ti_ne555_datasheet.pdf` | Texas Instruments | 2.25 MB | Verified on Disk |
| **Datasheet** | `ti_lm7805_datasheet.pdf` | Texas Instruments | 2.58 MB | Verified on Disk |
| **PCB Faults** | `ti_snva558_thermal_pcb_design.pdf` | Texas Instruments | 188 KB | Verified on Disk |
| **PCB Faults** | `ti_slva951_power_layout_faults.pdf` | Texas Instruments | 164 KB | Verified on Disk |
| **PCB Faults** | `ti_snva021_pcb_layout_guidelines.pdf` | Texas Instruments | 84 KB | Verified on Disk |
| **Repair Guide** | `ti_snoa405_smt_rework_guidelines.pdf` | Texas Instruments | 594 KB | Verified on Disk |
| **Repair Guide** | `ipc_7721_procedure_4_2_3_jumper_wire_reference.md` | IPC Reference | 1.8 KB | Verified on Disk |
| **Comp Ref** | `microchip_an2519_hardware_design.pdf` | Microchip Tech | 1.28 MB | Verified on Disk |
| **Comp Ref** | `ti_slva079_ldo_basics.pdf` | Texas Instruments | 202 KB | Verified on Disk |
| **Comp Ref** | `ipc_a_610_component_acceptance_reference.md` | IPC Reference | 1.2 KB | Verified on Disk |
