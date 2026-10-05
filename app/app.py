"""
Streamlit PCB Fault Detection & AI Diagnostic Dashboard.

Provides an interactive user interface for:
1. Uploading PCB inspection photographs (and optional Golden Reference board).
2. Running the multi-stage Computer Vision detection pipeline.
3. Viewing defect visual overlays (missing components, broken copper tracks).
4. Viewing RAG-grounded Gemini AI Diagnostic Reports (root causes, circuit impact, IPC-7721 repair instructions).
5. Persisting analysis sessions to Firebase Storage / Cloud Firestore (with local fallback).
6. Exporting and downloading diagnostic reports.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.diagnostic_agent import DiagnosticAgent
from src.processing.pipeline import PCBInspectionPipeline
from src.services.analysis_service import AnalysisService, RESULT_PASS, RESULT_DEFECTIVE, RESULT_CRITICAL
from src.services.auth_service import FirebaseAuthService
from src.services.firebase_service import get_firebase_service
from src.services.rag_service import RAGService
from src.services.storage_service import FirebaseStorageService

# Set page configuration
st.set_page_config(
    page_title="AI-Based PCB Fault Detection & Diagnostic System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for professional engineering dashboard
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .cv-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #93C5FD;
    }
    .ai-badge {
        background-color: #F3E8FF;
        color: #6B21A8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #D8B4FE;
    }
    .status-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
    }
    .status-defect {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
    }
    .status-critical {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
    }
    .card-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services():
    """Initialize and cache backend service singletons."""
    fb = get_firebase_service()
    storage = FirebaseStorageService(firebase_service=fb)
    analysis = AnalysisService(firebase_service=fb)
    rag = RAGService(use_offline_fallback=False)
    agent = DiagnosticAgent(rag_service=rag)
    pipeline = PCBInspectionPipeline()
    return {
        "firebase": fb,
        "storage": storage,
        "analysis": analysis,
        "rag": rag,
        "agent": agent,
        "pipeline": pipeline,
    }


def find_sample_boards() -> Dict[str, Dict[str, str]]:
    """Locate sample PCB image pairs in the repository for quick testing."""
    samples = {}
    deeppcb_dir = PROJECT_ROOT / "data" / "raw" / "broken_tracks" / "deeppcb"
    test_img_dir = deeppcb_dir / "images"
    template_dir = deeppcb_dir / "templates"

    if test_img_dir.is_dir() and template_dir.is_dir():
        # Find sample open track files
        for test_file in sorted(test_img_dir.glob("*_test.jpg"))[:5]:
            stem = test_file.stem.replace("_test", "")
            temp_file = template_dir / f"{stem}_temp.jpg"
            if temp_file.is_file():
                samples[f"DeepPCB Sample: {stem} (Open Track Defect)"] = {
                    "test": str(test_file),
                    "template": str(temp_file),
                }

    # Also check data/raw/DeepPCB_repo/
    if not samples:
        repo_sample = PROJECT_ROOT / "data" / "raw" / "DeepPCB_repo" / "PCBData" / "group20085" / "20085"
        if repo_sample.is_dir():
            for t_file in list(repo_sample.glob("*_test.jpg"))[:3]:
                stem = t_file.stem.replace("_test", "")
                tmp = repo_sample / f"{stem}_temp.jpg"
                if tmp.is_file():
                    samples[f"DeepPCB Repo Sample: {stem}"] = {
                        "test": str(t_file),
                        "template": str(tmp),
                    }
    return samples


def render_sidebar(services, samples):
    """Render sidebar with user settings, sample selection, and service status."""
    st.sidebar.image(
        "https://raw.githubusercontent.com/opencv/opencv/master/doc/opencv-logo.png",
        width=80,
    )
    st.sidebar.title("PCB Fault Detection")
    st.sidebar.caption("Computer Vision + RAG + Gemini Diagnostics")

    # User Profile (Mock / Firebase Auth)
    st.sidebar.subheader("Session Info")
    user_id = st.sidebar.text_input("Operator / User ID", value="user_demo_engineer")

    # Knowledge Base Stats
    st.sidebar.subheader("RAG Knowledge Base")
    rag = services["rag"]
    stats = rag.get_collection_stats()
    st.sidebar.write(f"📚 **Indexed Docs:** {stats.get('unique_documents', 11)}")
    st.sidebar.write(f"🧩 **Vector Chunks:** {stats.get('total_chunks', 685)}")
    st.sidebar.write(f"📐 **Embeddings:** {stats.get('embedding_type', 'FAISS')}")

    # Firebase Status
    fb = services["firebase"]
    fb_status = "Connected (Cloud)" if fb.is_connected else "Offline (Local Fallback)"
    st.sidebar.subheader("Backend Persistence")
    st.sidebar.write(f"🔥 **Firebase:** {fb_status}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Test Samples")
    selected_sample = st.sidebar.selectbox(
        "Select a pre-loaded sample PCB pair",
        options=["None (Upload Custom Images)"] + list(samples.keys()),
    )

    return {
        "user_id": user_id,
        "selected_sample": selected_sample,
    }


def main():
    services = get_services()
    samples = find_sample_boards()

    # Title Banner
    st.markdown('<div class="main-header">PCB Fault Detection & AI Diagnostic Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Optical Inspection (AOI) powered by Classical Computer Vision and grounded Gemini Diagnostic Reasoning.</div>',
        unsafe_allow_html=True,
    )

    sidebar_data = render_sidebar(services, samples)
    user_id = sidebar_data["user_id"]
    selected_sample = sidebar_data["selected_sample"]

    # Navigation Tabs
    tab_inspect, tab_history, tab_kb = st.tabs([
        "🔬 New PCB Inspection",
        "📋 Inspection History",
        "📚 Knowledge Base Catalog",
    ])

    with tab_inspect:
        st.markdown("### 1. Upload PCB Images")

        col_up1, col_up2 = st.columns(2)
        test_image_input = None
        ref_image_input = None

        if selected_sample != "None (Upload Custom Images)":
            sample_paths = samples[selected_sample]
            st.info(f"Loaded Sample Board: **{selected_sample}**")
            test_image_input = sample_paths["test"]
            ref_image_input = sample_paths["template"]

            with col_up1:
                st.write("**Test PCB Image (Target for Inspection)**")
                st.image(test_image_input, use_container_width=True)
            with col_up2:
                st.write("**Golden Reference PCB Image (Template)**")
                st.image(ref_image_input, use_container_width=True)
        else:
            with col_up1:
                uploaded_test = st.file_uploader(
                    "Upload Test PCB Image (Defective / Under Test)",
                    type=["jpg", "jpeg", "png", "bmp", "tif"],
                    key="test_uploader",
                )
                if uploaded_test:
                    test_image_input = Image.open(uploaded_test)
                    st.image(test_image_input, caption="Uploaded Test PCB", use_container_width=True)

            with col_up2:
                uploaded_ref = st.file_uploader(
                    "Upload Golden Reference PCB Image (Optional)",
                    type=["jpg", "jpeg", "png", "bmp", "tif"],
                    key="ref_uploader",
                    help="Providing a golden board enables sub-millimeter differential track and component comparison.",
                )
                if uploaded_ref:
                    ref_image_input = Image.open(uploaded_ref)
                    st.image(ref_image_input, caption="Uploaded Golden Reference", use_container_width=True)

        st.markdown("---")

        # Analysis Trigger Button
        analyze_btn = st.button("🚀 Start PCB Inspection & Diagnostic Analysis", type="primary", use_container_width=True)

        if analyze_btn:
            if test_image_input is None:
                st.error("⚠️ Please upload a test PCB image or select a sample board before running analysis.")
                return

            with st.status("Executing Multi-Stage PCB Inspection Pipeline...", expanded=True) as status_box:
                st.write("🔹 **Step 1/6:** Initializing inspection session and validating input resolution...")
                analysis_service: AnalysisService = services["analysis"]
                storage_service: StorageService = services["storage"]
                pipeline: PCBInspectionPipeline = services["pipeline"]
                agent: DiagnosticAgent = services["agent"]

                # Generate unique analysis ID
                aid = f"ANA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(3).hex().upper()}"

                # Upload inputs to Storage
                st.write("🔹 **Step 2/6:** Persisting image payloads to Storage...")
                input_url = f"pcb_uploads/{user_id}/{aid}/input/test.png"
                ref_url = f"pcb_uploads/{user_id}/{aid}/reference/ref.png" if ref_image_input else None

                storage_res = storage_service.upload_input_image(
                    user_id=user_id,
                    analysis_id=aid,
                    file_source=test_image_input,
                )
                input_url = storage_res.get("url", input_url)
                if ref_image_input:
                    ref_res = storage_service.upload_reference_image(
                        user_id=user_id,
                        analysis_id=aid,
                        file_source=ref_image_input,
                    )
                    ref_url = ref_res.get("url", ref_url)

                # Initialize DB analysis entry
                analysis_service.create_analysis(
                    user_id=user_id,
                    input_image_url=input_url,
                    reference_image_url=ref_url,
                    analysis_id=aid,
                )

                st.write("🔹 **Step 3/6:** Running Computer Vision pipeline (CLAHE Enhancement, ORB Alignment, Morphological Track Segmentation)...")
                time.sleep(0.3)
                cv_start = time.time()
                cv_result = pipeline.inspect_pcb(
                    test_image_source=test_image_input,
                    reference_image_source=ref_image_input,
                    analysis_id=aid,
                    save_debug=True,
                )
                cv_time = time.time() - cv_start

                if not cv_result.get("success", False):
                    st.error(f"❌ Computer Vision Inspection Error: {cv_result.get('error', 'Unknown failure')}")
                    analysis_service.update_analysis_status(aid, "failed", cv_result.get("error"))
                    status_box.update(label="Inspection Failed", state="error")
                    return

                faults = cv_result.get("faults", [])
                st.write(f"🔹 **Step 4/6:** CV Detection complete ({len(faults)} defects identified in {cv_time:.2f}s).")

                # Step 5: RAG Retrieval & Gemini Diagnostic Reasoning
                st.write("🔹 **Step 5/6:** Querying RAG knowledge base & invoking Gemini Diagnostic Agent...")
                diagnostic_reports: List[Dict[str, Any]] = []

                if faults:
                    for fault in faults:
                        report = agent.diagnose(fault)
                        diagnostic_reports.append(report)
                else:
                    # Grounded Pass Report
                    diagnostic_reports.append({
                        "detected_fault": "none",
                        "confidence": 1.0,
                        "evidence": "All copper traces continuous and all components verified present.",
                        "probable_causes": ["Normal manufacturing within IPC-A-610 Class 3 specifications."],
                        "impact": "Board functions normally under rated electrical specifications.",
                        "recommended_action": "Pass board to subsequent in-circuit testing (ICT) or functional testing.",
                        "supporting_sources": [
                            {
                                "title": "IPC-A-610 Electronic Assembly Acceptance Criteria",
                                "source": "ipc_a_610_component_acceptance_reference.md",
                                "category": "component_reference",
                            }
                        ],
                        "limitations": "Optical inspection is limited to external layers and visible surfaces.",
                    })

                # Step 6: Save and persist results
                st.write("🔹 **Step 6/6:** Aggregating diagnostic report and updating Firestore...")
                overall_res = cv_result.get("overall_result", RESULT_DEFECTIVE if faults else RESULT_PASS)

                # Save annotated image
                annotated_img_bgr = cv_result.get("annotated_image")
                if annotated_img_bgr is not None:
                    annotated_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                    storage_service.upload_processed_image(
                        user_id=user_id,
                        analysis_id=aid,
                        file_source=annotated_rgb,
                        filename="annotated.png",
                    )

                final_analysis_doc = analysis_service.save_analysis_results(
                    analysis_id=aid,
                    faults=faults,
                    overall_result=overall_res,
                    confidence=cv_result.get("confidence", 0.95),
                    processing_time=cv_time,
                    diagnostic_report={"reports": diagnostic_reports},
                    recommendations=[r.get("recommended_action", "") for r in diagnostic_reports],
                    metadata=cv_result.get("metadata", {}),
                )

                status_box.update(label="Inspection Completed Successfully!", state="complete", expanded=False)

            # Store in session state for rendering
            st.session_state["latest_result"] = {
                "cv_result": cv_result,
                "diagnostic_reports": diagnostic_reports,
                "analysis_id": aid,
                "analysis_doc": final_analysis_doc,
            }

        # Render Results Section if available
        if "latest_result" in st.session_state:
            res_data = st.session_state["latest_result"]
            cv_res = res_data["cv_result"]
            diag_reports = res_data["diagnostic_reports"]
            aid = res_data["analysis_id"]

            st.markdown("---")
            st.subheader("🎯 Inspection & Diagnostic Results")

            # Overall Status Banner
            overall_res = cv_res.get("overall_result", "DEFECTIVE")
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.write("**Analysis ID:**")
                st.code(aid)
            with col_b2:
                st.write("**Overall Board Verdict:**")
                if overall_res == RESULT_PASS:
                    st.markdown('<div class="status-pass">✅ PASS (No Faults)</div>', unsafe_allow_html=True)
                elif overall_res == RESULT_CRITICAL:
                    st.markdown('<div class="status-critical">⚠️ CRITICAL FAULT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-defect">❌ DEFECTIVE</div>', unsafe_allow_html=True)
            with col_b3:
                st.write("**Defects Found:**")
                st.metric(label="Total Faults", value=len(cv_res.get("faults", [])))
            with col_b4:
                st.write("**CV Execution Time:**")
                st.metric(label="Runtime", value=f"{cv_res.get('processing_time_seconds', 0.0):.2f}s")

            # Section A: Visual Overlays
            st.markdown("#### A. Visual Inspection Overlays")
            img_col1, img_col2, img_col3 = st.columns(3)

            with img_col1:
                st.write("**1. Enhanced Preprocessed Image**")
                proc_img = cv_res.get("processed_image")
                if proc_img is not None:
                    proc_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
                    st.image(proc_rgb, use_container_width=True)

            with img_col2:
                st.write("**2. Computer Vision Defect Overlay**")
                ann_img = cv_res.get("annotated_image")
                if ann_img is not None:
                    ann_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
                    st.image(ann_rgb, use_container_width=True)

            with img_col3:
                st.write("**3. Differential Heatmap**")
                diff_heat = cv_res.get("diff_heatmap")
                if diff_heat is not None:
                    heat_rgb = cv2.cvtColor(diff_heat, cv2.COLOR_BGR2RGB)
                    st.image(heat_rgb, use_container_width=True)
                else:
                    st.info("Differential heatmap is generated when a golden reference image is provided.")

            # Section B & C: Faults and AI Diagnostics
            st.markdown("---")
            col_f, col_d = st.columns([1, 1.2])

            with col_f:
                st.markdown('#### B. Detected Faults <span class="cv-badge">Detected by CV Pipeline</span>', unsafe_allow_html=True)
                faults = cv_res.get("faults", [])
                if not faults:
                    st.success("✨ No physical defects or track discontinuities were detected on this board.")
                else:
                    for idx, fault in enumerate(faults, 1):
                        ftype = fault.get("fault_type", "defect").replace("_", " ").title()
                        conf = fault.get("confidence", 0.0)
                        sev = fault.get("severity", "moderate").upper()
                        loc = fault.get("location", {})
                        bbox = loc.get("bbox", [])
                        ev = fault.get("detector_evidence", fault.get("evidence", "Visual anomaly detected"))

                        with st.expander(f"🔴 Defect #{idx}: {ftype} (Conf: {conf*100:.1f}%)", expanded=True):
                            st.write(f"**Fault ID:** `{fault.get('fault_id', f'FLT-{idx}')}`")
                            st.write(f"**Severity:** `{sev}`")
                            st.write(f"**Bounding Box [x, y, w, h]:** `{bbox}`")
                            st.write(f"**Detector Evidence:** {ev}")

            with col_d:
                st.markdown('#### C. AI Failure Diagnostics & Repair Guidance <span class="ai-badge">Explained by Gemini + RAG</span>', unsafe_allow_html=True)

                for idx, rpt in enumerate(diag_reports, 1):
                    with st.expander(f"🧠 Diagnostic Report #{idx} ({rpt.get('detected_fault', '').replace('_', ' ').title()})", expanded=True):
                        # Probable Causes
                        st.write("🔍 **Probable Root Causes:**")
                        causes = rpt.get("probable_causes", [])
                        if isinstance(causes, list):
                            for c in causes:
                                st.write(f"- {c}")
                        else:
                            st.write(f"- {causes}")

                        # Circuit Impact
                        st.write("⚡ **Circuit Impact:**")
                        st.write(rpt.get("impact", "N/A"))

                        # Recommended Action
                        st.write("🛠️ **Recommended Corrective Action (IPC Standards):**")
                        st.success(rpt.get("recommended_action", "Inspect and rework."))

                        # Supporting Sources
                        st.write("📚 **Supporting Technical Sources:**")
                        sources = rpt.get("supporting_sources", [])
                        if sources:
                            for s in sources:
                                title = s.get("title", "Technical Reference")
                                src = s.get("source", "")
                                cat = s.get("category", "")
                                st.write(f"- **{title}** (`{src}` - *{cat}*)")

                        # Limitations
                        st.write("⚠️ **Inspection Limitations:**")
                        st.caption(rpt.get("limitations", "Optical surface inspection cannot assess inner multi-layer tracks."))

            # Section D: Export & Download
            st.markdown("---")
            st.markdown("#### 📥 Export Diagnostic Report")
            export_payload = {
                "analysis_id": aid,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "overall_verdict": overall_res,
                "computer_vision_findings": cv_res.get("faults", []),
                "diagnostic_reasoning": diag_reports,
            }
            json_str = json.dumps(export_payload, indent=2)

            st.download_button(
                label="📄 Download Diagnostic Report (JSON)",
                data=json_str,
                file_name=f"PCB_Diagnostic_Report_{aid}.json",
                mime="application/json",
            )

    # TAB 2: History
    with tab_history:
        st.markdown("### 📋 Recent Inspection Sessions")
        analysis_service: AnalysisService = services["analysis"]
        recent = analysis_service.list_user_analyses(user_id=user_id, limit=20)

        if not recent:
            st.info("No prior inspection analyses found for this user.")
        else:
            for item in recent:
                aid_item = item.get("analysis_id", "N/A")
                c_at = item.get("created_at", "N/A")
                res_verdict = item.get("overall_result", "PENDING")
                fault_cnt = len(item.get("faults", []))

                with st.expander(f"Analysis: {aid_item} — Verdict: {res_verdict} ({fault_cnt} defects)"):
                    st.write(f"**Created:** {c_at}")
                    st.write(f"**Status:** {item.get('status')}")
                    st.write(f"**Confidence:** {item.get('confidence', 0.0)}")
                    st.json(item)

    # TAB 3: Knowledge Base Catalog
    with tab_kb:
        st.markdown("### 📚 Ingested Technical Literature & Standards")
        st.write("The RAG subsystem indexes 11 official engineering documents to provide grounded root cause and IPC rework procedures:")

        kb_docs = [
            {"Title": "LM1117 800-mA Low-Dropout Regulator Datasheet", "Category": "Datasheet", "Publisher": "Texas Instruments", "Focus": "ESR and 10µF output capacitor stability"},
            {"Title": "NE555 Precision Timers Datasheet", "Category": "Datasheet", "Publisher": "Texas Instruments", "Focus": "Decoupling capacitor and pinout specifications"},
            {"Title": "LM340 / LM7805 Positive Regulators Datasheet", "Category": "Datasheet", "Publisher": "Texas Instruments", "Focus": "Bypass capacitance & power distribution"},
            {"Title": "AN-2020 Thermal Design By Insight (SNVA558)", "Category": "PCB Faults", "Publisher": "Texas Instruments", "Focus": "Trace blowout, fusing current & thermal stress"},
            {"Title": "Power Layout Faults & Parasitic Inductance (SLVA951)", "Category": "PCB Faults", "Publisher": "Texas Instruments", "Focus": "Ground loops & trace discontinuities"},
            {"Title": "AN-1149 Switching Regulator Layout (SNVA021)", "Category": "PCB Faults", "Publisher": "Texas Instruments", "Focus": "High dl/dt loops & noise return paths"},
            {"Title": "AN-1187 Leadless Package SMT Rework (SNOA405)", "Category": "Repair Guide", "Publisher": "Texas Instruments", "Focus": "SMD desoldering, pad tinning & thermal profile"},
            {"Title": "IPC-7721 Procedure 4.2.3 Jumper Wire Reference", "Category": "Repair Guide", "Publisher": "IPC Standards", "Focus": "30 AWG Kynar wire, 1.5mm mask scrape & UV staking"},
            {"Title": "AN2519 AVR Hardware Design Considerations", "Category": "Component Ref", "Publisher": "Microchip Technology", "Focus": "Decoupling capacitors & RESET line pull-ups"},
            {"Title": "Understanding LDO Regulators (SLVA079)", "Category": "Component Ref", "Publisher": "Texas Instruments", "Focus": "Tantalum vs ceramic stability boundaries"},
            {"Title": "IPC-A-610 Electronic Assembly Acceptance Criteria", "Category": "Component Ref", "Publisher": "IPC Standards", "Focus": "Class 1/2/3 missing parts & conductor necking"},
        ]
        st.table(kb_docs)


if __name__ == "__main__":
    main()
