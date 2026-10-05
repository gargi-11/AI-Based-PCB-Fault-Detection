# Team Setup Guide: AI-Based PCB Fault Detection Platform

Welcome to the **AI-Based PCB Fault Detection & AI Diagnostic Platform** team! This guide walks you step-by-step through setting up, configuring, and verifying the complete project on a Windows development machine.

---

## Architecture Overview

Before starting, here is how the system components fit together:

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Web UI (app/app.py)             │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│   Computer Vision Pipeline   │ │   Diagnostic AI Agent      │
│  - CLAHE Preprocessing       │ │  - Google Gemini API       │
│  - Alignment & Differential  │ │  - FAISS Vectorstore (RAG) │
│  - Broken Track & Component  │ │  - IPC & TI Standards KB   │
└──────────────┬───────────────┘ └─────────────┬──────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │  Data Persistence (Cloud / Local Fallback)   │
        │  - Cloud: Firebase Firestore & Storage       │
        │  - Offline Fallback: Local results/ folder   │
        └──────────────────────────────────────────────┘
```

---

## 1. Prerequisites

Ensure your Windows machine has the following tools installed:

| Tool | Minimum Version | Installation & Notes |
| :--- | :--- | :--- |
| **Windows OS** | Windows 10 or 11 (64-bit) | PowerShell 5.1 or PowerShell 7+ recommended. |
| **Python** | **Python 3.12** *(3.10 – 3.12 supported)* | Download from [python.org](https://www.python.org/downloads/). **Check "Add python.exe to PATH"** during setup. |
| **Git** | Latest 64-bit | Download from [git-scm.com](https://git-scm.com/download/win). |
| **Antigravity IDE** | Latest Version | Google Antigravity Agentic IDE workspace. |
| **C++ Build Tools** *(Optional)* | Visual Studio C++ Build Tools | Required only if compiling binary dependencies from source (wheels are pre-built for Python 3.12). |

To verify your installations, open **PowerShell** and run:
```powershell
git --version
python --version
```
Expected output:
```text
git version 2.x.x
Python 3.12.x (or 3.10.x / 3.11.x)
```

---

## 2. Clone the GitHub Repository

Open **PowerShell** or **Windows Terminal**, navigate to your preferred workspace directory (e.g., `Desktop` or `Projects`), and clone the repository:

```powershell
git clone https://github.com/gargi-11/AI-Based-PCB-Fault-Detection.git
cd AI-Based-PCB-Fault-Detection
```

Verify that you are in the project root:
```powershell
Get-ChildItem
```
You should see folders such as `app`, `docs`, `knowledge_base`, `src`, and `tests`.

---

## 3. Obtain the Data Package (`PCB_PROJECT_DATA_v1.zip`)

Because raw PCB image datasets and pre-computed vector embeddings are large (~270 MB), they are distributed outside of Git version control.

1. Obtain `PCB_PROJECT_DATA_v1.zip` from your team lead or shared team storage:
   - **Google Drive / Shared Drive link** (provided internally by the team lead)
   - Or USB drive / team NAS
2. Copy or move `PCB_PROJECT_DATA_v1.zip` directly into the project root directory:
   ```powershell
   # Target location:
   AI-Based-PCB-Fault-Detection\PCB_PROJECT_DATA_v1.zip
   ```

---

## 4. Extract the Data Folder into the Project Root

The archive contains the complete `data/` folder structure, including raw PCB test images and the pre-indexed FAISS knowledge base vectorstore.

### Option A: Using PowerShell (Recommended)
Run the following command from the project root:
```powershell
Expand-Archive -Path PCB_PROJECT_DATA_v1.zip -DestinationPath . -Force
```

### Option B: Using Windows File Explorer
1. Right-click `PCB_PROJECT_DATA_v1.zip`.
2. Click **Extract All...**.
3. Set the destination path to the project root: `...\AI-Based-PCB-Fault-Detection`.
4. Click **Extract**.

### Verify Extracted Folder Structure
Ensure that the directory is located at `AI-Based-PCB-Fault-Detection\data` (not nested inside `data\data`):

```powershell
Get-ChildItem data
```

Expected directory tree:
```text
AI-Based-PCB-Fault-Detection/
├── data/
│   ├── raw/                 <- DeepPCB & synthetic PCB defect images
│   ├── vectorstore/         <- Pre-computed FAISS RAG index:
│   │   ├── faiss_index.bin
│   │   ├── chunks_metadata.json
│   │   └── vectorstore_config.json
│   ├── processed/           <- Alignment and intermediate preprocessed images
│   ├── chroma_db/
│   └── test/
```

---

## 5. Create and Activate a Python Virtual Environment

Isolate project dependencies by creating a dedicated virtual environment:

```powershell
# 1. Create the virtual environment named 'venv'
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

> **Note on PowerShell Script Execution Policy:**  
> If you encounter `cannot be loaded because running scripts is disabled on this system`, run this command in your PowerShell session and try activating again:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> .\venv\Scripts\Activate.ps1
> ```

When activated, you will see `(venv)` prepended to your terminal prompt:
```text
(venv) PS C:\...\AI-Based-PCB-Fault-Detection>
```

---

## 6. Install Project Dependencies

Upgrade `pip` and install all required libraries listed in `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run Sanity Check
Test that core dependencies (OpenCV, NumPy, Pandas, PIL) were installed properly:
```powershell
python src/test_setup.py
```
Expected output:
```text
================================
PCB PROJECT SETUP SUCCESSFUL!
================================
Python is working
OpenCV version: 5.x.x (or 4.x.x)
NumPy version: 2.x.x
Pandas version: 3.x.x
================================
```

---

## 7. Create your `.env` Configuration File

Create a local `.env` file by copying the provided `.env.example` template:

```powershell
Copy-Item .env.example .env
```

Open `.env` in your editor (e.g., Antigravity, VS Code, or Notepad):
```powershell
notepad .env
```

---

## 8. Configure API Keys and Environment Variables

Here is what you need to configure in `.env`:

```env
# ==============================================================================
# AI-Based PCB Fault Detection - Environment Configuration
# ==============================================================================

# 1. Google Gemini API Configuration (REQUIRED for AI Diagnostics)
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash

# 2. Firebase Configuration (OPTIONAL for local development)
FIREBASE_PROJECT_ID=
FIREBASE_STORAGE_BUCKET=
FIREBASE_WEB_API_KEY=
FIREBASE_SERVICE_ACCOUNT_PATH=

# 3. Offline / Local Fallback Mode (KEEP AS 'true' FOR LOCAL USE)
FIREBASE_MOCK_FALLBACK=true
```

### Values You Must Provide:
1. **`GEMINI_API_KEY` (Required for Gemini AI diagnostics):**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Sign in with your Google account and click **Create API Key**.
   - Copy the API key and paste it as `GEMINI_API_KEY=AIzaSy...` in your `.env` file.
   - *Free tier works completely fine for all testing.*
2. **`GEMINI_MODEL`:**
   - Keep default: `gemini-3.6-flash` (or `gemini-2.5-flash`).
3. **`FIREBASE_MOCK_FALLBACK`:**
   - Set to `true`. This allows full local execution without cloud setup.

### Test Gemini Connection
Verify your Gemini API key is configured correctly:
```powershell
python tests/test_gemini_connection.py
```
Expected output:
```text
Gemini API connection successful.
```

---

## 9. Firebase Setup & Access Modes

This project supports two execution modes:

### Mode A: Local Offline Mode (Recommended for Teammates)
- **Zero Firebase credentials required!**
- With `FIREBASE_MOCK_FALLBACK=true` in `.env`, the system automatically:
  - Stores uploaded and annotated images locally under `results/storage/`.
  - Persists analysis records and diagnostic reports locally under `results/analyses/`.
  - Emulates user sessions without connecting to external cloud services.
- **You are fully operational immediately.**

### Mode B: Cloud Firebase Mode (Optional)
If your role involves syncing inspection history and images to the shared Cloud Firestore and Firebase Storage:
1. **Request Access:** Ask the team project owner to add your Google account to the team Firebase project with `Editor` or `Viewer` role.
2. **Download Service Account Key:**
   - In [Firebase Console](https://console.firebase.google.com/), go to **Project Settings** (⚙️) > **Service accounts**.
   - Click **Generate new private key** and download the resulting JSON file.
3. **Place the Key:**
   - Rename the key to `serviceAccountKey.json` and place it in the project root:
     `AI-Based-PCB-Fault-Detection/serviceAccountKey.json`
   - Or set its location in `.env`:
     `FIREBASE_SERVICE_ACCOUNT_PATH=path/to/your/serviceAccountKey.json`
4. **Update `.env` Variables:**
   - `FIREBASE_PROJECT_ID=your_team_project_id`
   - `FIREBASE_STORAGE_BUCKET=your_team_project_id.appspot.com`
   - `FIREBASE_MOCK_FALLBACK=false`

> **CRITICAL SECURITY RULE:**  
> Never commit `serviceAccountKey.json`, `.env`, or any private API keys to Git. These files are already registered in `.gitignore` to prevent accidental credential leaks.

---

## 10. Start the Streamlit Application

Ensure your virtual environment is active `(venv)` and run:

```powershell
streamlit run app/app.py
```

Streamlit will compile and launch the local web server:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open **`http://localhost:8501`** in Google Chrome or Microsoft Edge.

---

## 11. Verification Checklist

Follow this end-to-end walkthrough to verify that every component is working properly:

### Step 1: Check Sidebar Status
- **RAG Knowledge Base:** Verify that the sidebar displays:
  - 📚 **Indexed Docs:** `11`
  - 🧩 **Vector Chunks:** `685`
  - 📐 **Embeddings:** `FAISS`
- **Backend Persistence:** Verify status displays `🔥 Firebase: Offline (Local Fallback)` (or `Connected (Cloud)` if in Cloud Mode).

### Step 2: PCB Upload
- In the sidebar under **Quick Test Samples**, select:
  `DeepPCB Sample: 20085000 (Open Track Defect)`
- Verify that both the **Test PCB Image** and the **Golden Reference PCB Image** appear side-by-side in Section 1.

### Step 3: Run Inspection & Preprocessing
- Click the large blue button: **`🚀 Start PCB Inspection & Diagnostic Analysis`**.
- Observe the 6-stage execution status:
  - Step 1: Session initialization
  - Step 2: Image persistence
  - Step 3: Computer Vision pipeline (CLAHE enhancement, ORB alignment, morphological filtering)
  - Step 4: CV detection completion
  - Step 5: RAG vectorstore retrieval & Gemini diagnostic reasoning
  - Step 6: Diagnostic report aggregation

### Step 4: Verify Broken-Track Detection
Under **Section A: Visual Inspection Overlays**:
- Image 1: **Enhanced Preprocessed Image** (high-contrast CLAHE grayscale).
- Image 2: **Computer Vision Defect Overlay** (shows sharp red bounding boxes marking track discontinuities).
- Image 3: **Differential Heatmap** (highlights structural pixel divergence between the test and golden board).
- Under **Section B: Detected Faults**, verify defect details:
  - Fault type: `Broken Track` or `Open Defect`
  - Bounding box coordinates `[x, y, w, h]`
  - Confidence score (> 90%)

### Step 5: Verify RAG Knowledge Retrieval
Under **Section C: AI Failure Diagnostics & Repair Guidance**:
- Check **Supporting Technical Sources**: The report should cite specific documents from the knowledge base (e.g., `IPC-7721 Procedure 4.2.3 Jumper Wire Reference`, `AN-2020 Thermal Design By Insight`, or `IPC-A-610`).

### Step 6: Verify Gemini Diagnostic Reasoning
Confirm that Gemini generated grounded engineering reasoning:
- 🔍 **Probable Root Causes:** e.g., thermal trace blowout, chemical over-etching, mechanical flexing, ESD overcurrent.
- ⚡ **Circuit Impact:** e.g., open circuit, floating input, high-impedance loop, rail discontinuity.
- 🛠️ **Recommended Action:** Step-by-step IPC-7721 repair instructions (e.g., scraping solder mask with 1.5mm clearance, using 30 AWG Kynar jumper wire, staking with UV epoxy).

### Step 7: Verify Report Generation & Export
- Check the **Overall Board Verdict** banner (displays `❌ DEFECTIVE` or `⚠️ CRITICAL FAULT`).
- Click **`📄 Download Diagnostic Report (JSON)`**.
- Open the downloaded `PCB_Diagnostic_Report_*.json` and confirm it contains the full defect payload, RAG sources, and Gemini recommendations.
- Switch to the **`📋 Inspection History`** tab and verify your analysis session is logged with timestamp, analysis ID, and verdict.

---

## 12. Troubleshooting Common Issues

### Issue 1: PowerShell Script Execution Error
- **Symptom:** `File ...\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Fix:** In PowerShell run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .\venv\Scripts\Activate.ps1
  ```

---

### Issue 2: Streamlit Port Conflict (`Port 8501 is already in use`)
- **Symptom:** Error stating port 8501 is already occupied by another process.
- **Fix:** Start Streamlit on a different port:
  ```powershell
  streamlit run app/app.py --server.port 8502
  ```

---

### Issue 3: Missing `data/` or FAISS Vectorstore Error
- **Symptom:** `FileNotFoundError: ... data/vectorstore/faiss_index.bin not found` or 0 indexed chunks in sidebar.
- **Cause:** `PCB_PROJECT_DATA_v1.zip` was not extracted or was extracted into a nested subfolder like `data/data` or `TEAM_DATA_PACKAGE/data`.
- **Fix:** Verify file locations with:
  ```powershell
  Test-Path data\vectorstore\faiss_index.bin
  ```
  If this returns `False`, re-extract `PCB_PROJECT_DATA_v1.zip` directly into the project root so `data\` sits alongside `app\` and `src\`.

---

### Issue 4: Gemini API Key Errors (`403 Forbidden` / `API_KEY_INVALID`)
- **Symptom:** `Error during API call: API_KEY_INVALID` or `403 Request had invalid authentication credentials`.
- **Fix:**
  1. Open `.env` and verify there are no accidental spaces or quotation marks around the key:
     ```env
     GEMINI_API_KEY=AIzaSy...
     ```
  2. Test directly using the test script:
     ```powershell
     python tests/test_gemini_connection.py
     ```
  3. Ensure your Google AI Studio project has enabled the Gemini API and that your network does not block `generativelanguage.googleapis.com`.

---

### Issue 5: OpenCV Missing Windows Media Foundation / DLLs
- **Symptom:** `ImportError: DLL load failed while importing cv2`
- **Fix:** Ensure the Windows Media Feature Pack or latest Visual C++ Redistributable is installed on your Windows machine:
  - Download [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).
  - Run installer and reboot terminal.

---

### Issue 6: Run Full Automated Test Suite
To confirm all unit tests and subsystem integrations pass on your machine, run:
```powershell
python -m unittest discover tests -v
```
All tests should pass or gracefully fallback to local mode.

---

## 13. Team Contacts & Support

- **Project Lead:** Saurabh / Gargi
- **Repository:** [AI-Based PCB Fault Detection on GitHub](https://github.com/gargi-11/AI-Based-PCB-Fault-Detection)
- **Documentation:** Check the [`docs/`](docs/) directory for detailed system architecture, RAG specifications, and computer vision algorithm pipelines.
