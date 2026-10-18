# ??? Punara Vistara (???? ???????)
### AI-Powered Historical Artifact Inpainting & Ancient Epigraphy Decipherment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E.svg)](https://huggingface.co/docs/diffusers/)
[![Model](https://img.shields.io/badge/Model-Stable--Diffusion--2--Inpainting-9cf.svg)](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
[![Gemini](https://img.shields.io/badge/Vision_LLM-Google_Gemini-4285F4.svg)](https://ai.google.dev/)

---

## ?? Overview

**Punara Vistara** is an advanced artificial intelligence platform dedicated to the preservation, non-destructive restoration, and epigraphical analysis of ancient cultural heritage. Designed for archaeologists, epigraphists, museum conservators, and heritage enthusiasts, Punara Vistara provides:

1. **AI Historical Inpainting & Reconstruction**: Synthetically regenerates broken, chipped, weathered, or desecrated temple sculptures, stone reliefs, and bronzes using a fine-tuned Stable Diffusion 2 Inpainting diffusion pipeline with material-conditioned prompting and alpha-feathered compositing.
2. **Epigraphy OCR & Scholarly Decipherment**: Analyzes ancient lithic inscriptions and copper-plate grants across multiple historic Indic scripts (Brahmi, Grantha, Vatteluttu, Devanagari, Old Kannada, Kharosthi, etc.) using Google Gemini Vision LLMs with automatic script transcription, archaeological analysis, and modern English translation.

---

## ??? System Architecture

\\\mermaid
graph TD
    subgraph ClientLayer ["Client Interface (HTML5 / Modern CSS / Vanilla JS)"]
        UI_Hub["Navigation Hub\n(/)"]
        UI_Restore["Sculpture Restoration Studio\n(/restoration)"]
        UI_OCR["Epigraphy Decipherment Studio\n(/ocr)"]
    end

    subgraph ServerLayer ["Flask Application Server (backend_server.py)"]
        Router["Flask Request Dispatcher & CORS"]
        EP_Inpaint["/api/inpaint\n(Restoration Engine)"]
        EP_OCR["/process\n(Epigraphy Decipherment)"]
        EP_Health["/api/health\n(Status Monitor)"]
    end

    subgraph RestorationEngine ["Historical Inpainting Pipeline"]
        Preprocess["B64 Decoder & Mask Normalization (512x512)"]
        PromptEngine["Material & Heritage Context Prompt Generator"]
        SD_Model["Stable Diffusion 2 Inpainting\n(stabilityai/stable-diffusion-2-inpainting)"]
        Scheduler["DPMSolver++ Multistep Scheduler\n(use_karras_sigmas=True)"]
        Postprocess["Alpha-Feathered Compositing\n(High-Res Non-Destructive Blend)"]
    end

    subgraph EpigraphyEngine ["Multimodal Epigraphy Analysis Pipeline"]
        ImageOptimizer["Image Dimension Optimizer (Max 1600px)"]
        FallbackCascade["Gemini Model Fallback Cascade"]
        Gemini38["Gemini 3.8 Flash"]
        Gemini20["Gemini 2.0 Flash"]
        Gemini15["Gemini 1.5 Flash"]
        Parser["Epigraphy Robust Parser\n(JSON Repair & Regex Auto-Recovery)"]
    end

    UI_Hub --> Router
    UI_Restore --> EP_Inpaint
    UI_OCR --> EP_OCR

    Router --> EP_Inpaint
    Router --> EP_OCR
    Router --> EP_Health

    EP_Inpaint --> Preprocess
    Preprocess --> PromptEngine
    PromptEngine --> SD_Model
    Scheduler --> SD_Model
    SD_Model --> Postprocess
    Postprocess --> EP_Inpaint

    EP_OCR --> ImageOptimizer
    ImageOptimizer --> FallbackCascade
    FallbackCascade --> Gemini38
    Gemini38 -.->|On Fail| Gemini20
    Gemini20 -.->|On Fail| Gemini15
    FallbackCascade --> Parser
    Parser --> EP_OCR
\\\

---

## ?? Restoration & Inpainting Pipeline

The inpainting pipeline ensures that only the damaged areas indicated by the user mask are regenerated, while 100% of the untouched original archaeological stone texture and surrounding features are preserved at original camera resolution.

\\\mermaid
sequenceDiagram
    autonumber
    actor User as Conservator / User
    participant Frontend as Restoration UI (Canvas)
    participant Backend as Flask Server (/api/inpaint)
    participant Model as SD2 Inpainting + DPMSolver++
    participant Blender as Alpha-Feathering Compositor

    User->>Frontend: Upload artifact photo & paint damage mask
    User->>Frontend: Select material (Sandstone, Granite, Terracotta, Bronze, Marble)
    Frontend->>Backend: POST base64 original image + binary mask + prompt config
    Backend->>Backend: Downscale inputs to 512x512 for latent diffusion
    Backend->>Model: Execute inpainting synthesis (DPMSolver++ Karras, 10-25 steps)
    Model-->>Backend: 512x512 reconstructed patch synthesis
    Backend->>Blender: Upscale synthesis to original resolution (Lanczos)
    Backend->>Blender: Apply Gaussian blur feathering to original mask
    Backend->>Blender: Image.composite(inpainted, original, feathered_mask)
    Blender-->>Backend: High-fidelity restored composite
    Backend-->>Frontend: Return reconstructed image + reproduction seed
    Frontend-->>User: Display interactive before/after side-by-side comparison
\\\

---

## ?? Epigraphy & Inscription Decipherment Workflow

\\\mermaid
flowchart LR
    A["Raw Inscription Photo\n(Rock Edict / Pillar / Copper Plate)"] --> B["Image Preprocessor\n(Normalizes bounds to max 1600px)"]
    B --> C["Epigraphy Structured Prompt\n(Paleography & Historical Linguistic Expert)"]
    C --> D{"Gemini Vision API\nFallback Cascade"}
    D -->|Attempt 1| M1["gemini-3.8-flash"]
    D -->|Fallback 1| M2["gemini-2.0-flash"]
    D -->|Fallback 2| M3["gemini-1.5-flash"]
    M1 & M2 & M3 --> E["Robust JSON & Regex Parser\n(Auto-repairs truncated responses)"]
    E --> F["Transcribed Original Script\n(Brahmi, Grantha, Old Kannada, etc.)"]
    E --> G["Scholarly English Translation\n(Historical & Cultural Interpretation)"]
\\\

---

## ? Key Features

- **Material-Conditioned Prompting**:
  - *Sandstone*: Porous red sandstone texture with authentic ancient temple rock relief patina.
  - *Granite*: Weathered crystalline rock texture with chiseled heritage relief details.
  - *Terracotta*: Fired clay ochre tones with kiln weathering.
  - *Bronze*: Antique Chola lost-wax cast idol with natural green verdigris oxidation patina.
  - *Marble*: Natural veined temple marble with soft antique matte luster.
- **Dual-Resolution Alpha Feathering**: Avoids full-image recompression or blurriness by isolating the diffusion synthesis to the mask and seamlessly blending it using Gaussian feathering.
- **Multi-Script Epigraphy Recognition**: Handles ancient, medieval, and classical scripts including Brahmi, Grantha, Vatteluttu, Devanagari, Old Kannada, Kadamba, Telugu, Kharosthi, and Old Marathi.
- **Fail-Safe Decipherment Cascade**: Multi-model fallback across Gemini 3.8 Flash, 2.0 Flash, and 1.5 Flash ensures high availability.
- **Interactive Web Studio**: Integrated canvas masking tool with brush controls, side-by-side comparison sliders, and prompt adjustments.

---

## ?? Project Structure

\\\
Punara_Vistara/
+-- backend_server.py      # Main Flask server with SD2 inpainting & Gemini Epigraphy pipeline
+-- requirements.txt       # Python dependencies (Torch, Diffusers, Transformers, Flask)
+-- .gitignore             # Git exclusion rules (venv, cache, temporary files)
+-- README.md              # Project documentation and architectural blueprints
+-- static/
¦   +-- ocr.css            # Styles for epigraphy decipherment interface
¦   +-- ocr.js             # Client logic for epigraphy camera/file upload & API calls
¦   +-- theme.css          # Core design system, variables, and typography
¦   +-- runtime.txt        # Runtime specifications
+-- template/
    +-- features.html      # Landing page / module selector
    +-- frontend.html      # Ancient Epigraphy OCR & Decipherment Studio
    +-- index.html         # AI Historical Sculpture Restoration Studio
\\\

---

## ?? Getting Started

### Prerequisites
- Python 3.10+ installed
- CUDA-compatible GPU (recommended for fast inpainting) or CPU (supported with attention slicing)

### 1. Clone the Repository
\\\ash
git clone https://github.com/Akshitha-Veepanjeri-G/Punara_Vistara.git
cd Punara_Vistara
\\\

### 2. Create and Activate Virtual Environment
\\\ash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
\\\

### 3. Install Dependencies
\\\ash
pip install -r requirements.txt
\\\

### 4. Run the Application
\\\ash
python backend_server.py
\\\

Once the server initializes:
- Access the **Navigation Portal**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- Access the **Restoration Studio**: [http://127.0.0.1:5000/restoration](http://127.0.0.1:5000/restoration)
- Access the **Epigraphy Decipherment Studio**: [http://127.0.0.1:5000/ocr](http://127.0.0.1:5000/ocr)

---

## ?? API Reference

### 1. Historical Inpainting
- **Endpoint**: \POST /api/inpaint\
- **Content-Type**: \pplication/json\
- **Request Body**:
\\\json
{
  "image_base64": "<base64_encoded_original_image>",
  "mask_base64": "<base64_encoded_binary_mask>",
  "material": "sandstone",
  "description": "missing forearm holding a trishula",
  "steps": 20,
  "guidance": 8.0,
  "seed": 42
}
\\\
- **Response**:
\\\json
{
  "status": "success",
  "reconstructed_image_base64": "<base64_encoded_composite>",
  "seed": 42
}
\\\

### 2. Epigraphy OCR & Translation
- **Endpoint**: \POST /process\
- **Content-Type**: \pplication/json\
- **Request Body**:
\\\json
{
  "image": "<base64_encoded_inscription_image>",
  "mime_type": "image/jpeg"
}
\\\
- **Response**:
\\\json
{
  "text": "Transcribed original script glyphs",
  "translation": "Scholarly modern English translation with historical context"
}
\\\

### 3. Health Check
- **Endpoint**: \GET /api/health\
- **Response**:
\\\json
{
  "status": "ready"
}
\\\

---

## ?? License
This project is open-source under the MIT License. Developed for cultural heritage preservation.
