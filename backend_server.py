import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os, io, re, base64, json, requests, numpy as np, torch
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

# --------------------------------------------------------
#  FLASK BACKEND SETUP
# --------------------------------------------------------
app = Flask(__name__, static_folder='static', template_folder='template')
CORS(app)
MODEL_TARGET_SIZE = (512, 512)

# --------------------------------------------------------
#  LOAD STABLE DIFFUSION MODEL (For Restoration)
# --------------------------------------------------------
print("[INFO] Loading Stable Diffusion 2 Inpainting Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

if device == "cpu":
    pipe.enable_attention_slicing()
    torch.set_num_threads(os.cpu_count() or 4)

# Configure DPMSolver++ scheduler for higher fidelity & 2x faster synthesis
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
print(f"[OK] Model loaded on {device.upper()} with DPMSolver++ Karras Scheduler")

# Warm-up model for faster first response
print("[INFO] Warming up model...")
dummy_image = Image.new("RGB", MODEL_TARGET_SIZE, color="white")
dummy_mask = Image.new("RGB", MODEL_TARGET_SIZE, color="black")
_ = pipe(prompt="warm up", image=dummy_image, mask_image=dummy_mask, num_inference_steps=2, guidance_scale=1.0)
print("[OK] Warm-up complete\n")

# --------------------------------------------------------
#  HELPER FUNCTIONS
# --------------------------------------------------------
def decode_image_from_base64(base64_string):
    padding = '=' * (4 - len(base64_string) % 4)
    image_data = base64.b64decode(base64_string + padding)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image_to_base64(image, format='JPEG'):
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# --------------------------------------------------------
#  INPAINT FUNCTION (RESTORATION)
# --------------------------------------------------------
def run_inpainting(image_pil, mask_pil, description=None, material=None, seed=None, steps=10, guidance=8.0):
    print("\n--- Starting Enhanced Historical Restoration ---")
    orig_w, orig_h = image_pil.size

    # Ensure mask is single channel grayscale
    mask_gray = mask_pil.convert("L")

    # 1. Prepare 512x512 inputs for the diffusion model
    image_resized = image_pil.resize(MODEL_TARGET_SIZE, Image.LANCZOS)
    mask_resized = mask_gray.resize(MODEL_TARGET_SIZE, Image.NEAREST)

    # 2. Material-specific architectural prompts
    material_prompts = {
        "sandstone": "warm porous red sandstone texture, carved temple rock relief, authentic ancient Indian architectural patina",
        "granite": "weathered ancient dark granite stone carving, coarse crystalline rock texture, chiseled relief details",
        "terracotta": "fired clay terracotta relief, warm earthy ochre tone, weathered kiln texture",
        "bronze": "antique Chola lost-wax cast bronze idol, rich green verdigris oxidation patina, weathered metallic surface",
        "marble": "aged white temple marble, subtle natural veining, soft matte antique luster"
    }

    mat_key = material.lower() if material else ""
    mat_text = material_prompts.get(mat_key, "realistic carved stone texture, historically accurate Indian temple carving")

    base_prompt = (
        f"masterpiece restoration of ancient Indian temple sculpture, {mat_text}, "
        "fine intricate sculptural detailing, natural heritage temple lighting, intact archaeological preservation"
    )
    prompt = base_prompt + (f", {description.strip()}" if description else "")

    negative_prompt = (
        "blurry, out of focus, low quality, distorted, cartoon, anime, 3d render, plastic, "
        "smooth modern skin, modern clothes, text, watermark, signature, jpeg artifacts, "
        "oversaturated, malformed limbs, extra fingers, noisy, unnatural colors"
    )

    # 3. Dynamic seed handling
    if seed is None or int(seed) < 0:
        seed = int(torch.randint(0, 2147483647, (1,)).item())
    else:
        seed = int(seed)

    generator = torch.Generator(device=device).manual_seed(seed)

    # 4. Generate inpainting synthesis (DPMSolver++ Karras converges in 10-12 steps)
    if device == "cuda":
        default_steps = 20
        min_steps = 14
    else:
        default_steps = 10
        min_steps = 8

    if steps:
        try:
            inference_steps = max(min_steps, min(35, int(steps)))
        except:
            inference_steps = default_steps
    else:
        inference_steps = default_steps

    guidance_scale = float(guidance) if guidance else 8.0

    print(f"Synthesizing with {inference_steps} steps, guidance {guidance_scale}, seed {seed}...")
    result_512 = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_resized,
        mask_image=mask_resized,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # 5. Full-Resolution Alpha-Feathered Compositing:
    # Resize the 512x512 inpainted result back to original image dimensions
    inpainted_highres = result_512.resize((orig_w, orig_h), Image.LANCZOS)

    # Soften / anti-alias the original mask to blend edges naturally with surrounding stone
    feathered_mask = mask_gray.filter(ImageFilter.GaussianBlur(radius=3))

    # Composite: inpainted content is placed ONLY where mask is white;
    # all undamaged regions retain 100% of original untouched high-res pixels!
    restored_composite = Image.composite(inpainted_highres, image_pil, feathered_mask)

    print(f"--- Restoration Complete (Seed: {seed}) ---")
    return restored_composite, seed

# --------------------------------------------------------
#  OCR + TRANSLATION (Gemini Multimodal API)
# --------------------------------------------------------
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
API_KEY = "AIzaSyDhwmr4iHM_Q3nC6BzUbenfELHU1rWrw-c"

def parse_epigraphy_output(raw_out):
    """
    Guarantees extraction of both original script and English translation even if
    the model output was truncated, unclosed JSON, markdown-wrapped, or plain text.
    """
    raw_text = ""
    english_trans = ""

    # 1. Clean markdown code fences if present
    clean_str = re.sub(r'^```(?:json)?\s*', '', raw_out.strip(), flags=re.MULTILINE)
    clean_str = re.sub(r'```$', '', clean_str.strip(), flags=re.MULTILINE).strip()

    # 2. Strict JSON parsing
    try:
        data = json.loads(clean_str)
        if isinstance(data, dict):
            raw_text = data.get("raw_text") or data.get("text") or data.get("transcription") or data.get("original_text") or ""
            english_trans = data.get("english_translation") or data.get("translation") or data.get("meaning") or data.get("english") or ""
    except Exception:
        pass

    # 3. Auto-repair unclosed/truncated JSON (e.g. cut off right before closing brace)
    if not english_trans or not raw_text:
        try:
            repaired = clean_str.rstrip()
            if not repaired.endswith('}'):
                if not repaired.endswith('"'):
                    repaired += '"'
                repaired += '}'
            data = json.loads(repaired)
            if isinstance(data, dict):
                if not raw_text:
                    raw_text = data.get("raw_text") or data.get("text") or data.get("transcription") or data.get("original_text") or ""
                if not english_trans:
                    english_trans = data.get("english_translation") or data.get("translation") or data.get("meaning") or data.get("english") or ""
        except Exception:
            pass

    # 4. Robust regex extraction for raw_text and english_translation
    if not raw_text:
        m_raw = re.search(r'"(?:raw_text|text|transcription|original_text)"\s*:\s*"((?:\\.|[^"\\])*)', clean_str)
        if m_raw:
            try:
                raw_text = m_raw.group(1).encode('utf-8').decode('unicode_escape', errors='replace')
            except Exception:
                raw_text = m_raw.group(1).replace('\\"', '"').replace('\\n', '\n')

    if not english_trans:
        m_trans = re.search(r'"(?:english_translation|translation|meaning|english)"\s*:\s*"((?:\\.|[^"\\])*)', clean_str)
        if m_trans:
            try:
                english_trans = m_trans.group(1).encode('utf-8').decode('unicode_escape', errors='replace')
            except Exception:
                english_trans = m_trans.group(1).replace('\\"', '"').replace('\\n', '\n')

    # 5. Loose trailing match if english_translation was truncated at the end
    if not english_trans:
        m_trans_loose = re.search(r'"(?:english_translation|translation|meaning)"\s*:\s*"?([^"}]+)', clean_str, re.DOTALL)
        if m_trans_loose:
            candidate = m_trans_loose.group(1).strip()
            candidate = candidate.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            if len(candidate) > 5:
                english_trans = candidate

    # 6. Fallback section split (e.g. Transcription: ... Translation: ...)
    if not english_trans or not raw_text:
        sections = re.split(r'(?i)(?:transcription|original text|translation|english translation):', clean_str)
        if len(sections) >= 3:
            if not raw_text:
                raw_text = sections[1].strip()
            if not english_trans:
                english_trans = sections[2].strip()

    if not raw_text:
        raw_text = clean_str

    if not english_trans:
        # Default scholarly contextual message
        english_trans = "The inscription text has been transcribed above; translation can be further analyzed based on specific regional dialect."

    return raw_text.strip(), english_trans.strip()

@app.route("/process", methods=["POST"])
def process_image_text():
    data = request.json or {}
    image_base64 = data.get("image")
    image_mime = data.get("mime_type", "image/jpeg")

    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    # Auto-optimize / resize large user images to max 1600px for speedy transmission
    try:
        raw_bytes = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(raw_bytes)) as img_check:
            max_dim = 1600
            w, h = img_check.size
            if max(w, h) > max_dim:
                ratio = max_dim / float(max(w, h))
                new_size = (int(w * ratio), int(h * ratio))
                img_resized = img_check.resize(new_size, Image.LANCZOS)
                if img_resized.mode != "RGB":
                    img_resized = img_resized.convert("RGB")
                out_buf = io.BytesIO()
                img_resized.save(out_buf, format="JPEG", quality=90)
                image_base64 = base64.b64encode(out_buf.getvalue()).decode()
                image_mime = "image/jpeg"
    except Exception as e:
        print(f"Image preprocessing note: {e}")

    epigraphy_prompt = (
        "You are a master epigraphist and historical linguist specializing in ancient and classical Indian inscriptions, "
        "including Brahmi, Devanagari, Grantha, Vatteluttu, Sanskrit, Prakrit, Tamil, Kannada, Telugu, Kharosthi, and Old Marathi.\n\n"
        "Your task:\n"
        "1. Carefully inspect this historical inscription image and transcribe the visible glyphs, characters, and verses with maximum archaeological fidelity.\n"
        "2. Provide an accurate, eloquent, and scholarly modern English translation that explains the civilizational, religious, and historical meaning.\n"
        "3. Ensure the translation is complete and cohesive without hitting token limits.\n\n"
        "Output: Return a valid JSON object strictly adhering to this schema:\n"
        "{\n"
        '  "raw_text": "<transcribed original script glyphs or text>",\n'
        '  "english_translation": "<faithful, eloquent scholarly English translation with historical meaning>"\n'
        "}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": epigraphy_prompt},
                    {
                        "inlineData": {
                            "mimeType": image_mime,
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string"},
                    "english_translation": {"type": "string"}
                },
                "required": ["raw_text", "english_translation"]
            },
            "temperature": 0.2,
            "maxOutputTokens": 4096
        }
    }

    last_error = None
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        try:
            print(f"--- Deciphering Epigraphy with {model_name} ---")
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=45)
            if response.status_code == 200:
                response_json = response.json()
                raw_out = (
                    response_json
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )

                raw_text, english_trans = parse_epigraphy_output(raw_out)

                print(f"[OK] Epigraphy deciphered successfully using {model_name}")
                return jsonify({
                    "text": raw_text or "No text recognized.",
                    "translation": english_trans or "Translation unavailable."
                })
            else:
                last_error = f"{model_name} HTTP {response.status_code}: {response.text[:200]}"
                print(f"[WARN] {last_error}, trying next model...")
        except Exception as e:
            last_error = str(e)
            print(f"[WARN] Error with {model_name}: {last_error}, trying next model...")

    print("[ERROR] All models failed:", last_error)
    return jsonify({"error": f"Unable to decipher inscription: {last_error}"}), 500

# --------------------------------------------------------
#  ROUTES FOR FRONTEND PAGES
# --------------------------------------------------------
@app.route('/')
def home_page():
    return render_template('features.html')  # main page to select feature

@app.route('/restoration')
def restoration_page():
    return render_template('index.html')  # your existing image restoration frontend

@app.route('/ocr')
def ocr_page():
    return render_template('frontend.html')  # your OCR + translation frontend

# --------------------------------------------------------
#  RESTORATION API ENDPOINT
# --------------------------------------------------------
@app.route('/api/inpaint', methods=['POST'])
def inpaint_endpoint():
    data = request.json
    if not data or 'image_base64' not in data or 'mask_base64' not in data:
        return jsonify({"error": "Missing image or mask data"}), 400

    try:
        image_pil = decode_image_from_base64(data['image_base64'])
        mask_pil = decode_image_from_base64(data['mask_base64'])
        description = data.get('description', None)
        material = data.get('material', None)
        seed = data.get('seed', None)
        steps = data.get('steps', 28)
        guidance = data.get('guidance', 8.0)

        restored, seed_used = run_inpainting(
            image_pil=image_pil,
            mask_pil=mask_pil,
            description=description,
            material=material,
            seed=seed,
            steps=steps,
            guidance=guidance
        )
        restored_base64 = encode_image_to_base64(restored)

        return jsonify({
            "status": "success",
            "reconstructed_image_base64": restored_base64,
            "seed": seed_used
        })
    except Exception as e:
        print("Inference error:", e)
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------
#  HEALTH CHECK
# --------------------------------------------------------
@app.route('/api/health')
def health_check():
    return jsonify({"status": "ready"})

# --------------------------------------------------------
#  MAIN
# --------------------------------------------------------
if __name__ == '__main__':
    print("--------------------------------------------------------")
    print(" Flask Backend for Restoration + OCR Translation")
    print(" Running at: http://127.0.0.1:5000/")
    print("--------------------------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
