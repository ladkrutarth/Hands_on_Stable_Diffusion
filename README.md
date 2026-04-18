# 🎨 AI Hairstyle & Makeup Style Generator
### Controlled Image Generation with Stable Diffusion XL

> **CS 5588 – Data Science Capstone | UMKC | Spring 2026**  
> Option 1: AI-powered beauty styling via structured prompt engineering

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[![CLIP](https://img.shields.io/badge/Eval-OpenCLIP%20ViT--B%2F32-green)](https://github.com/mlfoundations/open_clip)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

---

## 📌 Overview

This project builds a **data-driven, controlled image generation pipeline** for a virtual beauty styling application. Given structured user attributes — hairstyle, hair color, makeup style, and occasion — the system generates realistic portrait images using **Stable Diffusion XL (SDXL)** and evaluates output quality via **CLIP cosine similarity scores**.

The core research question: _Does structured prompt engineering produce higher-quality, more consistent outputs than naive (bag-of-words) prompts?_

---

## 🗂 Repository Structure

```
.
├── Hands_on_Stable_Diffusion.ipynb   # Main Colab notebook (run this)
├── README.md                          # This file
├── outputs/
│   ├── structured/                    # Generated images (structured prompts)
│   ├── naive/                         # Generated images (naive prompts)
│   └── clip_scores.json               # CLIP evaluation results
├── slides/
│   └── AI_Style_Generator.pptx        # Presentation slides
└── demo/
    └── demo_video.mp4                 # Screen recording of pipeline run
```

---

## 🚀 Quick Start

### 1. Open in Google Colab
Click the badge above or go to `File → Open Notebook → GitHub` and paste this repo URL.

### 2. Install Dependencies
The first cell handles everything:
```bash
pip install torch torchvision diffusers transformers accelerate \
            safetensors pillow matplotlib tqdm open_clip_torch
```

### 3. Run All Cells
The notebook is fully self-contained. Run top-to-bottom on a **T4 or A100 GPU** (free Colab tier works with T4).

> ⚠️ **CPU Warning:** Image generation on CPU takes 10–30 min per image. Reduce `NUM_IMAGES` and `STEPS` for testing.

---

## 🏗 System Architecture

```
Structured Input (BeautyAttributes)
        │
        ├──► structured_prompt()  ──►  SDXL Pipeline  ──►  4 images (seeds 42–45)
        │                                                          │
        └──► naive_prompt()       ──►  SDXL Pipeline  ──►  4 images (seeds 42–45)
                                                                   │
                                                              show_grid()
                                                                   │
                                                         CLIP ViT-B/32 scoring
                                                                   │
                                                         Cosine Similarity Report
```

---

## 📐 Core Components

### `BeautyAttributes` — Input Schema
```python
@dataclass
class BeautyAttributes:
    gender: str          # "man" | "woman"
    hairstyle: str       # e.g. "sleek low bun", "short textured fade"
    hair_color: str      # e.g. "dark brown", "black", "auburn"
    makeup_style: str    # e.g. "soft glam with rose lipstick", "clean grooming"
    occasion: str        # 47 supported occasions (see list below)
```

### Prompt Engineering: Naive vs Structured

| Aspect | Naive Prompt | Structured Prompt |
|--------|-------------|-------------------|
| **Format** | Bag-of-words | Grammatically complete sentences |
| **Context** | None | `"professional portrait photograph of one adult {gender}"` |
| **Photography** | ❌ | ✅ `"85mm lens, shallow depth of field, soft diffused lighting"` |
| **Quality tags** | ❌ | ✅ `"high detail, realistic, editorial beauty photography"` |
| **Negative prompt** | ❌ | ✅ `"lowres, blurry, distorted eyes, extra fingers..."` |

**Example — Woman, Sleek Low Bun, Evening Gala:**

```
# Naive
woman sleek low bun dark brown hair soft glam with rose lipstick evening gala face photo

# Structured
professional portrait photograph of one adult woman, sleek low bun hairstyle,
dark brown hair, soft glam with rose lipstick, suitable for evening gala,
natural skin texture, soft diffused lighting, 85mm lens, shallow depth of field,
high detail, realistic, editorial beauty photography
```

### Generation Parameters

| Parameter | Value |
|-----------|-------|
| Model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Resolution | 1024 × 1024 |
| Inference Steps | 35 |
| Guidance Scale | 7.5 |
| Seeds | base_seed + i (default: 42–45) |
| Scheduler | SDXL default (DPM-Solver++) |

---

## 📊 Evaluation Results

### CLIP Cosine Similarity (ViT-B/32)

| Prompt Type | Scores | Mean | Std |
|------------|--------|------|-----|
| **Structured** | [0.2806, 0.2709, 0.2561, 0.2678] | **0.2688** | 0.0087 |
| **Naive** | [0.2903, 0.2799, 0.2647, 0.2882] | **0.2808** | 0.0101 |

### Key Insight

> **Naive prompts score marginally higher on CLIP** because the CLIP text encoder favors short, token-dense phrases. However, **structured prompts produce more consistent outputs** (std 0.0087 vs 0.0101), indicating higher reliability across different seeds — which is more valuable for a production styling application.

---

## 🎛 Supported Occasions (47)

Organized by context:

| Category | Examples |
|----------|---------|
| **Professional** | Job interviews, Business meetings, Conferences, Presentations |
| **Social** | Birthday parties, Dates, Clubbing/nightlife, Dinner with friends |
| **Formal** | Weddings, Award functions, Gala events, Engagement ceremonies |
| **Cultural** | Festivals (Diwali, Eid, Christmas), Religious ceremonies |
| **Casual** | Vacations, Beach outings, Working from home, Running errands |
| **Athletic** | Gym, Running/jogging, Yoga sessions, Sports matches |
| **Academic** | School/college, Graduation ceremonies, Convocations |

---

## ⚠️ Failure Cases & Limitations

| Issue | Severity | Description |
|-------|----------|-------------|
| Identity Inconsistency | 🔴 HIGH | Different seeds produce different people; no cross-run identity |
| CLIP Score Paradox | 🟡 MED | Naive prompts score higher despite lower visual quality |
| Style Leakage | 🟡 MED | Strong occasion keywords can override explicit attribute requests |
| Anatomical Errors | 🟢 LOW | Occasional hand/hair artifacts despite negative prompts |
| Demographic Bias | 🔴 HIGH | Model skews toward Western beauty norms for certain occasions |
| No Face Reference | 🟡 MED | Cannot preserve a specific identity without ControlNet/IP-Adapter |

---

## 🔮 Future Work

- **ControlNet** — Add face landmark conditioning for identity-preserving restyling
- **IP-Adapter** — Use reference images to fix identity across all seed variations
- **InstructPix2Pix** — Enable iterative style editing on existing photos
- **FID / IS Metrics** — Add Fréchet Inception Distance for distribution-level quality measurement
- **LoRA Fine-tuning** — Fine-tune SDXL on FFHQ for sharper portrait outputs
- **Demographic Diversity** — Curate culturally diverse prompt templates

---

## 📦 Datasets (Reference)

| Dataset | Size | Usage |
|---------|------|-------|
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | 202K images, 40 attributes | Attribute vocabulary design |
| [FFHQ](https://github.com/NVlabs/ffhq-dataset) | 70K at 1024×1024 | Generation quality benchmark |
| [PSGAN / Makeup Transfer](https://github.com/wtjiang98/PSGAN) | Before/after pairs | Style transfer reference |

> **Note:** Datasets are used as design reference only. SDXL generates images directly from prompts — no fine-tuning performed.

---

## 🛠 Tech Stack

```
Language:    Python 3.10+
Framework:   HuggingFace Diffusers 0.27+
Model:       stabilityai/stable-diffusion-xl-base-1.0
Evaluation:  OpenCLIP ViT-B/32 (open_clip_torch)
Hardware:    CUDA GPU (Google Colab T4/A100)
Viz:         Matplotlib + PIL
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Stability AI](https://stability.ai/) for SDXL
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) for evaluation
- [CompVis](https://github.com/CompVis/stable-diffusion) for Stable Diffusion foundations
