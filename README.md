# 🔬 Smart Scan Enhancer

AI-powered radiology image quality booster. Four autonomous agents detect and fix
noise, blur, and low contrast — no model training required.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│  Agent 1: Quality Detector  │  → detects noise / blur / low contrast
└─────────────┬───────────────┘
              │ issues list
              ▼
┌─────────────────────────────┐
│  Agent 2: Enhancement       │  → CLAHE, NL-Means, Bilateral, Unsharp,
└─────────────┬───────────────┘    Wiener, Wavelet, Gamma, Gaussian
              │ enhanced image
              ▼
┌─────────────────────────────┐
│  Agent 3: Evaluation        │  → SSIM gain, PSNR gain, Sharpness, Contrast
└─────────────┬───────────────┘
              │ metrics
              ▼
┌─────────────────────────────┐
│  Agent 4: Decision          │  → Accept or Retry (up to N attempts)
└─────────────────────────────┘
```

## Enhancement Methods

| Method          | Best For        |
|----------------|-----------------|
| CLAHE           | Low contrast    |
| NL-Means        | High noise      |
| Bilateral Filter| Edge-safe denoise|
| Wavelet Denoise | Multi-scale noise|
| Unsharp Mask    | Blur            |
| Wiener Filter   | Deblurring      |
| Gamma Correct   | Dark images     |
| Gaussian Denoise| Light smoothing |

## Supported Formats
JPEG, PNG, BMP, TIFF — grayscale or RGB
Works with X-ray, MRI, CT, Ultrasound, DICOM slices (exported as PNG/JPEG)

## Configuration (sidebar)
- Noise / Blur / Contrast sensitivity thresholds
- Max retry attempts (1–5)
- SSIM improvement target
