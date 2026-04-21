"""
Four autonomous agents for the Smart Scan Enhancer pipeline.
No training required — all OpenCV / scikit-image based.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
from scipy.signal import wiener
from typing import Tuple, List, Dict


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Quality Detection Agent
# Diagnoses whether the image has: noise · blur · low contrast
# ═══════════════════════════════════════════════════════════════════════════════

class QualityDetectionAgent:
    """
    Analyses a radiology image and returns a list of detected quality issues.
    Uses three independent metrics:
      - Noise: high-frequency variance relative to image signal
      - Blur:  Laplacian variance (low = blurry)
      - Contrast: dynamic range utilisation
    """

    def __init__(self, noise_thresh=0.5, blur_thresh=0.5, contrast_thresh=0.4):
        self.noise_thresh = noise_thresh
        self.blur_thresh = blur_thresh
        self.contrast_thresh = contrast_thresh
        self.scores: Dict[str, float] = {}

    def _noise_score(self, img: np.ndarray) -> float:
        """
        Estimate noise via residual of median filter subtraction.
        Returns value in [0, 1] where 1 = very noisy.
        """
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        smooth = cv2.medianBlur(img.astype(np.uint8), 5).astype(np.float32)
        residual = np.abs(img - smooth)
        score = residual.std() / (img.std() + 1e-8)
        return float(np.clip(score, 0, 1))

    def _blur_score(self, img: np.ndarray) -> float:
        """
        Estimate blur via normalised Laplacian variance.
        Returns value in [0, 1] where 1 = very blurry.
        """
        gray = img.astype(np.uint8)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalise: typical sharp medical image ~200-2000, blurry < 50
        normalised = 1.0 - np.clip(lap_var / 500.0, 0, 1)
        return float(normalised)

    def _contrast_score(self, img: np.ndarray) -> float:
        """
        Measure contrast as used dynamic range fraction.
        Returns value in [0, 1] where 1 = very low contrast (bad).
        """
        p2, p98 = np.percentile(img, [2, 98])
        dynamic_range = p98 - p2
        max_range = 255.0
        utilisation = dynamic_range / max_range
        # Low contrast = utilisation < threshold
        return float(1.0 - np.clip(utilisation, 0, 1))

    def detect(self, img: np.ndarray) -> List[str]:
        """
        Run all detectors and return list of issue strings.
        Also populates self.scores for UI display.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        noise    = self._noise_score(gray)
        blur     = self._blur_score(gray)
        contrast = self._contrast_score(gray)

        self.scores = {
            "noise":    round(noise,    4),
            "blur":     round(blur,     4),
            "contrast": round(contrast, 4),
        }

        issues = []
        if noise    > self.noise_thresh:    issues.append("noise")
        if blur     > self.blur_thresh:     issues.append("blur")
        if contrast > self.contrast_thresh: issues.append("low_contrast")
        return issues


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Enhancement Agent
# Selects and applies the most appropriate enhancement method
# based on detected issues and the current attempt number.
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancementAgent:
    """
    Applies one of 8 enhancement methods.  Method selection is issue-driven:
      - noise      → NL-Means → Bilateral → Wavelet → Gaussian
      - blur       → Unsharp Mask → Wiener → CLAHE
      - contrast   → CLAHE → Gamma Correction
      - multiple   → combined pipeline
    """

    # Maps issues to ordered method sequences (used with attempt index)
    METHOD_MAP = {
        "noise":        ["nl_means",      "bilateral",   "wavelet",   "gaussian_denoise"],
        "blur":         ["unsharp_mask",  "wiener",      "clahe"],
        "low_contrast": ["clahe",         "gamma_correct"],
        "multi":        ["clahe+denoise", "unsharp_mask+clahe", "bilateral+clahe"],
        "none":         ["clahe",         "unsharp_mask"],
    }

    # ── individual methods ───────────────────────────────────────────────────

    def _clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return clahe.apply(img)

    def _unsharp_mask(self, img, sigma=1.5, strength=1.2):
        blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
        sharp = img.astype(np.float32) + strength * (img.astype(np.float32) - blur)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _bilateral(self, img):
        if len(img.shape) == 3:
            return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def _gaussian_denoise(self, img, ksize=5):
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def _nl_means(self, img):
        if len(img.shape) == 3:
            sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
            return (denoise_nl_means(
                img.astype(np.float32) / 255.0,
                h=1.15 * sigma_est, fast_mode=True,
                patch_size=7, patch_distance=11, channel_axis=-1
            ) * 255).astype(np.uint8)
        else:
            sigma_est = estimate_sigma(img)
            return (denoise_nl_means(
                img.astype(np.float32) / 255.0,
                h=1.15 * sigma_est, fast_mode=True,
                patch_size=7, patch_distance=11, channel_axis=None
            ) * 255).astype(np.uint8)

    def _wiener(self, img):
        f = img.astype(np.float32) / 255.0
        denoised = wiener(f, mysize=5)
        return np.clip(denoised * 255, 0, 255).astype(np.uint8)

    def _gamma_correct(self, img, gamma=1.4):
        table = (np.arange(256) / 255.0) ** (1.0 / gamma) * 255
        table = table.astype(np.uint8)
        return cv2.LUT(img, table)

    def _wavelet_denoise(self, img):
        if len(img.shape) == 3:
            return (denoise_wavelet(
                img.astype(np.float32) / 255.0,
                method="BayesShrink", mode="soft",
                wavelet_levels=3, channel_axis=-1, rescale_sigma=True
            ) * 255).astype(np.uint8)
        else:
            return (denoise_wavelet(
                img.astype(np.float32) / 255.0,
                method="BayesShrink", mode="soft",
                wavelet_levels=3, channel_axis=None, rescale_sigma=True
            ) * 255).astype(np.uint8)

    # ── method dispatcher ────────────────────────────────────────────────────

    def _apply_single(self, name, img):
        dispatch = {
            "clahe":           self._clahe,
            "unsharp_mask":    self._unsharp_mask,
            "bilateral":       self._bilateral,
            "gaussian_denoise":self._gaussian_denoise,
            "nl_means":        self._nl_means,
            "wiener":          self._wiener,
            "gamma_correct":   self._gamma_correct,
            "wavelet":         self._wavelet_denoise,
        }
        fn = dispatch.get(name)
        if fn is None:
            return img
        return fn(img)

    def _apply_combo(self, combo, img):
        result = img.copy()
        for part in combo.split("+"):
            result = self._apply_single(part.strip(), result)
        return result

    def enhance(self, img: np.ndarray, issues: List[str], attempt: int = 0
                ) -> Tuple[str, np.ndarray]:
        """
        Select and apply the best method given current issues and attempt number.
        Returns (method_display_name, enhanced_image).
        """
        if len(issues) > 1:
            seq = self.METHOD_MAP["multi"]
        elif len(issues) == 1:
            seq = self.METHOD_MAP[issues[0]]
        else:
            seq = self.METHOD_MAP["none"]

        method = seq[attempt % len(seq)]
        display_name = method.replace("_", " ").replace("+", " + ").title()

        enhanced = self._apply_combo(method, img.copy())
        return display_name, enhanced


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Evaluation Agent
# Measures SSIM, PSNR, and sharpness before vs after enhancement
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationAgent:
    """
    Computes quality metrics comparing original and enhanced images.
    Since we don't have a clean reference, we use:
      - SSIM: structural similarity (how much structure was preserved/improved)
      - PSNR: peak SNR (signal quality)
      - Sharpness: Laplacian variance ratio
      - Edge density: Canny edge count ratio
    """

    @staticmethod
    def _to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def evaluate(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        orig_g = self._to_gray(original)
        enh_g  = self._to_gray(enhanced)

        # SSIM: compare enhanced against a lightly smoothed version of original
        # (smoother = idealized reference, so SSIM improvement = real gain)
        smooth_ref = cv2.GaussianBlur(orig_g, (3, 3), 0)
        ssim_orig = ssim(smooth_ref, orig_g, data_range=255)
        ssim_enh  = ssim(smooth_ref, enh_g,  data_range=255)
        ssim_gain = ssim_enh - ssim_orig

        # PSNR: compare to smooth reference
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                psnr_orig = psnr(smooth_ref, orig_g, data_range=255)
                psnr_enh  = psnr(smooth_ref, enh_g,  data_range=255)
            psnr_val  = float(psnr_enh) if not np.isinf(psnr_enh) else 0.0
            psnr_base = float(psnr_orig) if not np.isinf(psnr_orig) else 0.0
            psnr_gain = psnr_val - psnr_base
        except Exception:
            psnr_val  = 0.0
            psnr_gain = 0.0

        # Sharpness (Laplacian variance)
        sharp_orig = cv2.Laplacian(orig_g, cv2.CV_64F).var()
        sharp_enh  = cv2.Laplacian(enh_g,  cv2.CV_64F).var()
        sharpness_gain = sharp_enh - sharp_orig

        # Contrast (dynamic range)
        p2o, p98o = np.percentile(orig_g, [2, 98])
        p2e, p98e = np.percentile(enh_g,  [2, 98])
        contrast_orig = p98o - p2o
        contrast_enh  = p98e - p2e
        contrast_gain = contrast_enh - contrast_orig

        # Edge density
        edges_orig = cv2.Canny(orig_g, 50, 150).sum() / (orig_g.size + 1e-8)
        edges_enh  = cv2.Canny(enh_g,  50, 150).sum() / (enh_g.size  + 1e-8)
        edge_gain  = edges_enh - edges_orig

        return {
            "ssim_orig":      round(float(ssim_orig),     5),
            "ssim_enhanced":  round(float(ssim_enh),      5),
            "ssim_gain":      round(float(ssim_gain),     5),
            "psnr_enhanced":  round(float(psnr_val),      3),
            "psnr_gain":      round(float(psnr_gain),     3),
            "sharpness_orig": round(float(sharp_orig),    2),
            "sharpness_enh":  round(float(sharp_enh),     2),
            "sharpness_gain": round(float(sharpness_gain),2),
            "contrast_orig":  round(float(contrast_orig), 2),
            "contrast_enh":   round(float(contrast_enh),  2),
            "contrast_gain":  round(float(contrast_gain), 2),
            "edge_gain":      round(float(edge_gain),     6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Decision Agent
# Accepts the enhanced image or triggers a retry
# ═══════════════════════════════════════════════════════════════════════════════

class DecisionAgent:
    """
    Multi-criteria acceptance decision.
    Accepts if SSIM gain >= target AND no regression in sharpness.
    """

    def __init__(self, ssim_target=0.05):
        self.ssim_target = ssim_target

    def decide(self, metrics: Dict, issues: List[str]) -> Tuple[bool, str]:
        """
        Composite acceptance: weighted score across SSIM, sharpness, and contrast.
        SSIM alone is unreliable when no clean reference is available.
        """
        sharpness_gain = metrics.get("sharpness_gain", 0)
        contrast_gain  = metrics.get("contrast_gain", 0)
        ssim_gain      = metrics.get("ssim_gain", 0)

        reasons = []
        score = 0.0

        # Sharpness improvement (primary for blur/noise)
        if sharpness_gain > 0:
            score += min(sharpness_gain / 100.0, 1.0) * 0.5
            reasons.append(f"sharp +{sharpness_gain:.1f}")

        # Contrast improvement (primary for low_contrast)
        if contrast_gain > 0:
            score += min(contrast_gain / 30.0, 1.0) * 0.35
            reasons.append(f"contrast +{contrast_gain:.1f}")

        # SSIM (use absolute, not gain)
        ssim_enh = metrics.get("ssim_enhanced", 0)
        if ssim_enh > 0.70:
            score += 0.15
            reasons.append(f"SSIM {ssim_enh:.3f}")

        # Hard reject if things got dramatically worse
        if sharpness_gain < -200:
            return False, f"Sharpness degraded {sharpness_gain:.1f}"

        # Issue-specific thresholds
        threshold = 0.0
        if "blur" in issues:
            threshold = 0.25  # need meaningful sharpness gain
        elif "low_contrast" in issues:
            threshold = 0.20
        else:
            threshold = 0.10

        if score >= threshold:
            return True, " · ".join(reasons) if reasons else "quality improved"
        else:
            return False, f"composite score {score:.2f} < {threshold:.2f}"
