"""
ScanEnhancementPipeline — orchestrates the four agents.
Can be used standalone (without Streamlit) for batch processing.
"""

from agents import QualityDetectionAgent, EnhancementAgent, EvaluationAgent, DecisionAgent
import numpy as np


class ScanEnhancementPipeline:
    def __init__(self, max_retries=3, ssim_target=0.05,
                 noise_thresh=0.5, blur_thresh=0.5, contrast_thresh=0.4):
        self.max_retries = max_retries
        self.ssim_target = ssim_target
        self.noise_thresh = noise_thresh
        self.blur_thresh = blur_thresh
        self.contrast_thresh = contrast_thresh

    def run(self, img: np.ndarray) -> dict:
        """
        Full pipeline: detect → enhance → evaluate → decide (loop).
        Returns dict with best image, metrics, attempt history.
        """
        det_agent = QualityDetectionAgent(
            noise_thresh=self.noise_thresh,
            blur_thresh=self.blur_thresh,
            contrast_thresh=self.contrast_thresh,
        )
        issues = det_agent.detect(img)
        scores = det_agent.scores

        best_image = img.copy()
        best_metrics = None
        history = []

        for attempt in range(self.max_retries):
            enh_agent  = EnhancementAgent()
            eval_agent = EvaluationAgent()
            dec_agent  = DecisionAgent(ssim_target=self.ssim_target)

            method, enhanced = enh_agent.enhance(
                img if attempt == 0 else best_image,
                issues, attempt=attempt
            )
            metrics = eval_agent.evaluate(img, enhanced)
            accepted, reason = dec_agent.decide(metrics, issues)

            history.append({
                "attempt": attempt + 1,
                "method":  method,
                "metrics": metrics,
                "accepted": accepted,
                "reason": reason,
            })

            if accepted or attempt == self.max_retries - 1:
                best_image = enhanced
                best_metrics = metrics
                break
            else:
                best_image = enhanced
                best_metrics = metrics

        return {
            "issues":      issues,
            "scores":      scores,
            "best_image":  best_image,
            "best_metrics": best_metrics,
            "history":     history,
            "verdict":     "accepted" if best_metrics and best_metrics["ssim_gain"] >= self.ssim_target
                           else "best_available",
        }
