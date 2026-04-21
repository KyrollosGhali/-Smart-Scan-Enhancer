import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import json
from agents import QualityDetectionAgent, EnhancementAgent, EvaluationAgent, DecisionAgent
from pipeline import ScanEnhancementPipeline

st.set_page_config(
    page_title="Smart Scan Enhancer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0a0e1a; color: #e2e8f0; }
[data-testid="stSidebar"] { background-color: #0f1424; border-right: 1px solid #1e2740; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #7dd3fc !important; font-family: 'Space Mono', monospace; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; }
.header-box { background: #0f1a35; border: 1px solid #1e3a6e; border-top: 2px solid #3b82f6; border-radius: 12px; padding: 28px 36px; margin-bottom: 24px; }
.header-title { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #f0f9ff; margin: 0; }
.header-sub { color: #64748b; font-size: 0.88rem; margin-top: 6px; }
.header-badge { display:inline-block; background:#0c2342; border:1px solid #1e4d7a; color:#7dd3fc; font-size:0.68rem; font-family:'Space Mono',monospace; padding:3px 10px; border-radius:4px; margin-top:10px; }
.metric-card { background:#0f1628; border:1px solid #1e2d4a; border-radius:10px; padding:16px 18px; margin-bottom:10px; }
.metric-lbl { font-size:0.68rem; font-family:'Space Mono',monospace; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:5px; }
.metric-val { font-size:1.4rem; font-weight:600; color:#f0f9ff; }
.metric-sub { font-size:0.73rem; color:#64748b; margin-top:2px; }
.bar-bg { background:#1e2740; border-radius:4px; height:5px; margin-top:6px; overflow:hidden; }
.bar-fill { height:100%; border-radius:4px; }
.agent-card { background:#0f1628; border:1px solid #1e2d4a; border-radius:10px; padding:12px 16px; margin-bottom:8px; }
.agent-card.active { border-color:#3b82f6; background:#0f1f3d; }
.agent-card.done   { border-color:#10b981; background:#0a1f17; }
.agent-card.retry  { border-color:#f59e0b; background:#1a150a; }
.ag-name { font-family:'Space Mono',monospace; font-size:0.72rem; font-weight:700; letter-spacing:0.05em; color:#94a3b8; text-transform:uppercase; }
.ag-status { font-size:0.8rem; color:#64748b; margin-top:3px; }
.agent-card.active .ag-name { color:#7dd3fc; }
.agent-card.done   .ag-name { color:#34d399; }
.agent-card.retry  .ag-name { color:#fbbf24; }
.issue-tag { display:inline-block; padding:3px 9px; border-radius:20px; font-size:0.7rem; font-family:'Space Mono',monospace; margin:2px; text-transform:uppercase; }
.t-noise    { background:#2d1b1b; border:1px solid #7f1d1d; color:#fca5a5; }
.t-blur     { background:#1b2040; border:1px solid #1e3a8a; color:#93c5fd; }
.t-contrast { background:#1b2d1b; border:1px solid #14532d; color:#86efac; }
.t-ok       { background:#1b2d2d; border:1px solid #164e63; color:#67e8f9; }
.method-pill { display:inline-block; background:#1e3a5f; border:1px solid #2563eb; color:#93c5fd; font-size:0.68rem; font-family:'Space Mono',monospace; padding:3px 10px; border-radius:20px; margin:2px; }
.verdict-ok  { background:#052e16; border:1px solid #166534; border-radius:8px; padding:12px 16px; color:#4ade80; font-family:'Space Mono',monospace; font-size:0.78rem; }
.verdict-warn{ background:#1c1003; border:1px solid #92400e; border-radius:8px; padding:12px 16px; color:#fbbf24; font-family:'Space Mono',monospace; font-size:0.78rem; }
.img-cap { text-align:center; font-size:0.7rem; font-family:'Space Mono',monospace; color:#475569; text-transform:uppercase; letter-spacing:0.06em; margin-top:6px; }
.attempt-row { background:#0f1628; border:1px solid #1e2d4a; border-radius:8px; padding:12px 16px; margin-bottom:6px; }
.divider { border:none; border-top:1px solid #1e2740; margin:18px 0; }
.stButton > button { background:linear-gradient(135deg,#1d4ed8,#0284c7); color:white; border:none; border-radius:8px; font-family:'Space Mono',monospace; font-size:0.78rem; letter-spacing:0.05em; text-transform:uppercase; width:100%; padding:10px 20px; }
.stButton > button:hover { background:linear-gradient(135deg,#2563eb,#0369a1); box-shadow:0 4px 20px rgba(59,130,246,0.3); }
[data-testid="metric-container"] { background:#0f1628; border:1px solid #1e2d4a; border-radius:10px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <div class="header-title">🔬 Smart Scan Enhancer</div>
  <div class="header-sub">AI-powered radiology image quality booster — noise · blur · low contrast</div>
  <div class="header-badge">◆ PLUG &amp; PLAY · 4-AGENT PIPELINE · NO TRAINING REQUIRED</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Pipeline Config")
    st.markdown("**Detection thresholds**")
    noise_thresh    = st.slider("Noise sensitivity",    0.0, 1.0, 0.50, 0.05)
    blur_thresh     = st.slider("Blur sensitivity",     0.0, 1.0, 0.50, 0.05)
    contrast_thresh = st.slider("Contrast sensitivity", 0.0, 1.0, 0.40, 0.05)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Enhancement options**")
    max_retries = st.slider("Max retry attempts", 1, 5, 3)
    ssim_target = st.slider("Min SSIM gain target", 0.0, 0.20, 0.03, 0.01,
                             help="Pipeline accepts if SSIM gain >= this value")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Methods available**")
    pills = ["CLAHE","Unsharp Mask","Bilateral Filter","NL-Means","Wiener Filter",
             "Wavelet Denoise","Gamma Correct","Gaussian Denoise"]
    st.markdown(" ".join(f'<span class="method-pill">{p}</span>' for p in pills),
                unsafe_allow_html=True)

# ── Upload / sample ───────────────────────────────────────────────────────────
uc1, uc2 = st.columns([2, 1])
with uc1:
    uploaded = st.file_uploader("Upload radiology scan (JPG / PNG / BMP / TIFF)",
                                 type=["jpg","jpeg","png","bmp","tif","tiff"])
with uc2:
    st.markdown("<br>", unsafe_allow_html=True)
    use_sample      = st.checkbox("Use synthetic test scan", value=False)
    force_grayscale = st.checkbox("Force grayscale", value=True)
    show_hist       = st.checkbox("Show histograms", value=False)

def make_synthetic():
    np.random.seed(42)
    s = 400
    img = np.zeros((s, s), dtype=np.float32)
    cy, cx = s//2, s//2
    Y, X = np.ogrid[:s, :s]
    r = np.sqrt((X-cx)**2+(Y-cy)**2)
    img[r<170] = 0.35; img[r<160] = 0.42
    img[150:260,185:215] = 0.85
    img[160:240,130:160] = 0.70; img[160:240,240:270] = 0.70
    img[140:170,110:180] = 0.60; img[140:170,220:290] = 0.60
    img[130:310,100:195] = 0.14; img[130:310,205:300] = 0.14
    img += np.random.normal(0, 0.09, img.shape)
    img = np.clip(img, 0, 1)
    img = cv2.GaussianBlur(img, (7,7), 1.8)
    return (img*255).astype(np.uint8)

input_image = None
if use_sample:
    arr = make_synthetic()
    input_image = Image.fromarray(arr)
    st.info("🧪 Synthetic chest X-ray–like scan with artificial noise + blur")
elif uploaded:
    input_image = Image.open(uploaded)

# ── Pipeline UI ───────────────────────────────────────────────────────────────
if input_image is not None:
    img_array = np.array(input_image)
    if force_grayscale and len(img_array.shape)==3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    original = img_array.copy()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("## Pipeline Execution")

    col_ag, col_main = st.columns([1, 3])

    with col_ag:
        st.markdown("#### Agent Status")
        ph = [st.empty() for _ in range(4)]

        def render_agent(idx, name, icon, status, state="idle"):
            cls = {"active":"active","done":"done","retry":"retry"}.get(state,"")
            ph[idx].markdown(f"""
<div class="agent-card {cls}">
  <div class="ag-name">{icon} {name}</div>
  <div class="ag-status">{status}</div>
</div>""", unsafe_allow_html=True)

        for i,(n,ic,s) in enumerate([
            ("Quality Detector","🔍","Waiting..."),
            ("Enhancement","⚡","Waiting..."),
            ("Evaluation","📊","Waiting..."),
            ("Decision","🎯","Waiting..."),
        ]):
            render_agent(i, n, ic, s)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Pipeline Log")
        log_ph = st.empty()
        log_entries = []

        def add_log(msg, level="info"):
            ts = time.strftime("%H:%M:%S")
            color = {"info":"#7dd3fc","ok":"#34d399","warn":"#fbbf24","error":"#f87171"}.get(level,"#94a3b8")
            log_entries.append(f'<div style="font-family:Space Mono,monospace;font-size:0.7rem;color:{color};padding:2px 0;border-bottom:1px solid #0f1628;">[{ts}] {msg}</div>')
            log_ph.markdown(
                '<div style="max-height:300px;overflow-y:auto;background:#07090f;border:1px solid #1e2740;border-radius:8px;padding:10px;">'
                + "\n".join(log_entries[-25:]) + "</div>",
                unsafe_allow_html=True
            )

    with col_main:
        run_btn = st.button("▶  RUN ENHANCEMENT PIPELINE", use_container_width=True)

        if run_btn:
            add_log("Pipeline initiated", "info")

            # ── Agent 1 ──────────────────────────────────────────────────
            render_agent(0, "Quality Detector", "🔍", "Analysing image...", "active")
            add_log("Agent 1: Detecting quality issues...", "info")
            time.sleep(0.5)

            det = QualityDetectionAgent(noise_thresh, blur_thresh, contrast_thresh)
            issues = det.detect(original)
            iscores = det.scores

            render_agent(0, "Quality Detector", "🔍",
                         f"Issues: {', '.join(issues) if issues else 'none'}", "done")
            add_log(f"Detected: {issues if issues else 'no issues'}", "ok")

            # Detection report
            st.markdown("### 🔍 Detection Report")
            dc1, dc2, dc3 = st.columns(3)
            for col, label, key, flagged_color in [
                (dc1,"Noise Level",   "noise",    "#ef4444"),
                (dc2,"Blur Level",    "blur",     "#3b82f6"),
                (dc3,"Contrast Score","contrast", "#10b981"),
            ]:
                score = iscores.get(key, 0)
                flagged = key in issues or (key=="contrast" and "low_contrast" in issues)
                c = flagged_color if flagged else "#34d399"
                bar = int(score*100)
                with col:
                    st.markdown(f"""
<div class="metric-card">
  <div class="metric-lbl">{label}</div>
  <div class="metric-val" style="color:{c};">{score:.3f}</div>
  <div class="metric-sub">{'⚠ FLAGGED' if flagged else '✓ OK'}</div>
  <div class="bar-bg"><div class="bar-fill" style="width:{bar}%;background:{c};"></div></div>
</div>""", unsafe_allow_html=True)

            if issues:
                tag_map = {"noise":"t-noise","blur":"t-blur","low_contrast":"t-contrast"}
                tags = " ".join(f'<span class="issue-tag {tag_map.get(i,"t-ok")}">{i.replace("_"," ")}</span>' for i in issues)
                st.markdown(f"**Detected:** {tags}", unsafe_allow_html=True)
            else:
                st.markdown('<span class="issue-tag t-ok">✓ Image quality OK</span>', unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Agents 2-4 loop ──────────────────────────────────────────
            best_image = original.copy()
            best_metrics = None
            history = []
            total_attempts = 0

            for attempt in range(max_retries):
                total_attempts = attempt + 1

                render_agent(1, "Enhancement", "⚡", f"Attempt {attempt+1}/{max_retries}...", "active")
                add_log(f"Agent 2: Enhancement attempt {attempt+1}...", "info")
                time.sleep(0.35)

                enh = EnhancementAgent()
                method, enhanced = enh.enhance(
                    original if attempt==0 else best_image,
                    issues, attempt=attempt
                )
                render_agent(1, "Enhancement", "⚡", f"Applied: {method}", "done")
                add_log(f"Method: {method}", "ok")

                render_agent(2, "Evaluation", "📊", "Computing metrics...", "active")
                add_log("Agent 3: Measuring improvement...", "info")
                time.sleep(0.35)

                ev = EvaluationAgent()
                metrics = ev.evaluate(original, enhanced)
                render_agent(2, "Evaluation", "📊",
                             f"SSIM Δ{metrics['ssim_gain']:+.4f}", "done")
                add_log(f"SSIM gain: {metrics['ssim_gain']:+.4f}  PSNR gain: {metrics['psnr_gain']:+.2f}", "ok")

                render_agent(3, "Decision", "🎯", "Evaluating acceptance...", "active")
                time.sleep(0.3)

                dec = DecisionAgent(ssim_target=ssim_target)
                accepted, reason = dec.decide(metrics, issues)
                history.append({"attempt":attempt+1,"method":method,"metrics":metrics,
                                 "accepted":accepted,"reason":reason,"image":enhanced.copy()})

                add_log(f"Decision: {'ACCEPT' if accepted else 'RETRY'} — {reason}",
                        "ok" if accepted else "warn")

                if accepted or attempt==max_retries-1:
                    best_image = enhanced
                    best_metrics = metrics
                    render_agent(3, "Decision", "🎯",
                                 "✓ Accepted" if accepted else "⚠ Best available", "done")
                    add_log(f"Pipeline complete — {total_attempts} attempt(s)", "ok")
                    break
                else:
                    best_image = enhanced
                    best_metrics = metrics
                    render_agent(3, "Decision", "🎯", f"Retry — {reason}", "retry")
                    time.sleep(0.2)

            # ── Results ───────────────────────────────────────────────────
            st.markdown("### 📊 Enhancement Results")
            rc1, rc2 = st.columns(2)
            ch = "GRAY" if len(original.shape)==2 else "RGB"
            with rc1:
                st.image(original, use_container_width=True, clamp=True, channels=ch)
                st.markdown('<div class="img-cap">◀ Original</div>', unsafe_allow_html=True)
            with rc2:
                st.image(best_image, use_container_width=True, clamp=True, channels=ch)
                st.markdown('<div class="img-cap">▶ Enhanced</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m = best_metrics
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("SSIM Gain",      f"{m['ssim_gain']:+.4f}", delta=f"{m['ssim_gain']:.4f}")
            mc2.metric("PSNR Gain",      f"{m['psnr_gain']:+.2f} dB")
            mc3.metric("Sharpness Gain", f"{m['sharpness_gain']:+.1f}")
            mc4.metric("Attempts",       f"{total_attempts}/{max_retries}")

            accepted_final = m["ssim_gain"] >= ssim_target
            best_method = history[-1]["method"]
            vhtml = (
                f'<div class="verdict-ok">✓ ACCEPTED — {best_method} · SSIM {m["ssim_gain"]:+.4f}</div>'
                if accepted_final else
                f'<div class="verdict-warn">⚠ BEST AVAILABLE — {best_method} · SSIM {m["ssim_gain"]:+.4f}</div>'
            )
            st.markdown(vhtml, unsafe_allow_html=True)

            # Attempt history
            if len(history) > 1:
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown("### 🔄 Attempt History")
                for h in history:
                    hm = h["metrics"]
                    status_color = "#34d399" if h["accepted"] else "#fbbf24"
                    st.markdown(f"""
<div class="attempt-row">
  <span class="method-pill">#{h['attempt']} {h['method']}</span>
  &nbsp; SSIM <b style='color:{status_color};'>{hm['ssim_gain']:+.4f}</b>
  &nbsp; PSNR <b style='color:{status_color};'>{hm['psnr_gain']:+.2f} dB</b>
  &nbsp; Sharp <b style='color:{status_color};'>{hm['sharpness_gain']:+.1f}</b>
  &nbsp; <span style='color:#475569;font-size:0.72rem;'>{h['reason']}</span>
</div>""", unsafe_allow_html=True)

            # Histogram
            if show_hist:
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown("### 📈 Intensity Histogram")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1,2,figsize=(10,3),facecolor="#0f1628")
                for ax, im, title in [(axes[0],original,"Original"),(axes[1],best_image,"Enhanced")]:
                    ax.set_facecolor("#0a0e1a")
                    flat = im.flatten() if len(im.shape)==2 else cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).flatten()
                    ax.hist(flat, bins=128, color="#3b82f6", alpha=0.85, edgecolor="none")
                    ax.set_title(title, color="#94a3b8", fontsize=10)
                    ax.tick_params(colors="#475569")
                    for sp in ax.spines.values(): sp.set_edgecolor("#1e2740")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Export
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("### 💾 Export")
            ex1, ex2 = st.columns(2)
            with ex1:
                pil_out = Image.fromarray(best_image)
                buf = io.BytesIO()
                pil_out.save(buf, format="PNG")
                st.download_button("⬇ Download Enhanced Image", buf.getvalue(),
                                   "enhanced_scan.png", "image/png", use_container_width=True)
            with ex2:
                report = {
                    "pipeline": "Smart Scan Enhancer v1.0",
                    "issues_detected": issues, "issue_scores": iscores,
                    "attempts": [{"attempt":h["attempt"],"method":h["method"],
                                  "ssim_gain":round(h["metrics"]["ssim_gain"],5),
                                  "psnr_gain":round(h["metrics"]["psnr_gain"],3),
                                  "accepted":h["accepted"]} for h in history],
                    "final_method": best_method,
                    "final_metrics": {k:round(v,4) for k,v in m.items()},
                    "verdict": "accepted" if accepted_final else "best_available",
                }
                st.download_button("⬇ Download JSON Report", json.dumps(report,indent=2),
                                   "report.json", "application/json", use_container_width=True)

        else:
            # Pre-run preview
            st.markdown("### 🖼 Input Preview")
            ch = "GRAY" if len(img_array.shape)==2 else "RGB"
            st.image(img_array, use_container_width=True, clamp=True, channels=ch)
            st.markdown('<div class="img-cap">Upload complete — press Run to start</div>',
                        unsafe_allow_html=True)
            h2,w2 = img_array.shape[:2]
            flat2 = img_array.flatten() if len(img_array.shape)==2 else img_array.mean(axis=2).flatten()
            qc1,qc2,qc3,qc4 = st.columns(4)
            qc1.metric("Width",  f"{w2}px")
            qc2.metric("Height", f"{h2}px")
            qc3.metric("Mean",   f"{flat2.mean():.1f}")
            qc4.metric("Std",    f"{flat2.std():.1f}")

else:
    st.markdown("""
<div style="text-align:center;padding:70px 20px;">
  <div style="font-size:2.8rem;margin-bottom:16px;">🔬</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.95rem;color:#475569;letter-spacing:0.06em;">
    UPLOAD A RADIOLOGY SCAN TO BEGIN
  </div>
  <div style="color:#334155;font-size:0.82rem;margin-top:10px;">
    X-ray · MRI · CT · Ultrasound · DICOM slices (PNG / JPEG)
  </div>
  <div style="margin-top:22px;">
    <span class="method-pill">CLAHE</span>
    <span class="method-pill">Unsharp Mask</span>
    <span class="method-pill">Bilateral Filter</span>
    <span class="method-pill">NL-Means</span>
    <span class="method-pill">Wiener Deblur</span>
    <span class="method-pill">Wavelet Denoise</span>
    <span class="method-pill">Gamma Correct</span>
  </div>
  <div style="margin-top:18px;color:#1e2d4a;font-size:0.73rem;">
    — or enable "Use synthetic test scan" in the sidebar —
  </div>
</div>
""", unsafe_allow_html=True)
