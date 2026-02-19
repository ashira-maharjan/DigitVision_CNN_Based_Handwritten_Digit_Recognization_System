

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from streamlit_drawable_canvas import st_canvas
from src.model import CNN

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DigitVision",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# CSS  ── "Instrument Panel" aesthetic
#   warm cream + deep charcoal + amber accent
#   typewriter + engineering grid
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300&family=Share+Tech+Mono&family=Bebas+Neue&display=swap');

/* ── reset & base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0e0d0b;
    color: #d4c9a8;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── engineering grid bg ───────────────────────────────────── */
body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,180,50,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,180,50,.025) 1px, transparent 1px);
    background-size: 48px 48px;
}

/* ── scanline overlay ──────────────────────────────────────── */
body::after {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,.18) 2px, rgba(0,0,0,.18) 4px
    );
}

/* ── all content above overlays ───────────────────────────── */
.main > div { position: relative; z-index: 1; }
section[data-testid="stSidebar"] { z-index: 2; }

/* ── masthead ──────────────────────────────────────────────── */
.masthead {
    background: #0e0d0b;
    border-bottom: 1px solid #3a3220;
    padding: 18px 48px;
    display: flex;
    align-items: baseline;
    gap: 20px;
}
.masthead-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 6px;
    color: white;
    line-height: 1;
}
.masthead-sub {
    font-size: .68rem;
    letter-spacing: 4px;
    color: white;
    text-transform: uppercase;
    border-left: 2px solid #3a3220;
    padding-left: 16px;
    margin-left: 4px;
}
.masthead-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #f5b942;
    box-shadow: 0 0 10px #f5b942, 0 0 24px rgba(245,185,66,.5);
    margin-left: auto;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .5; transform: scale(.8); }
}

/* ── page body padding ─────────────────────────────────────── */
.page-body { padding: 32px 48px; }

/* ── section labels ────────────────────────────────────────── */
.sec-label {
    font-size: .62rem;
    letter-spacing: 4px;
    color: white;
    text-transform: uppercase;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #2a2215;
}

/* ── input mode toggle ─────────────────────────────────────── */
.mode-toggle {
    display: flex;
    gap: 2px;
    background: #1a1810;
    border: 1px solid #2a2215;
    border-radius: 4px;
    padding: 3px;
    width: fit-content;
    margin-bottom: 24px;
}
.mode-btn {
    padding: 8px 22px;
    border-radius: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    cursor: pointer;
    transition: all .15s;
    background: transparent;
    color: white;
}
.mode-btn.active {
    background: #f5b942;
    color: white;
    font-weight: bold;
}

/* ── canvas frame ──────────────────────────────────────────── */
.canvas-frame {
    border: 1px solid #2a2215;
    border-radius: 4px;
    padding: 12px;
    background: #0a0906;
    position: relative;
}
.canvas-corner {
    position: absolute;
    width: 12px; height: 12px;
    border-color: #f5b942;
    border-style: solid;
}
.cc-tl { top: 6px; left: 6px;
          border-width: 1px 0 0 1px; }
.cc-tr { top: 6px; right: 6px;
          border-width: 1px 1px 0 0; }
.cc-bl { bottom: 6px; left: 6px;
          border-width: 0 0 1px 1px; }
.cc-br { bottom: 6px; right: 6px;
          border-width: 0 1px 1px 0; }

/* ── upload zone ───────────────────────────────────────────── */
.upload-zone {
    border: 1px dashed #2a2215;
    border-radius: 4px;
    padding: 40px 20px;
    text-align: center;
    color: #3a3220;
    background: #0a0906;
    transition: border-color .2s;
}
.upload-zone:hover { border-color: #f5b942; }

/* ── result panel ──────────────────────────────────────────── */
.result-panel {
    background: #0a0906;
    border: 1px solid #2a2215;
    border-radius: 4px;
    padding: 28px 24px;
    position: relative;
    overflow: hidden;
}
.result-panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #f5b942, transparent);
}
.result-digit-wrap {
    display: flex;
    align-items: center;
    gap: 28px;
    margin-bottom: 20px;
}
.result-digit {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 7rem;
    line-height: 1;
    color: #f5b942;
    text-shadow: 0 0 40px rgba(245,185,66,.4), 0 0 80px rgba(245,185,66,.15);
    letter-spacing: 4px;
}
.result-meta { flex: 1; }
.result-conf-bar-wrap {
    margin-top: 6px;
}
.result-conf-label {
    font-size: .62rem;
    letter-spacing: 3px;
    color: #5c5035;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.result-conf-track {
    height: 6px;
    background: #1a1810;
    border-radius: 1px;
    overflow: hidden;
}
.result-conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #f5b942, #ffda80);
    border-radius: 1px;
    transition: width .6s ease;
}
.conf-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #f5b942;
    letter-spacing: 2px;
    line-height: 1;
}

/* ── prob grid ─────────────────────────────────────────────── */
.prob-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    margin-top: 16px;
}
.prob-cell {
    background: #0e0d0b;
    border: 1px solid #1a1810;
    border-radius: 3px;
    padding: 8px 6px;
    text-align: center;
    transition: border-color .2s;
}
.prob-cell.top {
    border-color: #f5b942;
    background: rgba(245,185,66,.04);
}
.prob-cell-digit {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: #5c5035;
    line-height: 1;
}
.prob-cell.top .prob-cell-digit { color: #f5b942; }
.prob-cell-val {
    font-size: .6rem;
    color: #3a3220;
    margin-top: 2px;
}
.prob-cell.top .prob-cell-val { color: #d4c9a8; }

/* ── 28×28 preview ─────────────────────────────────────────── */
.preview-label {
    font-size: .6rem;
    letter-spacing: 3px;
    color: white;
    text-transform: uppercase;
    text-align: center;
    margin-top: 8px;
}

/* ── status tag ────────────────────────────────────────────── */
.tag {
    display: inline-block;
    font-size: .6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 2px 10px;
    border-radius: 2px;
    margin-bottom: 8px;
}
.tag-ok  { background: rgba(74,222,128,.08); color: #4ade80;
           border: 1px solid rgba(74,222,128,.2); }
.tag-off { background: rgba(245,185,66,.08); color: #f5b942;
           border: 1px solid rgba(245,185,66,.2); }

/* ── streamlit overrides ───────────────────────────────────── */
.stButton > button {
    background: #3b82f6  !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .72rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    font-weight: bold !important;
    transition: opacity .2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: .85 !important; }

[data-testid="stFileUploader"] {
    background: #0a0906;
    border: 1px dashed #2a2215;
    border-radius: 4px;
    padding: 12px;
}
[data-testid="stFileUploader"]:hover { border-color: #f5b942; }

[data-testid="stRadio"] { display: none; }

div[data-testid="stImage"] img { border-radius: 3px; }

.stSuccess, .stInfo, .stWarning {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .75rem !important;
    border-radius: 3px !important;
}

footer { display: none; }
#MainMenu { display: none; }
header[data-testid="stHeader"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SETUP — YOUR ORIGINAL LOGIC UNCHANGED
# ══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

@st.cache_resource
def load_model():
    m = CNN().to(device)
    m.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
    m.eval()
    return m

model = load_model()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# ══════════════════════════════════════════════════════════════
# MASTHEAD
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="masthead">
  <div class="masthead-title">DigitVision</div>
  <div class="masthead-sub">EEnd-to-End CNN-Based Handwritten Digit Recognition System</div>
  <div class="masthead-dot"></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE BODY
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="page-body">', unsafe_allow_html=True)

# ── Input mode state ──────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "Draw"

# ── Mode toggle (custom buttons over hidden radio) ────────────
col_btn1, col_btn2, col_spacer = st.columns([1, 1, 6])
if col_btn1.button(" Draw Digit"):
    st.session_state.mode = "Draw"
if col_btn2.button(" Upload Image"):
    st.session_state.mode = "Upload"

mode = st.session_state.mode

# ── Active mode indicator ─────────────────────────────────────
tag_html = (
    '<span class="tag tag-ok">● Draw Mode Active</span>'
    if mode == "Draw" else
    '<span class="tag tag-off">● Upload Mode Active</span>'
)
st.markdown(tag_html, unsafe_allow_html=True)

st.markdown("---", unsafe_allow_html=False)

# ══════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 1.4], gap="large")
image = None

# ─────────────────────────────────────────────────────────────
# LEFT — INPUT PANEL
# ─────────────────────────────────────────────────────────────
with left_col:

    # ── DRAW MODE ─────────────────────────────────────────────
    if mode == "Draw":
        st.markdown('<div class="sec-label">Canvas Input</div>', unsafe_allow_html=True)

        st.markdown('<div class="canvas-frame">'
                    '<div class="canvas-corner cc-tl"></div>'
                    '<div class="canvas-corner cc-tr"></div>'
                    '<div class="canvas-corner cc-bl"></div>'
                    '<div class="canvas-corner cc-br"></div>',
                    unsafe_allow_html=True)

        canvas = st_canvas(
            fill_color       = "black",
            stroke_width     = 18,
            stroke_color     = "white",
            background_color = "black",
            width            = 300,
            height           = 300,
            drawing_mode     = "freedraw",
            key              = "canvas",
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if canvas.image_data is not None:
            img = Image.fromarray(
                canvas.image_data[:, :, 0].astype("uint8")
            ).convert("L")
            image = img

            # 28×28 preview
            preview = img.resize((28, 28), Image.LANCZOS)
            preview_col, _ = st.columns([1, 3])
            with preview_col:
                st.image(np.array(preview), width=84, clamp=True)
                st.markdown('<div class="preview-label">28×28 input</div>',
                            unsafe_allow_html=True)

            save_path = os.path.join("uploads", "drawn_image.png")
            img.save(save_path)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⬡ Decode Digit")

    # ── UPLOAD MODE ───────────────────────────────────────────
    else:
        st.markdown('<div class="sec-label">Image Upload</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop PNG / JPG / JPEG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("L")

            # Show original + 28×28 side by side
            oc, pc = st.columns(2)
            with oc:
                st.markdown('<div class="sec-label">Original</div>',
                            unsafe_allow_html=True)
                st.image(np.array(image), width=140, clamp=True)
            with pc:
                preview = image.resize((28, 28), Image.LANCZOS)
                st.markdown('<div class="sec-label">28×28 Input</div>',
                            unsafe_allow_html=True)
                st.image(np.array(preview), width=84, clamp=True)

            save_path = os.path.join("uploads", uploaded_file.name)
            image.save(save_path)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⬡ Decode Digit")

# ─────────────────────────────────────────────────────────────
# RIGHT — RESULT PANEL
# ─────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="sec-label">Prediction Output</div>', unsafe_allow_html=True)

    # Run prediction when button pressed and image exists
    if 'predict_btn' in dir() and predict_btn and image is not None:

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs       = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted  = torch.max(outputs, 1)

        digit      = predicted.item()
        confidence = torch.max(probabilities).item() * 100
        probs_arr  = probabilities.squeeze().cpu().numpy()

        # ── Big result card ──────────────────────────────────
        st.markdown(f"""
        <div class="result-panel">
          <div class="result-digit-wrap">
            <div class="result-digit">{digit}</div>
            <div class="result-meta">
              <div class="result-conf-label">Confidence Score</div>
              <div class="conf-number">{confidence:.1f}<span style="font-size:1rem;color:#5c5035">%</span></div>
              <div class="result-conf-bar-wrap">
                <div class="result-conf-track">
                  <div class="result-conf-fill" style="width:{confidence:.1f}%"></div>
                </div>
              </div>
              <div style="font-size:.6rem;color:#3a3220;margin-top:8px;letter-spacing:2px">
                DEVICE &nbsp;·&nbsp; {'GPU' if torch.cuda.is_available() else 'CPU'}
                &nbsp;&nbsp;|&nbsp;&nbsp; MODEL &nbsp;·&nbsp; CNN
              </div>
            </div>
          </div>

          <div class="result-conf-label" style="margin-bottom:8px">All Class Probabilities</div>
          <div class="prob-grid">
            {''.join(
                f'<div class="prob-cell {"top" if i == digit else ""}">'
                f'  <div class="prob-cell-digit">{i}</div>'
                f'  <div class="prob-cell-val">{probs_arr[i]*100:.1f}%</div>'
                f'</div>'
                for i in range(10)
            )}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Matplotlib probability bar chart ─────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Distribution Chart</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor("#0a0906")
        ax.set_facecolor("#0a0906")

        bar_colors = ["#f5b942" if i == digit else "#1a1810" for i in range(10)]
        edge_colors= ["#f5b942" if i == digit else "#2a2215" for i in range(10)]
        bars = ax.bar(
            range(10), probs_arr * 100,
            color=bar_colors, edgecolor=edge_colors,
            linewidth=0.8, width=0.65
        )
        ax.set_xticks(range(10))
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(0, 115)
        ax.set_xlabel("Digit Class", color="#3a3220", fontsize=8,
                      fontfamily="monospace", labelpad=6)
        ax.set_ylabel("Probability %", color="#3a3220", fontsize=8,
                      fontfamily="monospace", labelpad=6)
        ax.tick_params(colors="#3a3220", labelsize=8)
        ax.tick_params(axis='x', colors="#5c5035")
        for s in ax.spines.values(): s.set_color("#1a1810")
        ax.yaxis.grid(True, color="#1a1810", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        for i, bar in enumerate(bars):
            if probs_arr[i] > 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{probs_arr[i]*100:.1f}%",
                    ha="center", va="bottom",
                    color="#f5b942" if i == digit else "#2a2215",
                    fontsize=7, fontfamily="monospace"
                )

        plt.tight_layout(pad=0.4)
        st.pyplot(fig)
        plt.close()

    elif 'predict_btn' in dir() and predict_btn and image is None:
        st.markdown("""
        <div class="result-panel" style="text-align:center;padding:60px 24px">
          <div style="font-size:2rem;margin-bottom:12px">⚠️</div>
          <div style="font-size:.72rem;letter-spacing:3px;color:#3a3220">
              NO INPUT DETECTED<br>
              <span style="color:#1a1810;font-size:.6rem">
                  Draw something or upload an image first
              </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div class="result-panel" style="min-height:340px;display:flex;
             flex-direction:column;align-items:center;justify-content:center;
             text-align:center;gap:16px">
          <div style="font-family:'Bebas Neue',sans-serif;font-size:5rem;
                      color:#1a1810;letter-spacing:4px;line-height:1">
              ?
          </div>
          <div style="font-size:.65rem;letter-spacing:4px;color:#2a2215;
                      text-transform:uppercase">
              Awaiting Input<br>
              <span style="font-size:.58rem;color:#1a1810">
                  Draw or upload a digit · then click Decode
              </span>
          </div>
          <div style="display:flex;gap:10px;margin-top:8px">
            <div style="width:6px;height:6px;border-radius:50%;background:#1a1810"></div>
            <div style="width:6px;height:6px;border-radius:50%;background:#2a2215"></div>
            <div style="width:6px;height:6px;border-radius:50%;background:#1a1810"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close page-body

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1a1810;margin:0 48px;padding:16px 0;
            font-size:.58rem;color:#2a2215;letter-spacing:3px;
            text-transform:uppercase;display:flex;justify-content:space-between">
  <span>DigitDecoder · CNN · MNIST</span>
  <span>PyTorch + Streamlit</span>
</div>
""", unsafe_allow_html=True)