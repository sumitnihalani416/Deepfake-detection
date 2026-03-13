"""
Streamlit dashboard for real-time deepfake video analysis.
Run: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepfakeShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #333;
        text-align: center;
    }
    .verdict-fake { color: #FF4B4B; font-size: 2rem; font-weight: 800; }
    .verdict-real { color: #00D68F; font-size: 2rem; font-weight: 800; }
    .stProgress > div > div { background: linear-gradient(90deg, #6C63FF, #FF6584); }
</style>
""", unsafe_allow_html=True)

# ─── Helper: lazy-load model and detector ────────────────────────────────────

@st.cache_resource
def load_pipeline(arch: str, detector_type: str, checkpoint_path: Optional[str]):
    """Cache the model and detector across Streamlit reruns."""
    from models.detector import build_model, load_checkpoint
    from utils.face_detector import get_detector
    from inference import DeepfakePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(architecture=arch, pretrained=(checkpoint_path is None))
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    detector = get_detector(detector_type)
    predictor = DeepfakePredictor(
        model=model,
        face_detector=detector,
        device=device,
        frames_to_sample=30,
        temporal_smoothing=True,
    )
    return predictor


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/DeepfakeShield-v1.0-6C63FF?style=for-the-badge", use_column_width=True)
    st.markdown("---")

    st.subheader("⚙️ Model Settings")
    architecture = st.selectbox(
        "Architecture",
        ["efficientnet_b4", "xception", "resnet18"],
        index=0,
        help="EfficientNet-B4 offers the best accuracy; ResNet18 is fastest.",
    )
    detector_type = st.selectbox(
        "Face Detector",
        ["mtcnn", "retinaface"],
        help="MTCNN is fast; RetinaFace is more robust for occluded/small faces.",
    )
    checkpoint_path = st.text_input(
        "Checkpoint path (optional)",
        placeholder="./checkpoints/best.pt",
        help="Path to a fine-tuned model checkpoint. Leave empty to use pretrained ImageNet weights.",
    )

    st.markdown("---")
    st.subheader("🔬 Inference Options")
    frames_to_sample = st.slider("Frames to analyse", 10, 60, 30)
    threshold = st.slider("Decision threshold", 0.3, 0.8, 0.5, 0.01)
    aggregation = st.selectbox("Score aggregation", ["mean", "max", "weighted_mean"])
    temporal_smoothing = st.toggle("Temporal smoothing", value=True)

    st.markdown("---")
    st.caption("🛡️ DeepfakeShield | Research use only")


# ─── Main area ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🛡️ DeepfakeShield</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered deepfake video analysis — frame by frame</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a video file here",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    label_visibility="collapsed",
)

# ─── Run analysis ─────────────────────────────────────────────────────────────

if uploaded:
    col_video, col_info = st.columns([1.2, 1])

    with col_video:
        st.video(uploaded)

    with col_info:
        st.markdown("#### 📋 File Info")
        st.write(f"**Name:** {uploaded.name}")
        st.write(f"**Size:** {uploaded.size / 1e6:.2f} MB")

    st.markdown("---")

    if st.button("🚀 Analyse Video", type="primary", use_container_width=True):
        with st.spinner("Loading model…"):
            predictor = load_pipeline(
                arch=architecture,
                detector_type=detector_type,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
            )
            predictor.frames_to_sample = frames_to_sample
            predictor.threshold = threshold
            predictor.aggregation = aggregation
            predictor.temporal_smoothing = temporal_smoothing

        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        progress_bar = st.progress(0, text="Extracting frames and detecting faces…")

        try:
            with st.spinner("Running inference…"):
                result = predictor.predict_video(tmp_path)
            progress_bar.progress(100, text="Done!")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            os.unlink(tmp_path)
            st.stop()
        finally:
            os.unlink(tmp_path)

        # ── Verdict ───────────────────────────────────────────────────────────

        st.markdown("---")
        vcol1, vcol2, vcol3 = st.columns(3)

        if result.get("is_fake") is None:
            vcol1.error("⚠️ No faces detected — cannot classify.")
        else:
            is_fake = result["is_fake"]
            video_score = result["video_score"]
            confidence = result["confidence"]
            label = result["label"]

            verdict_class = "verdict-fake" if is_fake else "verdict-real"
            icon = "🔴" if is_fake else "🟢"

            with vcol1:
                st.markdown(f"<div class='metric-card'><p style='color:#aaa;margin:0'>Verdict</p>"
                            f"<p class='{verdict_class}'>{icon} {label}</p></div>", unsafe_allow_html=True)
            with vcol2:
                st.markdown(f"<div class='metric-card'><p style='color:#aaa;margin:0'>Fake Score</p>"
                            f"<p style='font-size:2rem;font-weight:800;color:#6C63FF'>{video_score:.1%}</p></div>",
                            unsafe_allow_html=True)
            with vcol3:
                st.markdown(f"<div class='metric-card'><p style='color:#aaa;margin:0'>Confidence</p>"
                            f"<p style='font-size:2rem;font-weight:800;color:#FFB86C'>{confidence:.1%}</p></div>",
                            unsafe_allow_html=True)

            # ── Confidence gauge ───────────────────────────────────────────────

            st.markdown("#### Confidence Gauge")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=video_score * 100,
                title={"text": "Fake Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#FF4B4B" if is_fake else "#00D68F"},
                    "steps": [
                        {"range": [0, 40], "color": "#1a2a1a"},
                        {"range": [40, 60], "color": "#2a2a1a"},
                        {"range": [60, 100], "color": "#2a1a1a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
                delta={"reference": threshold * 100, "valueformat": ".1f"},
            ))
            gauge.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(gauge, use_container_width=True)

            # ── Frame-level timeline ──────────────────────────────────────────

            if result["frame_scores"]:
                st.markdown("#### Frame-by-Frame Analysis")
                frame_idxs = [s[0] for s in result["frame_scores"]]
                smoothed_scores = [s[1] for s in result["frame_scores"]]
                raw_scores = [s[1] for s in result.get("raw_frame_scores", result["frame_scores"])]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=frame_idxs, y=raw_scores,
                    mode="markers", name="Raw score",
                    marker=dict(color="#6C63FF", size=5, opacity=0.5),
                ))
                fig.add_trace(go.Scatter(
                    x=frame_idxs, y=smoothed_scores,
                    mode="lines", name="Smoothed",
                    line=dict(color="#FF6584", width=2.5),
                    fill="tozeroy", fillcolor="rgba(255,101,132,0.1)",
                ))
                fig.add_hline(
                    y=threshold, line_dash="dash", line_color="white", opacity=0.5,
                    annotation_text=f"Threshold ({threshold:.2f})",
                    annotation_position="top right",
                )
                fig.update_layout(
                    xaxis_title="Frame index",
                    yaxis_title="Fake probability",
                    yaxis=dict(range=[0, 1]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,30,0.8)",
                    font_color="white",
                    height=350,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Score distribution
                st.markdown("#### Score Distribution")
                dist_fig = px.histogram(
                    x=smoothed_scores, nbins=20,
                    labels={"x": "Fake probability"},
                    color_discrete_sequence=["#6C63FF"],
                )
                dist_fig.add_vline(
                    x=threshold, line_dash="dash", line_color="#FF6584",
                    annotation_text="Threshold", annotation_position="top right",
                )
                dist_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,30,0.8)",
                    font_color="white",
                    height=260,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(dist_fig, use_container_width=True)

            # ── Stats ─────────────────────────────────────────────────────────

            st.markdown("---")
            st.markdown("#### 📊 Analysis Summary")
            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("Total Frames", result["frame_count"])
            scol2.metric("Faces Detected", result["faces_found"])
            scol3.metric("FPS", f"{result['fps']:.1f}")
            scol4.metric(
                "High-risk frames",
                sum(1 for s in smoothed_scores if s >= threshold),
            )

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem; color: #555; border: 2px dashed #333; border-radius: 20px;">
        <h2 style="color:#6C63FF;">📹 Upload a video to begin</h2>
        <p>Supported formats: MP4, AVI, MOV, MKV, WEBM</p>
        <p style="font-size:0.85rem;">Supports MTCNN & RetinaFace detection + EfficientNet-B4 / Xception / ResNet18</p>
    </div>
    """, unsafe_allow_html=True)

