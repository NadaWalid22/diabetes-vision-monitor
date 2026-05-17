"""
GlucoSense AI — Streamlit Demo

Inspired by the GlucoSense Apple Watch UI concept:
  - Dashboard: live glucose + 30/60-min predictions
  - Trend: 24-hour curve with meal markers and projection
  - Alerts: nocturnal hypoglycemia detection
  - Analytics: Clarke Error Grid and TIR breakdown

Run: streamlit run demo/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from data.sample.generate_synthetic import simulate_cgm
from src.models import GlucoseLSTM, NaiveLastValueBaseline
from src.preprocessing import CGMProcessor
from src.evaluation import clarke_error_grid, plot_clarke_error_grid

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlucoSense AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e; color: white; border-radius: 12px;
    padding: 16px 20px; text-align: center;
}
.glucose-value { font-size: 3rem; font-weight: 700; color: #4ade80; }
.trend-arrow  { font-size: 1.5rem; }
.zone-badge   { display: inline-block; padding: 4px 10px; border-radius: 8px;
                font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cached model & data ─────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_processor():
    processor = CGMProcessor(seq_len=24, pred_horizons=[6, 12])
    df = simulate_cgm(days=14, seed=42)
    df = processor.clean(df)
    n = len(df)
    processor.fit_scaler(df.iloc[: int(n * 0.7)])
    return processor, df


@st.cache_data
def get_predictions(_processor, _df):
    X, y = _processor.make_windows(_df, normalize=True)
    model = NaiveLastValueBaseline()
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()
    true_mg = _processor.inverse_transform(y)
    pred_mg = _processor.inverse_transform(preds)
    return X, y, true_mg, pred_mg


processor, df = load_model_and_processor()
X, y, true_mg, pred_mg = get_predictions(processor, df)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/GlucoSense-AI-4ade80?style=for-the-badge", width=200)
    st.markdown("---")
    st.subheader("Upload your CGM data")
    uploaded = st.file_uploader("CSV with 'datetime' + 'glucose' columns", type="csv")
    if uploaded:
        user_df = pd.read_csv(uploaded, parse_dates=["datetime"])
        user_df = processor.clean(user_df)
        st.success(f"Loaded {len(user_df)} readings")
        df = user_df

    st.markdown("---")
    st.caption("⚠️ For demonstration only. Not a medical device.")
    st.caption("Dataset: Synthetic (OhioT1DM-compatible format)")


# ─── Title ───────────────────────────────────────────────────────────────────
st.title("🩸 GlucoSense AI")
st.caption("Predictive glucose monitoring · Research prototype")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Trend & Prediction", "🎯 Model Analytics"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard  (mirrors GlucoSense Watch Screen 1 + 4)
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    # Simulate "live" reading from last window
    last_window = df.tail(50)
    current_glucose = float(last_window["glucose"].iloc[-1])
    prev_glucose = float(last_window["glucose"].iloc[-2])
    roc = current_glucose - prev_glucose  # mg/dL per 5 min

    # Trend arrow
    if roc > 2:
        arrow, trend_label = "↑↑", "Rising fast"
    elif roc > 0.5:
        arrow, trend_label = "↑", "Rising"
    elif roc < -2:
        arrow, trend_label = "↓↓", "Falling fast"
    elif roc < -0.5:
        arrow, trend_label = "↓", "Falling"
    else:
        arrow, trend_label = "→", "Stable"

    # Glucose zone
    if current_glucose < 54:
        zone_color, zone_text = "#ef4444", "CRITICAL LOW"
    elif current_glucose < 70:
        zone_color, zone_text = "#f97316", "LOW"
    elif current_glucose <= 180:
        zone_color, zone_text = "#4ade80", "IN RANGE"
    elif current_glucose <= 250:
        zone_color, zone_text = "#facc15", "HIGH"
    else:
        zone_color, zone_text = "#ef4444", "CRITICAL HIGH"

    # Naive predictions (last value held forward — replace with trained model)
    pred_30 = current_glucose + roc * 6
    pred_60 = current_glucose + roc * 12

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; opacity:0.7;">Current Glucose</div>
            <div class="glucose-value">{current_glucose:.0f}</div>
            <div style="opacity:0.7;">mg/dL &nbsp;
                <span class="trend-arrow">{arrow}</span>
            </div>
            <div style="margin-top:6px;">
                <span class="zone-badge" style="background:{zone_color}20;color:{zone_color};">
                    {zone_text}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.metric("30-min Prediction", f"{pred_30:.0f} mg/dL",
                  delta=f"{pred_30 - current_glucose:+.0f}")
    with col3:
        st.metric("60-min Prediction", f"{pred_60:.0f} mg/dL",
                  delta=f"{pred_60 - current_glucose:+.0f}")
    with col4:
        tir = processor.time_in_range(last_window["glucose"].values)
        st.metric("Time in Range (last 50 readings)",
                  f"{tir['tir_normal']:.0f}%",
                  delta=f"Hypo: {tir['tir_hypo_l1'] + tir['tir_hypo_l2']:.0f}%")

    st.markdown("---")

    # Sparkline — last 3 hours (36 readings at 5-min)
    recent = df.tail(36)
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=180, fillcolor="#4ade80", opacity=0.08, line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1)
    fig.add_hline(y=180, line_dash="dash", line_color="#f97316", line_width=1)
    fig.add_trace(go.Scatter(
        x=recent["datetime"], y=recent["glucose"],
        mode="lines", line=dict(color="#4ade80", width=2.5),
        name="CGM"
    ))
    # Project 60-min ahead
    future_ts = pd.date_range(recent["datetime"].iloc[-1], periods=13, freq="5min")[1:]
    future_g = np.linspace(current_glucose, pred_60, 12)
    fig.add_trace(go.Scatter(
        x=future_ts, y=future_g,
        mode="lines", line=dict(color="#94a3b8", width=1.5, dash="dot"),
        name="Projection"
    ))
    fig.update_layout(
        title="Last 3 hours + 60-min projection",
        height=260, plot_bgcolor="#111827", paper_bgcolor="#111827",
        font_color="white", showlegend=True,
        xaxis=dict(showgrid=False), yaxis=dict(range=[40, 300], gridcolor="#374151"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Screen 4: Smart Alert panel
    if current_glucose < 70 or (current_glucose < 90 and roc < -1.5):
        st.error(f"""
        🚨 **Smart Alert — Hypoglycemia Risk**

        Current: **{current_glucose:.0f} mg/dL** &nbsp; Trend: **{trend_label}** ({roc:+.1f} mg/dL per 5 min)

        Predicted 30-min: **{pred_30:.0f} mg/dL**

        **Suggested action:** Consider 15g fast-acting carbohydrate snack.
        Recheck in 15 minutes.
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — 24-hour Trend  (mirrors GlucoSense Watch Screen 2)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("24-hour Glucose Trend")

    day_idx = st.slider("Day", 0, max(0, len(df) // 288 - 1), 0)
    day_df = df.iloc[day_idx * 288: (day_idx + 1) * 288]

    fig2 = go.Figure()
    fig2.add_hrect(y0=70, y1=180, fillcolor="#4ade80", opacity=0.07, line_width=0,
                   annotation_text="Target range", annotation_position="top left")
    fig2.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1)
    fig2.add_hline(y=180, line_dash="dash", line_color="#f97316", line_width=1)
    fig2.add_hline(y=54, line_dash="dot", line_color="#dc2626", line_width=1,
                   annotation_text="Critical low", annotation_position="bottom right")

    fig2.add_trace(go.Scatter(
        x=day_df["datetime"], y=day_df["glucose"],
        mode="lines", line=dict(color="#60a5fa", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.05)",
        name="CGM"
    ))

    # Simulated meal markers at 8am, 1pm, 7pm
    # Note: add_vline with annotations is broken in Plotly 6.x + pandas 3.x;
    # add_shape + add_annotation bypasses the internal timestamp arithmetic.
    for hour, meal in [(8, "Breakfast"), (13, "Lunch"), (19, "Dinner")]:
        meal_ts = day_df["datetime"].iloc[0].normalize() + pd.Timedelta(hours=hour)
        ts_str = meal_ts.isoformat()
        fig2.add_shape(type="line", x0=ts_str, x1=ts_str, y0=0, y1=1,
                       yref="paper", line=dict(color="#facc15", width=1.5, dash="dot"))
        fig2.add_annotation(x=ts_str, y=1.04, yref="paper", text=meal,
                            showarrow=False, font=dict(color="#facc15", size=10),
                            xanchor="center")

    # TIR bar annotation
    tir_day = processor.time_in_range(day_df["glucose"].values)
    fig2.update_layout(
        title=f"Day {day_idx + 1} — TIR: {tir_day['tir_normal']:.0f}% | "
              f"Hypo: {tir_day['tir_hypo_l1'] + tir_day['tir_hypo_l2']:.0f}% | "
              f"Hyper: {tir_day['tir_hyper_l1'] + tir_day['tir_hyper_l2']:.0f}%",
        height=380, plot_bgcolor="#111827", paper_bgcolor="#111827",
        font_color="white",
        xaxis=dict(showgrid=False, title="Time"),
        yaxis=dict(range=[40, 350], gridcolor="#374151", title="Glucose (mg/dL)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Glucose", f"{day_df['glucose'].mean():.0f} mg/dL")
    c2.metric("Std Dev (Variability)", f"{day_df['glucose'].std():.0f} mg/dL")
    c3.metric("Time in Range", f"{tir_day['tir_normal']:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Analytics
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Evaluation — Clarke Error Grid")
    st.caption("Evaluating naive last-value baseline on synthetic data. "
               "Train a model with `python train.py --model all` to see real results.")

    zones_pct, zone_idx = clarke_error_grid(true_mg[:, 0], pred_mg[:, 0])

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    for col, zone_name, color in zip(
        [col_a, col_b, col_c, col_d, col_e],
        list("ABCDE"),
        ["#4ade80", "#60a5fa", "#facc15", "#f97316", "#ef4444"],
    ):
        col.markdown(
            f"<div style='text-align:center;padding:10px;background:{color}20;"
            f"border-radius:8px;border:1px solid {color}40'>"
            f"<b style='color:{color};font-size:1.4rem'>Zone {zone_name}</b><br>"
            f"<span style='font-size:1.6rem;font-weight:700'>{zones_pct[zone_name]:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info("**Zone A** = clinically safe predictions. **Zone E** = dangerous errors. "
            "A well-trained LSTM should achieve >95% Zone A on OhioT1DM data.")

    fig_ceg = plot_clarke_error_grid(
        true_mg[:, 0], pred_mg[:, 0],
        title="Clarke Error Grid — 30-min Horizon (Baseline)"
    )
    st.pyplot(fig_ceg)

    st.markdown("---")
    st.subheader("Metrics Summary")
    from src.evaluation import compute_all_metrics
    m30 = compute_all_metrics(true_mg[:, 0], pred_mg[:, 0], 30)
    m60 = compute_all_metrics(true_mg[:, 1], pred_mg[:, 1], 60)
    metrics_df = pd.DataFrame([m30, m60], index=["30-min horizon", "60-min horizon"])
    st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)
