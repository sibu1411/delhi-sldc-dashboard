"""
Delhi SLDC - 3D AI Electricity Demand System & Feeder Command Center
Streamlit Application Entry Point
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Path configuration for module resolving
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model_engine import run_enhanced_pipeline

# Page Setup
st.set_page_config(
    page_title="Delhi SLDC - 3D AI Grid Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Dark Glassmorphic Neumorphic Dashboard Aesthetics */
    .stApp {
        background-color: #0c0e14;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b26, #10141d) !important;
        border-radius: 14px !important;
        box-shadow: 4px 4px 12px #07080b, -4px -4px 12px #1d2331 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 16px 20px !important;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #11151f;
        padding: 6px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 8px 18px;
        border: none;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        box-shadow: 0px 4px 14px rgba(37, 99, 235, 0.4);
    }

    /* Header Accent */
    .title-text {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application Header
st.markdown("<h1 class='title-text'>⚡ Delhi SLDC - 3D AI Grid Command Center</h1>", unsafe_allow_html=True)
st.caption("State Load Despatch Centre | Real-Time 3D Predictive Grid Analytics & Feeder Balancing Platform")

# Sidebar Controls
st.sidebar.header("🎛️ 3D AI Command Panel")
model_options = ["Random Forest", "Gradient Boosting", "XGBoost", "Linear Regression"]
model_choice = st.sidebar.selectbox("🤖 Predictive Model Engine", model_options, index=0)

st.sidebar.subheader("🛡️ Grid Threshold Settings")
grid_capacity = st.sidebar.number_input("Max Grid Capacity (MW)", value=8200, step=250)
warning_pct = st.sidebar.slider("Warning Limit Threshold (%)", 75, 98, 90)

st.sidebar.subheader("☀️ Rooftop Solar Offset Integration")
enable_solar = st.sidebar.checkbox("Apply Solar Offset Simulation", value=True)
solar_cap = st.sidebar.slider("Installed Capacity (MW)", 0, 2000, 750, 50)
solar_eff = st.sidebar.slider("Efficiency / Irradiance Index", 0.1, 1.0, 0.95, 0.05)

st.sidebar.subheader("🧪 Climate & Stress Simulation")
temp_stress = st.sidebar.slider("Heatwave Offset (°C)", 0.0, 8.0, 0.0, 0.5)
humidity_stress = st.sidebar.slider("Monsoon Humidity Offset (%)", -20.0, 20.0, 0.0, 2.0)

# Pipeline Execution
with st.spinner("Executing 3D thermodynamic neural load forecasting pipeline..."):
    train_df, forecast_df, metrics, importances, (t_grid, h_grid, z_matrix) = run_enhanced_pipeline(
        model_choice=model_choice, temp_offset=temp_stress, humidity_offset=humidity_stress
    )

# Solar & Net Load Calculation Layer
if enable_solar:
    solar_curve = np.maximum(0, np.sin((forecast_df["timestamp"].dt.hour - 6) * np.pi / 12))
    forecast_df["solar_gen_mw"] = solar_curve * solar_cap * solar_eff
    forecast_df["net_demand_mw"] = forecast_df["predicted_demand_mw"] - forecast_df["solar_gen_mw"]
else:
    forecast_df["solar_gen_mw"] = 0.0
    forecast_df["net_demand_mw"] = forecast_df["predicted_demand_mw"]

active_col = "net_demand_mw" if enable_solar else "predicted_demand_mw"
peak_val = forecast_df[active_col].max()
peak_time = forecast_df.loc[forecast_df[active_col].idxmax(), "timestamp"]
warning_limit = grid_capacity * (warning_pct / 100.0)
overload_mw = max(0, peak_val - grid_capacity)

# Alert Notifications
if peak_val >= grid_capacity:
    st.error(
        f"🚨 **CRITICAL OVERLOAD ALERT:** Projected load reaches **{peak_val:.0f} MW** on "
        f"{peak_time.strftime('%a, %b %d at %H:%M')}. Exceeds maximum grid capacity by **{overload_mw:.0f} MW**!"
    )
elif peak_val >= warning_limit:
    st.warning(
        f"⚠️ **GRID CAPACITY WARNING:** Projected peak load reaches **{peak_val:.0f} MW** "
        f"({(peak_val / grid_capacity) * 100:.1f}% capacity) on {peak_time.strftime('%a, %b %d at %H:%M')}."
    )
else:
    st.success("✅ **GRID STATUS NORMAL:** Projected demand operates comfortably within safe grid thresholds.")

# Key Performance Indicators (KPIs)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted Peak Load", f"{peak_val:.0f} MW",
          delta=f"{temp_stress:+.1f}°C Heatwave" if temp_stress > 0 else None)
k2.metric("Available Reserve Margin", f"{grid_capacity - peak_val:.0f} MW", delta_color="inverse")
k3.metric("Model Precision (R²)", f"{metrics['R2']:.3f}", delta=f"MAPE: {metrics['MAPE']:.2f}%")
k4.metric("Peak Solar Generation", f"{forecast_df['solar_gen_mw'].max():.0f} MW" if enable_solar else "Disabled")

st.markdown("---")

# Main Dashboard Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🧊 3D Load Topology Surface",
    "📈 Hourly Forecast & Analytics",
    "⚡ Discom Feeder Matrix",
    "📥 Operational Export & Diagnostics"
])

# Tab 1: Interactive 3D Surface Mesh
with tab1:
    st.subheader("3D Electricity Load Response Surface")
    st.markdown(
        "Interactive 3D thermodynamic mesh map showing non-linear power load scaling across **Temperature (°C)** and **Hour of Day**:")

    fig_3d = go.Figure(data=[
        go.Surface(
            x=t_grid,
            y=h_grid,
            z=z_matrix,
            colorscale="Viridis",
            colorbar=dict(title="Demand (MW)"),
            hovertemplate="Temp: %{x:.1f}°C<br>Hour: %{y}:00<br>Load: %{z:.0f} MW<extra></extra>"
        )
    ])

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title="Temperature (°C)", backgroundcolor="#0e1117", gridcolor="#1e293b"),
            yaxis=dict(title="Hour of Day", backgroundcolor="#0e1117", gridcolor="#1e293b"),
            zaxis=dict(title="Load (MW)", backgroundcolor="#0e1117", gridcolor="#1e293b"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=580
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# Tab 2: 2D Forecast Time-Series & Feature Diagnostics
with tab2:
    st.subheader("72-Hour Load Forecast vs Capacity Thresholds")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_df["timestamp"].tail(48), y=train_df["demand_mw"].tail(48),
                   name="Historical Load", line=dict(color="#3b82f6", width=2))
    )
    fig.add_trace(
        go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_demand_mw"],
                   name="Gross Demand Forecast", line=dict(color="#f97316", width=2, dash="dash"))
    )

    if enable_solar:
        fig.add_trace(
            go.Scatter(x=forecast_df["timestamp"], y=forecast_df["net_demand_mw"],
                       name="Net Demand (Post Solar Offset)", line=dict(color="#10b981", width=2.5))
        )

    fig.add_hline(y=grid_capacity, line_dash="solid", line_color="#ef4444", annotation_text="Max Capacity Threshold")
    fig.add_hline(y=warning_limit, line_dash="dot", line_color="#f59e0b", annotation_text="Warning Threshold")

    fig.update_layout(
        xaxis_title="Timeline", yaxis_title="Power Demand (MW)",
        hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance Diagnostics
    st.subheader("🤖 Model Feature Importance Analysis")
    fi_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values("Importance",
                                                                                                   ascending=True)
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color="Importance",
                    color_continuous_scale="Blues", title="Feature Contribution Weight")
    fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
    st.plotly_chart(fig_fi, use_container_width=True)

# Tab 3: Discom Feeder Load Shedding Allocation Matrix
with tab3:
    st.subheader("Discom Zone Load Allocation & Feeder Management Protocol")

    discom_weights = {
        "BSES Rajdhani Power Ltd (BRPL)": {"share": 0.42, "priority": "P2 (Residential / Commercial)"},
        "Tata Power DDL (TPDDL)": {"share": 0.28, "priority": "P3 (Industrial / Suburban)"},
        "BSES Yamuna Power Ltd (BYPL)": {"share": 0.22, "priority": "P2 (Dense Urban Residential)"},
        "New Delhi Municipal Council (NDMC)": {"share": 0.08, "priority": "P1 (VIP Core / Critical Infrastructure)"},
    }

    discom_rows = []
    for discom, info in discom_weights.items():
        allocated = peak_val * info["share"]
        shed_needed = overload_mw * info["share"] if overload_mw > 0 else 0.0
        status = "🚨 Overload Mitigation Needed" if shed_needed > 0 else "✅ Normal Operations"

        discom_rows.append({
            "Discom Zone": discom,
            "Grid Share": f"{info['share'] * 100:.0f}%",
            "Allocated Peak Load (MW)": round(allocated, 1),
            "Mandatory MW Shedding": round(shed_needed, 1),
            "Feeder Priority Level": info["priority"],
            "Operational Status": status
        })

    df_discom = pd.DataFrame(discom_rows)
    st.dataframe(df_discom, use_container_width=True, hide_index=True)

# Tab 4: Operational Data Export & Model Performance Breakdown
with tab4:
    st.subheader("Operational Forecast Data Export & Model Evaluation")

    c1, c2 = st.columns([2, 1])

    with c1:
        export_df = forecast_df[
            ["timestamp", "temperature", "humidity", "heat_index", "predicted_demand_mw", "solar_gen_mw",
             "net_demand_mw"]
        ].copy()
        export_df.columns = [
            "Timestamp", "Temperature (°C)", "Humidity (%)", "Heat Index (°C)",
            "Gross Load (MW)", "Solar Offset (MW)", "Net Load (MW)"
        ]

        st.dataframe(export_df, use_container_width=True, height=360)

        csv_buffer = io.BytesIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Complete 72-Hour Operational Schedule (CSV)",
            data=csv_buffer.getvalue(),
            file_name="delhi_sldc_72h_demand_forecast.csv",
            mime="text/csv",
        )

    with c2:
        st.markdown("### 📊 Model Diagnostics")
        st.write(f"**Engine:** `{model_choice}`")
        st.write(f"**R² Score:** `{metrics['R2']:.4f}`")
        st.write(f"**Mean Absolute Error:** `{metrics['MAE']:.2f} MW`")
        st.write(f"**Root Mean Squared Error:** `{metrics['RMSE']:.2f} MW`")
        st.write(f"**Mean Absolute Percentage Error:** `{metrics['MAPE']:.2f}%`")
