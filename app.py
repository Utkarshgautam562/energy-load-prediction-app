# app.py
# Streamlit — Cooling Load Predictor (India-friendly) with Predict Button
# Added: dynamic, input-aware baseline (heuristic) — better comparison than fixed value

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="Cooling Load Predictor — Smart Energy",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { font-family: 'Segoe UI', Roboto, sans-serif; }
    .big-num { font-size:40px; font-weight:700; }
    .muted { color: #6c757d; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Header
# ----------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("❄️ Cooling Load Predictor ")
    st.markdown("Predict estimated **cooling energy** required for a building.")
    st.write("Enter values on the left and press **Predict** to view the result.")
with col2:
    st.write("")

st.write("---")

# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model(path="energy_model_cooling.joblib"):
    data = joblib.load(path)
    if isinstance(data, dict):
        return data.get("model"), data.get("scaler"), {
            "r2": data.get("r2"),
            "mae": data.get("mae"),
            "feature_names": data.get("feature_names"),
            "medians": data.get("medians")  # if present from training
        }
    return data, None, {}

model, scaler, meta = load_model()

# Default performance
r2_val = meta.get("r2") or 0.96805
mae_val = meta.get("mae") or 1.08025

# Display R2 & MAE
p1, p2, p3 = st.columns([2, 2, 4])
with p1:
    st.metric("Model R²", f"{r2_val:.3f}")
with p2:
    st.metric("MAE", f"{mae_val:.2f}")
with p3:
    st.write("")

st.write("---")

# ----------------------
# Layout: Inputs + Result
# ----------------------
left, right = st.columns([1, 1.1])

with left:
    st.header("Enter Building Details")

    building_compactness = st.number_input(
        "Building compactness (0.0–1.0)",
        0.00, 1.00, 0.62, format="%.3f",
        help="How compact the building shape is."
    )
    st.caption("Example: Apartment ≈ 0.60, Bungalow ≈ 0.30")

    total_surface_area_sqm = st.number_input(
        "Total outer surface area (sq.m)",
        10.0, None, 500.0,
        format="%.2f"
    )
    st.caption("Example: 250 (2BHK), 500 (3BHK), 1200 (apartment)")

    total_wall_area_sqm = st.number_input(
        "Total wall area (sq.m)",
        10.0, None, 300.0,
        format="%.2f"
    )
    st.caption("Example: 200–400 typical")

    roof_area_sqm = st.number_input(
        "Roof area (sq.m)",
        0.0, None, 200.0,
        format="%.2f"
    )
    st.caption("Example: 50–300 depending on building")

    building_height_m = st.number_input(
        "Building height (m)",
        2.0, None, 3.50,
        format="%.2f"
    )
    st.caption("Example: 3.5 m (1 floor), 6–7 m (2 floors)")

    facing_direction = st.selectbox(
        "Facing direction",
        [
            ("North", 2),
            ("East", 3),
            ("South", 4),
            ("West", 5),
        ],
        format_func=lambda x: f"{x[0]} ({x[1]})"
    )
    st.caption("Example: South-facing (4) gets more sunlight")

    window_glass_area_sqm = st.number_input(
        "Window glass area (sq.m)",
        0.0, None, 10.0,
        format="%.2f"
    )
    st.caption("Example: 8 (few windows), 40 (many windows)")

    glass_area_distribution = st.selectbox(
        "Glass area distribution",
        [
            ("Evenly around building", 0),
            ("Mostly on front side", 1),
            ("Mostly on back side", 2),
            ("Left side mainly", 3),
            ("Right side mainly", 4),
            ("Large concentrated glazing", 5),
        ],
        format_func=lambda x: f"{x[0]} ({x[1]})"
    )
    st.caption("Example: Front windows/balcony → choose option 1")

    st.write("")
    predict_btn = st.button("🔍 Predict Cooling Load", use_container_width=True)


with right:
    st.header("Prediction")

    if predict_btn:
        # prepare inputs
        fd_num = facing_direction[1]
        gd_num = glass_area_distribution[1]

        features = np.array([[
            building_compactness,
            total_surface_area_sqm,
            total_wall_area_sqm,
            roof_area_sqm,
            building_height_m,
            fd_num,
            window_glass_area_sqm,
            gd_num
        ]])

        features_scaled = scaler.transform(features) if scaler else features

        prediction = model.predict(features_scaled)[0]

        # ---------------------------
        # Improved dynamic baseline
        # ---------------------------
        # Try to use medians saved in joblib (preferred). If not available, use a heuristic.
        medians = meta.get("medians", None)
        if medians:
            # Build reference vector using medians but scale to user's area/height
            ref = [
                medians.get("X1", 0.62),                      # compactness median
                float(total_surface_area_sqm),                # use user's surface so baseline scales
                medians.get("X3", total_wall_area_sqm),       # wall area median fallback to user's
                medians.get("X4", roof_area_sqm),             # roof area median
                float(building_height_m),                     # keep user's building height (floors matter)
                fd_num,                                       # keep user's facing direction
                medians.get("X7", window_glass_area_sqm),     # glass area median
                gd_num                                        # keep user's glass distribution
            ]
            ref = np.array([ref])
            try:
                ref_scaled = scaler.transform(ref) if scaler is not None else ref
                baseline = float(model.predict(ref_scaled)[0])
            except Exception:
                baseline = 30.0
        else:
            # Heuristic fallback baseline (simple, meaningful)
            floors = max(1, int(round(building_height_m / 3.0)))
            base_per_100sqm = 5.0
            glass_penalty = window_glass_area_sqm * 0.2
            compactness_factor = (1.0 - building_compactness) * 5.0

            baseline = (total_surface_area_sqm / 100.0) * base_per_100sqm
            baseline += glass_penalty
            baseline += floors * 1.5
            baseline -= compactness_factor
            baseline = float(max(5.0, min(baseline, 120.0)))

        diff = prediction - baseline

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"<div class='big-num'>{prediction:.2f} <span class='muted'>units</span></div>",
                unsafe_allow_html=True
            )
            st.write("**Predicted Cooling Load (Y2)** — estimated energy use.")
            st.caption(f"Baseline (comparable building) ≈ {baseline:.2f} units")
        with c2:
            st.metric("Vs baseline", f"{diff:+.2f} units")

        # progress bar
        st.progress(min(max((prediction / 60), 0), 1))

        # Feature importance
        st.markdown("### Top factors influencing cooling")
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            names = [
                "Building compactness",
                "Surface area",
                "Wall area",
                "Roof area",
                "Height",
                "Direction",
                "Window glass area",
                "Glass distribution"
            ]
            fi_df = pd.DataFrame({"feature": names, "importance": fi}).sort_values("importance", ascending=False)
            st.bar_chart(fi_df.set_index("feature"))

            st.write("Most impactful features:")
            for _, row in fi_df.head(3).iterrows():
                st.write(f"- **{row['feature']}** ({row['importance']:.3f})")
        else:
            st.write("Feature importance not available.")

# ----------------------
# Footer
# ----------------------
st.write("---")
st.caption("Made for learning & portfolio showcase — may not be suitable for real engineering decisions.")
st.caption("Created by Utkarsh Gautam")