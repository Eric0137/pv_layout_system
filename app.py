"""
Streamlit Web Application - PV Layout Design System
====================================================
Interactive web interface for automated rooftop PV layout design.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import json
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PANEL_SPEC, LAYOUT, MALAYSIA_IRRADIANCE,
    CNN_CONFIG, OBSTACLE_DATASET_DIR, PERFORMANCE, TARIFF
)
from roof_detector import RoofDetector
from obstacle_classifier import ObstacleClassifier, SyntheticObstacleGenerator, TORCH_AVAILABLE
from pv_optimizer import PVLayoutOptimizer
from energy_calculator import EnergyCalculator, compare_regions
from visualization import LayoutVisualizer, ChartGenerator


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PV Layout Designer - Malaysia",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☀️ Automated PV Layout Design System")
st.markdown("*Optimised for Malaysian Rooftop Solar under ATAP*")


# ============================================================
# SESSION STATE INIT
# ============================================================
if 'detector' not in st.session_state:
    st.session_state.detector = RoofDetector()
if 'classifier' not in st.session_state:
    st.session_state.classifier = ObstacleClassifier()
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = PVLayoutOptimizer()
if 'panels' not in st.session_state:
    st.session_state.panels = []
if 'image_loaded' not in st.session_state:
    st.session_state.image_loaded = False


# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================
with st.sidebar:
    st.header("⚙️ System Configuration")

    st.subheader("Panel Specifications")
    panel_power = st.number_input(
        "Panel Power (Wp)", value=PANEL_SPEC["rated_power_wp"],
        min_value=100, max_value=800, step=10
    )
    panel_length = st.number_input(
        "Panel Length (m)", value=PANEL_SPEC["length_m"],
        min_value=0.5, max_value=3.0, step=0.01, format="%.3f"
    )
    panel_width = st.number_input(
        "Panel Width (m)", value=PANEL_SPEC["width_m"],
        min_value=0.3, max_value=2.0, step=0.01, format="%.3f"
    )

    st.subheader("Layout Parameters")
    tilt_angle = st.slider("Tilt Angle (°)", 0, 30, LAYOUT["tilt_angle_deg"])
    azimuth = st.slider("Azimuth (° from N)", 0, 360, LAYOUT["azimuth_deg"])
    edge_setback = st.number_input(
        "Edge Setback (m)", value=LAYOUT["min_edge_setback_m"],
        min_value=0.0, max_value=2.0, step=0.1
    )
    orientation = st.selectbox(
        "Panel Orientation", ["auto", "portrait", "landscape"]
    )

    st.subheader("Location")
    region = st.selectbox(
        "Region", list(MALAYSIA_IRRADIANCE.keys()),
        index=list(MALAYSIA_IRRADIANCE.keys()).index("Kuala_Lumpur")
    )

    st.subheader("CNN Model")
    if TORCH_AVAILABLE:
        model_path = CNN_CONFIG["model_save_path"]
        if os.path.exists(model_path):
            st.success("✅ Trained model found")
            if st.button("Load Model"):
                st.session_state.classifier.load_model(model_path)
                st.success("Model loaded!")
        else:
            st.warning("⚠️ No trained model found")
            if st.button("Generate Synthetic Data & Train"):
                with st.spinner("Generating data..."):
                    gen = SyntheticObstacleGenerator()
                    gen.generate_dataset(OBSTACLE_DATASET_DIR, samples_per_class=200)
                with st.spinner("Training CNN (this may take a few minutes)..."):
                    clf = st.session_state.classifier
                    train_l, val_l, test_l = clf.prepare_datasets(OBSTACLE_DATASET_DIR)
                    history = clf.train(train_l, val_l, epochs=20, lr=0.001)
                    metrics = clf.evaluate(test_l)
                    st.success(
                        f"Training complete! Accuracy: {metrics['overall_accuracy']:.1%}"
                    )
    else:
        st.error("PyTorch not installed. CNN unavailable.")


# ============================================================
# MAIN CONTENT - TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Detect",
    "📐 Layout Design",
    "⚡ Energy Analysis",
    "💰 Financial Report"
])


# ============================================================
# TAB 1: IMAGE UPLOAD & ROOF DETECTION
# ============================================================
with tab1:
    st.header("Step 1: Upload Roof Image")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a roof image (aerial/satellite view)",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tif']
        )

        use_demo = st.checkbox("Use demo image instead")

        if use_demo:
            # Generate demo image
            demo_img = np.full((600, 800, 3), (180, 170, 160), dtype=np.uint8)
            roof_pts = np.array(
                [[100, 80], [700, 80], [720, 520], [80, 520]], dtype=np.int32
            )
            cv2.fillPoly(demo_img, [roof_pts], (200, 195, 185))
            cv2.polylines(demo_img, [roof_pts], True, (120, 115, 110), 2)
            cv2.rectangle(demo_img, (500, 100), (600, 180), (100, 95, 90), -1)
            cv2.circle(demo_img, (200, 400), 30, (90, 85, 80), -1)
            noise = np.random.randint(-8, 8, demo_img.shape, dtype=np.int16)
            demo_img = np.clip(
                demo_img.astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)

            st.session_state.detector.load_image_from_array(demo_img)
            st.session_state.image_loaded = True
            st.image(
                cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB),
                caption="Demo Roof Image", use_container_width=True
            )

        elif uploaded_file:
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()), dtype=np.uint8
            )
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                st.session_state.detector.load_image_from_array(img)
                st.session_state.image_loaded = True
                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    caption="Uploaded Roof Image", use_container_width=True
                )
            else:
                st.error("Could not read the uploaded image.")

    with col2:
        if st.session_state.image_loaded:
            st.subheader("Roof Dimensions")
            roof_length = st.number_input(
                "Roof Length (m)", value=20.0, min_value=1.0, max_value=200.0
            )
            roof_width = st.number_input(
                "Roof Width (m)", value=15.0, min_value=1.0, max_value=200.0
            )
            north_angle = st.slider(
                "North Direction (° from image top)", 0, 360, 0
            )

            detect_method = st.selectbox(
                "Detection Method",
                ["auto", "canny", "adaptive", "color", "manual"]
            )

            if st.button("🔍 Detect Roof Boundary", type="primary"):
                detector = st.session_state.detector
                detector.set_north_angle(north_angle)

                with st.spinner("Detecting roof boundary..."):
                    try:
                        if detect_method == "manual":
                            # Use full image as roof area
                            h, w = detector.image_shape
                            margin = 20
                            detector.manually_define_roof([
                                (margin, margin),
                                (w - margin, margin),
                                (w - margin, h - margin),
                                (margin, h - margin)
                            ])
                        else:
                            detector.detect_roof_boundary(method=detect_method)

                        detector.set_scale_from_roof_dims(roof_length, roof_width)

                        # Detect obstacles
                        candidates = detector.detect_potential_obstacles()

                        # Classify if model is trained
                        clf = st.session_state.classifier
                        if clf.is_trained:
                            candidates = clf.classify_batch(candidates)

                        # Remove non-"none" obstacles
                        obs_contours = [
                            o['contour'] for o in candidates
                            if o.get('label', 'unknown') != 'none'
                        ]
                        if obs_contours:
                            detector.remove_obstacles_from_mask(obs_contours)

                        # Store candidates
                        st.session_state.candidates = candidates

                        # Display results
                        summary = detector.get_detection_summary()
                        st.success("Roof boundary detected!")
                        st.json(summary)

                        # Show detected boundary overlay
                        vis_img = detector.original_image.copy()
                        cv2.drawContours(
                            vis_img, [detector.roof_contour], -1, (0, 255, 0), 3
                        )
                        for obs in candidates:
                            if obs.get('label') != 'none':
                                cv2.drawContours(
                                    vis_img, [obs['contour']], -1, (0, 0, 255), 2
                                )
                        st.image(
                            cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB),
                            caption="Detected Boundaries",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Detection failed: {str(e)}")
                        st.info("Try 'manual' method or adjust image quality.")


# ============================================================
# TAB 2: LAYOUT OPTIMISATION
# ============================================================
with tab2:
    st.header("Step 2: Optimise Panel Layout")

    detector = st.session_state.detector

    if detector.roof_contour is None or detector.scale_m_per_px is None:
        st.warning("⚠️ Please detect the roof boundary first (Tab 1).")
    else:
        if st.button("⚡ Generate Optimal Layout", type="primary"):
            # Update panel spec from sidebar
            PANEL_SPEC["rated_power_wp"] = panel_power
            PANEL_SPEC["length_m"] = panel_length
            PANEL_SPEC["width_m"] = panel_width
            LAYOUT["tilt_angle_deg"] = tilt_angle
            LAYOUT["azimuth_deg"] = azimuth
            LAYOUT["min_edge_setback_m"] = edge_setback

            optimizer = st.session_state.optimizer

            with st.spinner("Optimising panel placement..."):
                panels = optimizer.optimize_layout(
                    usable_area_mask=detector.usable_area_mask,
                    scale_m_per_px=detector.scale_m_per_px,
                    orientation=orientation,
                    tilt_deg=tilt_angle,
                    add_walkway=True
                )

                st.session_state.panels = panels

            # Display layout summary
            summary = optimizer.get_layout_summary()
            inverter = optimizer.get_inverter_recommendation()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Panels", summary['total_panels'])
                st.metric("Capacity", f"{summary['total_capacity_kwp']} kWp")
            with col2:
                st.metric("Roof Utilisation", f"{summary['roof_utilisation_pct']}%")
                st.metric("Strings", summary['number_of_strings'])
            with col3:
                st.metric("Inverter", f"{inverter['inverter_quantity']}x {inverter['inverter_size_kw']}kW")
                st.metric("DC/AC Ratio", inverter['dc_ac_ratio'])

            # Draw layout overlay
            candidates = st.session_state.get('candidates', [])
            result_image = LayoutVisualizer.draw_layout_overlay(
                image=detector.original_image,
                panels=panels,
                scale_m_per_px=detector.scale_m_per_px,
                roof_contour=detector.roof_contour,
                obstacles=candidates,
                usable_mask=detector.usable_area_mask,
                show_strings=True,
                alpha=0.5
            )

            st.image(
                cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                caption=f"Optimised Layout: {summary['total_panels']} panels, "
                        f"{summary['total_capacity_kwp']} kWp",
                use_container_width=True
            )

            # Detailed tables
            with st.expander("Layout Details"):
                st.json(summary)
            with st.expander("Inverter Recommendation"):
                st.json(inverter)


# ============================================================
# TAB 3: ENERGY ANALYSIS
# ============================================================
with tab3:
    st.header("Step 3: Energy Yield Estimation")

    panels = st.session_state.panels

    if not panels:
        st.warning("⚠️ Generate a layout first (Tab 2).")
    else:
        calc = EnergyCalculator(region=region)
        calc.set_system(
            n_panels=len(panels),
            tilt_deg=tilt_angle,
            azimuth_deg=azimuth
        )

        yield_data = calc.calculate_annual_yield()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Annual Yield", f"{yield_data['net_annual_yield_kwh']:,.0f} kWh")
        with col2:
            st.metric("Specific Yield", f"{yield_data['specific_yield_kwh_kwp']:,.0f} kWh/kWp")
        with col3:
            st.metric("Performance Ratio", f"{yield_data['performance_ratio']:.1%}")
        with col4:
            st.metric("Daily Average", f"{yield_data['daily_avg_yield_kwh']:.1f} kWh")

        # Monthly yield chart
        st.subheader("Monthly Energy Production")
        monthly = yield_data['monthly_yield_kwh']
        fig, ax = plt.subplots(figsize=(10, 4))
        months = list(monthly.keys())
        values = list(monthly.values())
        bars = ax.bar(months, values, color='#2196F3', edgecolor='white')
        ax.set_ylabel('Energy (kWh)')
        ax.set_title(f'Estimated Monthly Yield ({region})')
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 10,
                    f'{v:.0f}', ha='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Loss breakdown
        with st.expander("Loss Breakdown"):
            st.markdown(f"""
            | Factor | Value |
            |--------|-------|
            | Temperature Derating | {yield_data['temperature_derating']:.2%} |
            | Soiling | {yield_data['soiling_factor']:.2%} |
            | Mismatch | {yield_data['mismatch_factor']:.2%} |
            | Wiring | {yield_data['wiring_factor']:.2%} |
            | Inverter Efficiency | {yield_data['inverter_efficiency']:.2%} |
            | Availability | {yield_data['availability']:.2%} |
            | **Combined Factor** | **{yield_data['combined_system_factor']:.2%}** |
            """)

        # Regional comparison
        st.subheader("Regional Comparison")
        region_data = compare_regions(n_panels=len(panels))
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        regions_list = list(region_data.keys())
        yields_list = [region_data[r]['annual_yield_kwh'] for r in regions_list]
        colors = ['#FF9800' if r != region else '#4CAF50' for r in regions_list]
        ax2.barh(regions_list, yields_list, color=colors)
        ax2.set_xlabel('Annual Yield (kWh)')
        ax2.set_title('Yield by Region (green = selected)')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ============================================================
# TAB 4: FINANCIAL REPORT
# ============================================================
with tab4:
    st.header("Step 4: Financial Analysis (ATAP/NEM)")

    panels = st.session_state.panels

    if not panels:
        st.warning("⚠️ Generate a layout first (Tab 2).")
    else:
        # Financial inputs
        col1, col2 = st.columns(2)
        with col1:
            cost_per_wp = st.number_input(
                "Cost per Wp (MYR)",
                value=TARIFF["system_cost_myr_per_wp"],
                min_value=1.0, max_value=10.0, step=0.1
            )
            nem_rate = st.number_input(
                "NEM Rate (MYR/kWh)",
                value=TARIFF["nem_rate_myr_kwh"],
                min_value=0.1, max_value=1.0, step=0.01
            )
        with col2:
            annual_om = st.number_input(
                "Annual O&M (MYR)",
                value=float(TARIFF["annual_maintenance_myr"]),
                min_value=0.0, max_value=5000.0, step=100.0
            )

        # Override tariff
        TARIFF["system_cost_myr_per_wp"] = cost_per_wp
        TARIFF["nem_rate_myr_kwh"] = nem_rate
        TARIFF["annual_maintenance_myr"] = annual_om

        calc = EnergyCalculator(region=region)
        calc.set_system(n_panels=len(panels), tilt_deg=tilt_angle)
        yield_data = calc.calculate_annual_yield()
        financial = calc.calculate_financial_returns(yield_data['net_annual_yield_kwh'])

        # Key financial metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Cost", f"MYR {financial['total_system_cost_myr']:,.0f}")
        with col2:
            st.metric("Year 1 Savings", f"MYR {financial['year1_savings_myr']:,.0f}")
        with col3:
            st.metric("Payback", f"{financial['simple_payback_years']} years")
        with col4:
            st.metric("25-Year ROI", f"{financial['roi_pct']}%")

        # Payback chart
        st.subheader("Payback Projection")
        yearly = financial['yearly_breakdown']
        years = [d['year'] for d in yearly]
        cumulative = [d['cumulative_profit_myr'] for d in yearly]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, cumulative, 'b-o', markersize=3, linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.fill_between(years, cumulative, 0,
                        where=[c >= 0 for c in cumulative],
                        alpha=0.3, color='green', label='Profit')
        ax.fill_between(years, cumulative, 0,
                        where=[c < 0 for c in cumulative],
                        alpha=0.3, color='red', label='Payback')
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Profit (MYR)')
        ax.set_title('25-Year Financial Projection')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Detailed breakdown
        with st.expander("Yearly Breakdown"):
            import pandas as pd
            df = pd.DataFrame(yearly)
            st.dataframe(df, use_container_width=True)

        # Summary card
        st.subheader("Summary")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total System Cost | MYR {financial['total_system_cost_myr']:,.0f} |
        | LCOE | MYR {financial['lcoe_myr_kwh']}/kWh |
        | Simple Payback | {financial['simple_payback_years']} years |
        | 25-Year Energy | {financial['total_lifetime_energy_kwh']:,.0f} kWh |
        | 25-Year Savings | MYR {financial['total_lifetime_savings_myr']:,.0f} |
        | 25-Year Net Profit | MYR {financial['total_lifetime_profit_myr']:,.0f} |
        | ROI | {financial['roi_pct']}% |
        """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "*PV Layout Design System v1.0 | "
    "Optimised for Malaysian Rooftop Solar (ATAP) | "
    "620Wp Module Default*"
)
