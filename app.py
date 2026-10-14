import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, MiniMap
import plotly.graph_objects as go
import plotly.express as px
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bakawala — Mangrove Loss Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL STYLES
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
h1, h2, h3 {
    font-family: 'Georgia', serif;
    letter-spacing: -0.5px;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.bakawala-header {
    background: linear-gradient(-45deg, #1b4332, #2d6a4f, #52b788, #95d5b2);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite;
    padding: 40px 45px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(27,67,50,0.25);
}
.bakawala-header h1 {
    color: white;
    font-size: 52px;
    margin: 0;
    font-weight: 900;
    letter-spacing: -1.5px;
}
.bakawala-header p {
    color: rgba(255,255,255,0.92);
    font-size: 17px;
    margin: 10px 0 0 0;
}
.bakawala-header .tagline {
    font-style: italic;
    color: rgba(255,255,255,0.75);
    font-size: 14px;
    margin-top: 6px;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f7f4 100%);
    padding: 24px 20px;
    border-radius: 14px;
    border: 1px solid #d8f3dc;
    animation: fadeUp 0.6s ease-out;
    transition: all 0.3s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(45,106,79,0.15);
    border-color: #52b788;
}
.metric-card .label {
    font-size: 12px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
.metric-card .value {
    font-size: 34px;
    font-weight: 700;
    color: #2d6a4f;
    margin-top: 8px;
    line-height: 1;
}
.divider {
    display: flex;
    align-items: center;
    margin: 30px 0;
}
.divider .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, transparent, #52b788, transparent);
}
.divider .leaf {
    margin: 0 15px;
    color: #52b788;
    font-size: 18px;
}
.info-banner {
    background: linear-gradient(135deg, #d8f3dc 0%, #e8f5e9 100%);
    padding: 18px 22px;
    border-radius: 12px;
    border-left: 5px solid #2d6a4f;
    color: #1b4332;
    margin: 16px 0;
}
.float-btn {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: #2d6a4f;
    color: white !important;
    padding: 14px 18px;
    border-radius: 50%;
    text-decoration: none;
    font-size: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    z-index: 9999;
}
.sidebar-brand {
    text-align: center;
    padding: 20px 10px 15px 10px;
}
.sidebar-brand .badge {
    display: inline-block;
    padding: 6px 14px;
    background: #2d6a4f;
    color: white;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
}
.sidebar-brand h2 {
    color: #2d6a4f;
    margin: 12px 0 4px 0;
    font-size: 28px;
}
.sidebar-brand p {
    color: #666;
    font-size: 11px;
    line-height: 1.5;
    margin: 0;
}
.sidebar-brand .tagline {
    color: #888;
    font-size: 10px;
    font-style: italic;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_priority_sites():
    df = pd.read_csv('data/priority_sites.csv')
    df['adm4_en'] = df['adm4_en'].astype(str).str.strip()
    df['adm3_pcode'] = df['adm3_pcode'].astype(str).str.strip()
    if 'lon' not in df.columns:
        coords = pd.read_csv('data/bgy_coords.csv')
        coords['adm3_pcode'] = coords['adm3_pcode'].astype(str).str.strip()
        mun = coords.groupby('adm3_pcode')[['lon', 'lat']].mean().reset_index()
        df = df.merge(mun, on='adm3_pcode', how='left')
    df['lon'] = df['lon'].fillna(122.0)
    df['lat'] = df['lat'].fillna(12.0)
    return df

@st.cache_data
def load_top10():
    return pd.read_csv('data/top10_sites.csv')

@st.cache_data
def load_top100():
    return pd.read_csv('data/top100_sites.csv')

@st.cache_data
def load_cost_benefit():
    return pd.read_csv('data/cost_benefit.csv')

try:
    priority_df = load_priority_sites()
    top10_df = load_top10()
    top100_df = load_top100()
    cb_df = load_cost_benefit()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Data loading error: {e}")
    data_loaded = False
    priority_df = pd.DataFrame()
    top10_df = pd.DataFrame()
    top100_df = pd.DataFrame()
    cb_df = pd.DataFrame()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="badge">🇵🇭 PHILIPPINES</div>
    <h2>🌱 Bakawala</h2>
    <p>A Physics-Informed Ensemble Framework<br>for Mangrove Loss Prediction</p>
    <div class="tagline">bakawan (mangrove) + wala (gone)</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home",
     "🗺️ Priority Map (2020)",
     "🩺 Health Maps (1996–2026)",
     "🔮 Future Predictions (2030–2040)",
     "💰 Cost-Benefit",
     "📊 Model Info",
     "📥 Download"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:12px;line-height:1.6;color:#555;">
<b style="color:#2d6a4f;">Researcher</b><br>
Mary Isabelle M. Ibanez<br><br>
<b style="color:#2d6a4f;">Adviser</b><br>
Osmerando P. Alcantara Jr.<br><br>
<b style="color:#2d6a4f;">School</b><br>
Tanauan City Integrated High School<br><br>
<b style="color:#2d6a4f;">Category</b><br>
Mathematical and Computational Science
</div>
""", unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="bakawala-header">
        <h1>🌱 Bakawala</h1>
        <p>A Physics-Informed Ensemble Framework for Mangrove Loss Prediction</p>
        <div class="tagline">bakawan (mangrove) + wala (gone)</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="label">Barangays</div><div class="value">3,853</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="label">Model AUC</div><div class="value">0.732</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="label">Temporal AUC</div><div class="value">0.708</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="label">ROI</div><div class="value">41:1</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
    <b>Bakawala</b> — from the Filipino words <em>bakawan</em> (mangrove) and 
    <em>wala</em> (gone). The name reflects the reality this framework addresses: 
    the progressive loss of Philippine mangroves, and the restoration opportunity 
    that remains.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### 🎯 What Bakawala Does")
        st.markdown("""
        This framework predicts **severe mangrove loss** across Philippine coastal 
        barangays using a **physics-informed ensemble machine learning model** 
        trained on six freely available satellite datasets:
        
        - 🛰️ **Global Mangrove Watch v3.0** — mangrove extent
        - 🌊 **ERA5** — wave physics
        - 👥 **WorldPop** — population exposure
        - 🗺️ **GEBCO 2024** — bathymetry
        - ⛰️ **SRTM** — elevation and slope
        - 📍 **PSA** — barangay boundaries
        """)

    with col_right:
        st.markdown("### 🔑 Key Findings")
        st.markdown("""
        | Metric | Value |
        |---|---|
        | Model AUC | **0.732** |
        | Temporal AUC | **0.708** |
        | Top priority sites | **Pampanga & Bataan** |
        | Top 10 restoration | **34.90 ha** |
        | Top 10 investment | **₱2.96M** |
        | Top 10 benefit | **₱122.25M** |
        | **ROI** | **41.3 : 1** |
        """)

    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    st.markdown("### 🧭 Explore the Framework")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.info("🗺️ **Priority Map**\n\nSee the top 100 restoration sites on an interactive map.")
    with exp_col2:
        st.info("🩺 **Health Maps**\n\nMangrove health from 1996 to 2026 across the archipelago.")
    with exp_col3:
        st.info("🔮 **Future Predictions**\n\nProjected 2030 and 2040 scenarios under IPCC AR6.")

# ============================================================
# PRIORITY MAP
# ============================================================
elif page == "🗺️ Priority Map (2020)":
    st.markdown("""<h1 style="color:#2d6a4f;">🗺️ Mangrove Restoration Priority Map</h1>""", unsafe_allow_html=True)
    st.markdown("""<p style="color:#666;">Top priority sites ranked by composite priority score.</p>""", unsafe_allow_html=True)
    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Data not loaded.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Show Top N Sites", 10, 100, 50, step=10)
        with col2:
            view_mode = st.selectbox("View Mode", ["All Sites", "Top 10 Only", "Top 50 Only"])

        if view_mode == "Top 10 Only":
            top_n = 10
        elif view_mode == "Top 50 Only":
            top_n = 50

        with st.spinner("🗺️ Rendering map..."):
            sites = priority_df.nlargest(top_n, 'priority') if 'priority' in priority_df.columns else priority_df.head(top_n)

            m = folium.Map(
                location=[12.8797, 121.7740],
                zoom_start=6,
                tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
                attr='OpenStreetMap HOT'
            )
            Fullscreen().add_to(m)
            MiniMap(toggle_display=True, position='bottomright').add_to(m)

            for _, row in sites.iterrows():
                rank = row.get('rank', 0)
                if rank <= 10:
                    color, radius = '#d73027', 13
                elif rank <= 30:
                    color, radius = '#fc8d59', 10
                elif rank <= 60:
                    color, radius = '#fee08b', 7
                else:
                    color, radius = '#91cf60', 5

                popup_html = f"""
                <div style="font-family:Inter,sans-serif;font-size:12px;line-height:1.5;">
                <b style="font-size:14px;color:#2d6a4f;">#{rank}: {row.get('adm4_en', 'N/A')}</b><br>
                <hr style="margin:6px 0;">
                <b>Population:</b> {row.get('population', 0):,.0f}<br>
                <b>Loss Probability:</b> {row.get('loss_probability', 0):.1%}<br>
                <b>Priority Score:</b> {row.get('priority', 0):.3f}
                </div>
                """

                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=radius, color=color, fill=True, fill_opacity=0.75,
                    popup=folium.Popup(popup_html, max_width=280),
                    tooltip=f"#{rank} {row.get('adm4_en', '')}"
                ).add_to(m)

        st_folium(m, width=None, height=600, use_container_width=True)

        st.markdown("""
        <div style="background:#f8f9fa;padding:15px;border-radius:10px;
             border-left:4px solid #2d6a4f;margin-top:12px;">
            <b style="font-size:14px;color:#2d6a4f;">Priority Legend</b><br><br>
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                 background:#d73027;margin-right:8px;"></span> Top 10 (Critical)
            &nbsp;&nbsp;&nbsp;
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                 background:#fc8d59;margin-right:8px;"></span> Rank 11–30
            &nbsp;&nbsp;&nbsp;
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                 background:#fee08b;margin-right:8px;"></span> Rank 31–60
            &nbsp;&nbsp;&nbsp;
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                 background:#91cf60;margin-right:8px;"></span> Rank 61–100
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)
        st.markdown(f"### 📋 Top {top_n} Priority Sites")
        display_cols = ['rank', 'adm4_en', 'adm3_pcode', 'population', 'loss_probability', 'priority']
        display_cols = [c for c in display_cols if c in sites.columns]
        st.dataframe(sites[display_cols].round(3), use_container_width=True)

# ============================================================
# HEALTH MAPS
# ============================================================
elif page == "🩺 Health Maps (1996–2026)":
    st.markdown("""<h1 style="color:#2d6a4f;">🩺 Mangrove Health Trajectory</h1>""", unsafe_allow_html=True)
    st.markdown("""<p style="color:#666;">Visualizing mangrove health across the Philippines over 30 years.</p>""", unsafe_allow_html=True)
    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    year = st.selectbox("Select Year", ["1996", "2007", "2015", "2020", "2026 (Real)"])

    map_files = {
        "1996": "assets/assets_health_map_1996.png",
        "2007": "assets/assets_health_map_2007.png",
        "2015": "assets/assets_health_map_2015.png",
        "2020": "assets/assets_health_map_2020.png",
        "2026 (Real)": "assets/health_map_2026.png",
    }

    file_path = map_files[year]
    with st.spinner(f"🩺 Loading {year} health map..."):
        if os.path.exists(file_path):
            st.image(file_path, use_container_width=True)
        else:
            st.warning(f"⚠️ Map not found: {file_path}")

    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🌿 Health Categories")
        st.markdown("""
        - 🟢 **Healthy (80–100)** — stable or recovering mangrove
        - 🟢 **Moderate (60–79)** — mild decline
        - 🟠 **At Risk (40–59)** — consistent decline
        - 🔴 **Critical (0–39)** — severe loss
        """)

    with col2:
        st.markdown("### 📊 National Trend")
        trend_df = pd.DataFrame({
            'Year': [1996, 2007, 2015, 2020],
            'Extent (ha)': [127952, 124378, 121469, 122930]
        })
        fig = px.bar(
            trend_df, x='Year', y='Extent (ha)',
            color='Extent (ha)',
            color_continuous_scale=['#1b4332', '#2d6a4f', '#52b788', '#95d5b2'],
        )
        fig.update_layout(showlegend=False, plot_bgcolor='white', height=300,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **What this chart shows:**  
    Total mangrove extent in the Philippines from 1996 to 2020.

    **Key observations:**

    - **1996 (127,952 ha):** Baseline — start of the study period
    - **2007 (124,378 ha):** −2.8% decline over 11 years
    - **2015 (121,469 ha):** −5.1% decline — the lowest point
    - **2020 (122,930 ha):** −3.9% — a **+1,461 ha recovery** from 2015

    **Interpretation:**

    1. **Downward trend (1996 → 2015):** Philippine mangroves declined by over 
    6,000 hectares over 19 years due to aquaculture, coastal development, and 
    land conversion.

    2. **Notable recovery (2015 → 2020):** Mangrove extent increased by 1,461 
    hectares — the first time in the study period that mangroves grew back. 
    This coincides with the DENR National Greening Program and the Philippine 
    Mangrove and Beach Forest Development Project.

    3. **Net loss (1996 → 2020):** Despite the recovery, the Philippines still 
    lost **5,022 hectares** — a net decline of **3.9%**.

    **What this means for policymakers:** Restoration is working, but continued 
    investment is needed to reverse the historical decline.
    """)

# ============================================================
# FUTURE PREDICTIONS
# ============================================================
elif page == "🔮 Future Predictions (2030–2040)":
    st.markdown("""<h1 style="color:#2d6a4f;">🔮 Future Mangrove Predictions</h1>""", unsafe_allow_html=True)
    st.markdown("""<p style="color:#666;">Projected scenarios under IPCC AR6 SSP2-4.5 for the Philippines.</p>""", unsafe_allow_html=True)
    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📅 2030 Projection", "📅 2040 Projection"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🗺️ Priority Map")
            if os.path.exists('assets/priority_map_2030.png'):
                st.image('assets/priority_map_2030.png', use_container_width=True)
            else:
                st.warning("⚠️ `assets/priority_map_2030.png` not found")
        with col2:
            st.markdown("#### 🩺 Health Map")
            if os.path.exists('assets/health_map_2030.png'):
                st.image('assets/health_map_2030.png', use_container_width=True)
            else:
                st.warning("⚠️ `assets/health_map_2030.png` not found")

        st.markdown("#### ⚙️ 2030 Parameters")
        p1, p2, p3 = st.columns(3)
        p1.metric("Wave Energy", "+3%")
        p2.metric("Population", "+10%")
        p3.metric("Sea-Level Rise", "+0.10 m")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🗺️ Priority Map")
            if os.path.exists('assets/priority_map_2040.png'):
                st.image('assets/priority_map_2040.png', use_container_width=True)
            else:
                st.warning("⚠️ `assets/priority_map_2040.png` not found")
        with col2:
            st.markdown("#### 🩺 Health Map")
            if os.path.exists('assets/health_map_2040.png'):
                st.image('assets/health_map_2040.png', use_container_width=True)
            else:
                st.warning("⚠️ `assets/health_map_2040.png` not found")

        st.markdown("#### ⚙️ 2040 Parameters")
        p1, p2, p3 = st.columns(3)
        p1.metric("Wave Energy", "+7%")
        p2.metric("Population", "+20%")
        p3.metric("Sea-Level Rise", "+0.22 m")

    st.warning("⚠️ **Note:** Projections use simplified scaling factors based on IPCC AR6 SSP2-4.5 regional estimates. Interpret as indicative scenarios, not forecasts.")

# ============================================================
# COST-BENEFIT
# ============================================================
elif page == "💰 Cost-Benefit":
    st.markdown("""<h1 style="color:#2d6a4f;">💰 Cost-Benefit Calculator</h1>""", unsafe_allow_html=True)
    st.markdown("""<p style="color:#666;">Compute restoration ROI for any site or budget.</p>""", unsafe_allow_html=True)
    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Data not loaded.")
    else:
        if not cb_df.empty:
            st.markdown("### 📊 Summary — Top 10 vs Top 100")
            st.dataframe(cb_df.round(2), use_container_width=True)

        st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)
        st.markdown("### 🧮 Custom Scenario Calculator")

        col1, col2 = st.columns(2)
        with col1:
            hectares = st.number_input("Restoration area (hectares)", min_value=1.0, max_value=5000.0, value=61.45, step=1.0)
        with col2:
            exchange = st.number_input("Exchange rate (₱ per USD)", min_value=50.0, max_value=65.0, value=56.50, step=0.50)

        RESTORATION_COST = 1500
        PROTECTION_VALUE = 50000
        CARBON_VALUE = 12000
        PEOPLE_PER_HA = 20

        investment_usd = hectares * RESTORATION_COST
        protection_usd = hectares * PROTECTION_VALUE
        carbon_usd = hectares * CARBON_VALUE
        total_benefit_usd = protection_usd + carbon_usd
        people_protected = hectares * PEOPLE_PER_HA
        bcr = total_benefit_usd / investment_usd if investment_usd > 0 else 0

        st.markdown("### 📈 Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="label">Investment</div><div class="value">₱{investment_usd * exchange:,.0f}</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="label">20-year Benefit</div><div class="value">₱{total_benefit_usd * exchange:,.0f}</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="label">ROI</div><div class="value">{bcr:.1f} : 1</div></div>""", unsafe_allow_html=True)

        st.markdown("### 📋 Detailed Breakdown")
        breakdown = pd.DataFrame({
            'Category': ['Restoration Investment', 'Flood Protection Value', 'Blue Carbon Value', 'Total Benefit', 'People Protected'],
            'USD': [f"${investment_usd:,.0f}", f"${protection_usd:,.0f}", f"${carbon_usd:,.0f}", f"${total_benefit_usd:,.0f}", f"{people_protected:,.0f}"],
            'PHP': [f"₱{investment_usd*exchange:,.0f}", f"₱{protection_usd*exchange:,.0f}", f"₱{carbon_usd*exchange:,.0f}", f"₱{total_benefit_usd*exchange:,.0f}", f"{people_protected:,.0f}"]
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

        fig = go.Figure(data=[
            go.Bar(name='Investment', x=['Cost'], y=[investment_usd * exchange], marker_color='#d73027'),
            go.Bar(name='20-yr Benefit', x=['Benefit'], y=[total_benefit_usd * exchange], marker_color='#2d6a4f')
        ])
        fig.update_layout(title="Investment vs Return", yaxis_title="₱ (PHP)", height=400,
                          plot_bgcolor='white', title_font=dict(size=18, color='#2d6a4f'))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODEL INFO
# ============================================================
elif page == "📊 Model Info":
    st.markdown("""<h1 style="color:#2d6a4f;">📊 Model Performance & Interpretability</h1>""", unsafe_allow_html=True)
    st.markdown("""<p style="color:#666;">Validation, feature ablation, and interpretability results.</p>""", unsafe_allow_html=True)
    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    st.markdown("### 🎯 Cross-Validated Model Performance")
    perf_df = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Ensemble "Bakawala"'],
        'AUC': [0.726, 0.718, 0.729, 0.727, 0.732]
    })

    fig = px.bar(perf_df, x='Model', y='AUC', color='AUC',
                 color_continuous_scale=['#95d5b2', '#52b788', '#2d6a4f', '#1b4332'],
                 text='AUC')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(showlegend=False, plot_bgcolor='white', height=350,
                      yaxis_range=[0.68, 0.76])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **What AUC means:**  
    AUC measures how well the model ranks a high-loss barangay above a low-loss 
    one. Random guessing = 0.50; perfect prediction = 1.00.

    **Interpretation of each model:**

    - **XGBoost (0.726):** Strong baseline — correctly ranks 72.6% of pairs
    - **LightGBM (0.718):** Slightly lower — leaf-wise growth overfit
    - **CatBoost (0.729):** Best individual — better calibrated probabilities
    - **Random Forest (0.727):** Stable bagging-based model
    - **Ensemble "Bakawala" (0.732):** **Best overall** — combining all 4 improves accuracy

    **Key insight:** All four models cluster tightly (0.718–0.729), meaning the 
    predictive signal is **real** — not an artifact of any single algorithm. 
    The ensemble exceeds every individual model, proving that **algorithmic 
    diversity reduces variance** and improves generalization.
    """)

    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    st.markdown("### ✅ Validation Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card"><div class="label">AUC (95% CI)</div><div class="value">0.732</div><div style="font-size:12px;color:#666;margin-top:6px;">0.705 – 0.748</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="label">Temporal AUC</div><div class="value">0.708</div><div style="font-size:12px;color:#666;margin-top:6px;">Unseen future data</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="label">CV Inflation</div><div class="value">+0.016</div><div style="font-size:12px;color:#666;margin-top:6px;">Random vs Spatial</div></div>""", unsafe_allow_html=True)

    st.markdown("""
    **What each metric means:**

    - **AUC = 0.732 (95% CI: 0.705–0.748):** Based on 1,000 bootstrap resamples. 
    The model correctly ranks high-loss above low-loss barangays 73.2% of the 
    time. The 95% CI is the honest uncertainty range — wide enough to be 
    defensible, tight enough to be useful.

    - **Temporal AUC = 0.708:** Trained on 1996–2007 loss, tested on 2007–2020 
    loss. Retains **96% of original performance** on unseen future data. This 
    proves the model learned transferable rules, not just memorized patterns.

    - **CV Inflation = +0.016:** Random CV gave AUC = 0.721; spatial CV gave 
    0.704. Only **1.6 points of inflation** — versus **~10 points** typical in 
    pixel-based land cover studies. This confirms physics features capture 
    generalizable environmental signals.
    """)

    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    st.markdown("### 🧪 Feature Ablation")
    ablation = pd.DataFrame({
        'Removed Group': ['Baseline Extent', 'Wave Exposure', 'Bathymetry', 'Population'],
        'ΔAUC': [-0.043, -0.020, -0.011, -0.007]
    })

    fig2 = px.bar(ablation, x='ΔAUC', y='Removed Group', orientation='h',
                  color='ΔAUC',
                  color_continuous_scale=['#d73027', '#fc8d59', '#fee08b', '#91cf60'],
                  text='ΔAUC')
    fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig2.update_layout(showlegend=False, plot_bgcolor='white', height=300)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **How to read this chart:**  
    Each bar shows how much AUC drops when that feature group is removed. 
    **Bigger drops = more important features.**

    **Interpretation:**

    - **Baseline Extent (ΔAUC = −0.043):** The **strongest predictor**. This 
    confirms the **"size-protects" finding** — large, dense mangrove patches 
    resist disturbance better than small, fragmented ones.

    - **Wave Exposure (ΔAUC = −0.020):** Physical stress from ocean waves is 
    the second most important driver of mangrove loss.

    - **Bathymetry (ΔAUC = −0.011):** Coastal geometry contributes moderately.

    - **Population (ΔAUC = −0.007):** Human pressure matters, but is less 
    important than physics.

    **Key insight for policy:** Protecting large, intact mangrove patches is 
    more effective than focusing solely on restoring small, fragmented patches.
    """)

    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    st.markdown("### 📉 Model Performance Curves")
    if os.path.exists('assets/performance_curves.png'):
        st.image('assets/performance_curves.png', use_container_width=True)
    else:
        st.info("Add `performance_curves.png` to `assets/` folder.")

# ============================================================
# DOWNLOAD
# ============================================================
elif page == "📥 Download":
    st.markdown("""<h1 style="color:#2d6a4f;">📥 Download Data</h1>""", unsafe_allow_html=True)
    st.markdown("""<p style="color:#666;">Export priority rankings and economic analysis.</p>""", unsafe_allow_html=True)
    st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.error("Data not loaded.")
    else:
        st.markdown("### 🏆 Top 10 Priority Sites")
        st.dataframe(top10_df, use_container_width=True)
        st.download_button("📥 Download Top 10 (CSV)",
            top10_df.to_csv(index=False).encode('utf-8'),
            "top10_priority_sites.csv", "text/csv")

        st.markdown("---")

        st.markdown("### 🎯 Top 100 Priority Sites")
        st.dataframe(top100_df.head(20), use_container_width=True)
        st.markdown(f"*Showing first 20 of {len(top100_df)} sites.*")
        st.download_button("📥 Download Top 100 (CSV)",
            top100_df.to_csv(index=False).encode('utf-8'),
            "top100_priority_sites.csv", "text/csv")

        st.markdown("---")

        st.markdown("### 💰 Cost-Benefit Analysis")
        if not cb_df.empty:
            st.dataframe(cb_df, use_container_width=True)
            st.download_button("📥 Download Cost-Benefit (CSV)",
                cb_df.to_csv(index=False).encode('utf-8'),
                "cost_benefit_analysis.csv", "text/csv")

        st.markdown("---")

        st.markdown("### 📊 Full Priority Site List")
        st.download_button("📥 Download Full Priority List (CSV)",
            priority_df.to_csv(index=False).encode('utf-8'),
            "full_priority_sites.csv", "text/csv")

# ============================================================
# FLOATING BUTTON + FOOTER
# ============================================================
st.markdown('<a href="#top" class="float-btn">⬆️</a>', unsafe_allow_html=True)
st.markdown("""<div class="divider"><div class="line"></div><div class="leaf">🌱</div><div class="line"></div></div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#666;font-size:12px;padding:20px;">
🌱 <b style="color:#2d6a4f;font-size:15px;">Bakawala</b><br>
<em>A Physics-Informed Ensemble Framework for Mangrove Loss Prediction</em><br>
<em style="color:#888;">bakawan (mangrove) + wala (gone)</em><br><br>
Mary Isabelle M. Ibanez | Adviser: Osmerando P. Alcantara Jr.<br>
Tanauan City Integrated High School — STE Program
</div>
""", unsafe_allow_html=True)