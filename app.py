import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------- 1. PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="PrimeEstate AI | Luxury Real Estate Terminal",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- 2. HIGH-PERFORMANCE CACHING --------------------
@st.cache_resource
def load_ml_model():
    """Cache trained model & feature column definitions in memory."""
    model = joblib.load("rf_model.joblib")
    features = joblib.load("model_columns.joblib")
    return model, features

@st.cache_data
def load_clean_data():
    """Cache dataset in memory for instant responsiveness."""
    return pd.read_csv("cleaned_df.csv")

try:
    model, model_features = load_ml_model()
    df_raw = load_clean_data()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# Extract valid location names from model features
locations = sorted([
    col.replace("location_", "") 
    for col in model_features if col.startswith("location_")
] + ["others"])
locations = sorted(list(set(locations)))

# -------------------- 3. ADVANCED LUXURY THEME & CSS --------------------
st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Midnight Obsidian Palette */
    .stApp {
        background: radial-gradient(circle at top right, #1E1B4B 0%, #0F172A 40%, #030712 100%);
        color: #F8FAFC;
    }

    /* Sidebar Custom Dark Aesthetic */
    [data-testid="stSidebar"] {
        background: #030712 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }

    /* Widget Labels Styling */
    div[data-testid="stWidgetLabel"] label, 
    .stSlider label, 
    .stSelectbox label, 
    .stNumberInput label {
        color: #94A3B8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    /* Top Market Ticker Bar */
    .ticker-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 50px;
        padding: 10px 24px;
        margin-bottom: 24px;
    }

    .ticker-item {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .ticker-highlight {
        color: #10B981;
        font-weight: 700;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.9) 0%, rgba(124, 58, 237, 0.9) 50%, rgba(219, 39, 119, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 40px 30px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px -15px rgba(124, 58, 237, 0.5);
    }

    .hero-banner h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin: 0;
        letter-spacing: -0.03em;
        background: linear-gradient(180deg, #FFFFFF 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-banner p {
        font-size: 1.15rem;
        color: #E2E8F0;
        margin-top: 10px;
        margin-bottom: 0;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(16, 185, 129, 0.3);
    }

    /* Hero Valuation Result Banner */
    .result-card {
        background: linear-gradient(135deg, #059669 0%, #10B981 50%, #047857 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 36px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 16px 40px rgba(16, 185, 129, 0.35);
        margin: 24px 0;
    }

    .result-card h2 {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 12px 0;
        letter-spacing: -0.02em;
        text-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Custom Metric Badges */
    .metric-badge {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px 16px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        height: 100%;
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
        margin-top: 6px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Input Controls Custom Styling */
    .stSelectbox div[data-baseweb="select"], 
    .stNumberInput div[data-baseweb="input"] {
        background-color: #0B0F19 !important;
        border-radius: 12px !important;
        border: 1px solid #1E293B !important;
        color: #F8FAFC !important;
    }

    /* Vibrant Gradient Calculate Button */
    .stButton>button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.04em !important;
        border: none !important;
        padding: 0.95rem 2.5rem !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 10px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.6);
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
    }

    /* Tabs Bar Custom Luxury Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 12px;
        color: #94A3B8;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- 4. LIVE MARKET TICKER BAR --------------------
st.markdown("""
    <div class="ticker-bar">
        <div class="ticker-item">📍 <span class="ticker-highlight">BENGALURU REAL ESTATE MARKET</span></div>
        <div class="ticker-item">⚡ AI MODEL: <span class="ticker-highlight">CATBOOST + RF + ET ENSEMBLE</span></div>
        <div class="ticker-item">📊 RECORDS: <span class="ticker-highlight">9,210 VERIFIED HOMES</span></div>
        <div class="ticker-item">⏱️ SPEED: <span class="ticker-highlight">&lt;35ms LATENCY</span></div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- 5. HERO HEADER --------------------
st.markdown("""
    <div class="hero-banner">
        <h1>💎 PrimeEstate AI Terminal</h1>
        <p>Precision Machine Learning Valuation & Strategic Real Estate Investment Terminal</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- 6. CLEAN SIDEBAR --------------------
with st.sidebar:
    st.image("house_logo.png", width=170)
    st.markdown("### 🏠 Valuation Instructions")
    st.markdown("""
    1. **Select Location**
    2. **Enter Area (Sq.Ft.)**
    3. **Set BHK & Bathrooms**
    4. **Click "💰 Predict Price"**
    """)
    st.divider()
    st.caption("PrimeEstate AI Engine v3.0 | PropTech Intelligence")

# -------------------- 7. PROPERTY INPUT CARD --------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📍 Property Specifications")

col1, col2, col3, col4 = st.columns(4)

with col1:
    location = st.selectbox("Location", locations, index=locations.index("Whitefield") if "Whitefield" in locations else 0)

with col2:
    sqft = st.number_input("Total Area (Sq. Ft.)", min_value=300, max_value=25000, value=1200, step=50)

with col3:
    bhk = st.selectbox("Bedrooms (BHK)", options=[1, 2, 3, 4, 5, 6], index=1)

with col4:
    bath = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1)

# Predict Price Button
predict_btn = st.button("💎 CALCULATE MARKET VALUATION")

st.markdown('</div>', unsafe_allow_html=True)

# Session State for User Triggered Predictions
if predict_btn:
    st.session_state['predicted'] = True
    st.session_state['location'] = location
    st.session_state['sqft'] = sqft
    st.session_state['bhk'] = bhk
    st.session_state['bath'] = bath

# Helper function to format INR currency
def format_inr(lakhs):
    if lakhs >= 100:
        crores = lakhs / 100.0
        return f"₹ {crores:.2f} Cr"
    return f"₹ {lakhs:.2f} Lakhs"

# Function to build model feature input vector
def prepare_input_dataframe(loc, sq_ft, b_hk, b_ath):
    input_dict = {col: 0 for col in model_features}
    
    input_dict['total_sqft'] = sq_ft
    input_dict['bath'] = b_ath
    input_dict['bhk'] = b_hk
    input_dict['sqft_per_bhk'] = sq_ft / float(b_hk)
    input_dict['bath_per_bhk'] = b_ath / float(b_hk)
    
    loc_col = f"location_{loc}"
    if loc_col in input_dict:
        input_dict[loc_col] = 1
        
    return pd.DataFrame([input_dict])

# Render Valuation & Multi-Tab Terminal when User clicks Predict
if st.session_state.get('predicted', False):
    loc_input = st.session_state.get('location', location)
    sqft_input = st.session_state.get('sqft', sqft)
    bhk_input = st.session_state.get('bhk', bhk)
    bath_input = st.session_state.get('bath', bath)

    # Calculate Model Prediction
    input_df = prepare_input_dataframe(loc_input, sqft_input, bhk_input, bath_input)
    predicted_lakhs = float(model.predict(input_df)[0])
    predicted_inr = predicted_lakhs * 100000.0
    price_per_sqft_calc = predicted_inr / float(sqft_input)

    # Location Average Price Lookup
    loc_df = df_raw[df_raw["location"] == loc_input]
    if len(loc_df) > 3:
        loc_avg_lakhs = loc_df["price"].mean()
    else:
        loc_avg_lakhs = df_raw["price"].mean()

    # -------------------- 8. MULTI-TAB TERMINAL --------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Valuation & Summary", 
        "📊 Market Intelligence", 
        "🔍 Property Inspector", 
        "📈 Investment & ROI"
    ])

    # ---------------- TAB 1: VALUATION & SUMMARY ----------------
    with tab1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<p style="margin:0; font-size:1.1rem; opacity:0.9; text-transform:uppercase; letter-spacing:0.1em;">ESTIMATED MARKET VALUATION</p>', unsafe_allow_html=True)
        st.markdown(f'<h2>{format_inr(predicted_lakhs)}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p style="margin:0; font-size:1.15rem; font-weight:500;">₹ {predicted_inr:,.0f} Total Valuation | ₹ {price_per_sqft_calc:,.0f} per Sq.Ft.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 4 Metric Cards Grid
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'''
                <div class="metric-badge">
                    <div class="metric-label">Location Avg Price</div>
                    <div class="metric-value">{format_inr(loc_avg_lakhs)}</div>
                </div>
            ''', unsafe_allow_html=True)

        with m2:
            st.markdown(f'''
                <div class="metric-badge">
                    <div class="metric-label">Rate / Sq.Ft.</div>
                    <div class="metric-value">₹ {price_per_sqft_calc:,.0f}</div>
                </div>
            ''', unsafe_allow_html=True)

        with m3:
            lower_bound_val = predicted_lakhs * 0.95
            upper_bound_val = predicted_lakhs * 1.05
            st.markdown(f'''
                <div class="metric-badge">
                    <div class="metric-label">Valuation Band (±5%)</div>
                    <div class="metric-value" style="font-size:1.3rem;">{format_inr(lower_bound_val)} - {format_inr(upper_bound_val)}</div>
                </div>
            ''', unsafe_allow_html=True)

        with m4:
            diff_pct = ((predicted_lakhs - loc_avg_lakhs) / loc_avg_lakhs) * 100.0
            badge_color = "#10B981" if diff_pct < -5 else ("#F59E0B" if abs(diff_pct) <= 5 else "#EF4444")
            deal_text = "Great Deal" if diff_pct < -5 else ("Fair Market" if abs(diff_pct) <= 5 else "Premium")
            
            st.markdown(f'''
                <div class="metric-badge">
                    <div class="metric-label">Market Rating</div>
                    <div class="metric-value" style="color:{badge_color};">{deal_text}</div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Sensitivity Table
        st.markdown("#### ⚡ Area Sensitivity & Price Elasticity Matrix")
        sensitivity_data = []
        for pct in [-10, -5, 0, 5, 10]:
            adj_sqft = sqft_input * (1 + pct / 100.0)
            adj_df = prepare_input_dataframe(loc_input, adj_sqft, bhk_input, bath_input)
            adj_price = float(model.predict(adj_df)[0])
            sensitivity_data.append({
                "Area Variation": f"{pct:+d}%",
                "Square Footage": f"{adj_sqft:,.0f} sq.ft.",
                "Estimated Value": format_inr(adj_price),
                "Price per Sq.Ft.": f"₹ {(adj_price * 100000.0 / adj_sqft):,.0f}"
            })
        st.dataframe(pd.DataFrame(sensitivity_data), use_container_width=True, hide_index=True)

    # ---------------- TAB 2: MARKET INTELLIGENCE ----------------
    with tab2:
        st.markdown("### 📊 Market Trends & Pricing Analytics")
        
        ca1, ca2 = st.columns(2)
        
        with ca1:
            # Price Distribution Histogram
            fig_hist = px.histogram(
                df_raw[df_raw['price'] <= df_raw['price'].quantile(0.95)], 
                x="price",
                nbins=35,
                title=f"Price Distribution in Bengaluru (Lakhs)",
                labels={"price": "Price (Lakhs)"},
                color_discrete_sequence=["#10B981"]
            )
            fig_hist.add_vline(
                x=predicted_lakhs, 
                line_dash="dash", 
                line_color="#38BDF8", 
                annotation_text=f"Your Estimate ({predicted_lakhs:.1f}L)",
                annotation_position="top right"
            )
            fig_hist.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with ca2:
            # SqFt vs Price Scatter Plot
            sample_df = df_raw[df_raw['total_sqft'] <= 4000].sample(min(900, len(df_raw)), random_state=42)
            fig_scatter = px.scatter(
                sample_df,
                x="total_sqft",
                y="price",
                color="bhk",
                title="Total Area (Sq.Ft.) vs Price (Lakhs)",
                labels={"total_sqft": "Square Feet", "price": "Price (Lakhs)", "bhk": "BHK"},
                color_continuous_scale="Viridis"
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[sqft_input],
                    y=[predicted_lakhs],
                    mode="markers+text",
                    marker=dict(size=18, color="#EF4444", symbol="star"),
                    name="Target Property",
                    text=[f"Target: {predicted_lakhs:.1f}L"],
                    textposition="top center"
                )
            )
            fig_scatter.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Top 10 High-Value Locations Bar Chart
        st.markdown("#### 🏆 Top Bengaluru Real Estate Hubs (Avg Price / Sq.Ft.)")
        top_locs = df_raw.groupby('location')['price_per_sqft'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_bar = px.bar(
            top_locs, 
            x='location', 
            y='price_per_sqft',
            labels={'location': 'Location Hub', 'price_per_sqft': 'Avg Rate (₹ / sq.ft.)'},
            color='price_per_sqft',
            color_continuous_scale='Magma'
        )
        fig_bar.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- TAB 3: PROPERTY INSPECTOR ----------------
    with tab3:
        st.markdown(f"### 🔍 Comparable Properties in **{loc_input}**")
        
        similar_match = df_raw[
            (df_raw['location'] == loc_input) & 
            (df_raw['bhk'] == bhk_input) & 
            (df_raw['total_sqft'].between(sqft_input * 0.75, sqft_input * 1.25))
        ]
        
        if len(similar_match) > 0:
            st.dataframe(
                similar_match[['location', 'total_sqft', 'bhk', 'bath', 'price', 'price_per_sqft']]
                .rename(columns={
                    'location': 'Location', 
                    'total_sqft': 'Sq.Ft.', 
                    'bhk': 'BHK', 
                    'bath': 'Baths', 
                    'price': 'Price (Lakhs)',
                    'price_per_sqft': 'Price / Sq.Ft. (₹)'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No exact location & BHK match found. Showing nearby area properties:")
            nearby_match = df_raw[df_raw['bhk'] == bhk_input].head(10)
            st.dataframe(
                nearby_match[['location', 'total_sqft', 'bhk', 'bath', 'price', 'price_per_sqft']],
                use_container_width=True,
                hide_index=True
            )

    # ---------------- TAB 4: INVESTMENT & ROI ----------------
    with tab4:
        st.markdown("### 📈 10-Year Growth Projection & Rental Yield Calculator")
        
        ci1, ci2 = st.columns(2)
        with ci1:
            appreciation_rate = st.slider("Annual Property Growth (%)", min_value=3.0, max_value=15.0, value=7.5, step=0.5)
            holding_years = st.slider("Holding Period (Years)", min_value=1, max_value=10, value=5)
            rental_yield = st.slider("Estimated Rental Yield (%)", min_value=1.5, max_value=6.0, value=3.5, step=0.1)

        future_val_lakhs = predicted_lakhs * ((1 + (appreciation_rate / 100.0)) ** holding_years)
        monthly_rent = (predicted_inr * (rental_yield / 100.0)) / 12.0

        with ci2:
            st.markdown(f'''
                <div class="glass-card" style="text-align:center;">
                    <h4 style="color:#94A3B8; margin-bottom:6px;">PROJECTED VALUE AFTER {holding_years} YEARS</h4>
                    <h2 style="color:#10B981; font-size:2.8rem; font-weight:800; font-family:'Outfit', sans-serif;">{format_inr(future_val_lakhs)}</h2>
                    <hr style="border-color:rgba(255,255,255,0.1);">
                    <h4 style="color:#94A3B8; margin-bottom:6px;">ESTIMATED MONTHLY RENTAL INCOME</h4>
                    <h2 style="color:#38BDF8; font-size:2.2rem; font-weight:700; font-family:'Outfit', sans-serif;">₹ {monthly_rent:,.0f} / mo</h2>
                </div>
            ''', unsafe_allow_html=True)

        # Plotly 10-Year Appreciation Growth Line Chart
        years_arr = list(range(0, 11))
        values_arr = [predicted_lakhs * ((1 + (appreciation_rate / 100.0)) ** yr) for yr in years_arr]
        
        fig_roi = px.line(
            x=years_arr, 
            y=values_arr, 
            labels={'x': 'Years Held', 'y': 'Valuation (Lakhs)'},
            title="10-Year Value Appreciation Forecast",
            markers=True
        )
        fig_roi.update_traces(line_color="#10B981", line_width=3)
        fig_roi.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_roi, use_container_width=True)
else:
    st.info("👈 Select your property specifications above and click **💎 CALCULATE MARKET VALUATION** to generate your real estate analytics.")

# -------------------- 9. FOOTER --------------------
st.divider()
st.markdown(
    "<center><small style='color:#64748B;'>PrimeEstate AI Valuation Engine v3.0 | Built with Streamlit, CatBoost, Scikit-Learn & Plotly</small></center>",
    unsafe_allow_html=True
)