"""
Professional Data Cleaning Suite
A user-friendly web interface for cleaning CSV files and detecting duplicates.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
from data_cleaner import (
    DataCleaner, 
    ML_AVAILABLE, 
    ML_IMPORT_ERROR, 
    RECORDLINKAGE_AVAILABLE,
    RECORDLINKAGE_IMPORT_ERROR,
    PERFORMANCE_LIBS_AVAILABLE,
    RETAIL_LIBS_AVAILABLE
)

# Application Constants
APP_TITLE = "Professional Data Cleaning Suite"
APP_ICON = "📊"

# Performance estimation constants (based on benchmarking with typical hardware)
# Time estimates are per 1000 records for user messaging
# These are conservative estimates to set realistic expectations
PERF_ESTIMATE_MIN_PER_1K = 1  # Minimum seconds per 1000 records
PERF_ESTIMATE_MAX_PER_1K = 2  # Maximum seconds per 1000 records
LARGE_DATASET_WARNING_THRESHOLD = 10000  # Show warnings above this threshold

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional UI
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main container - clean white background */
    .main {
        background: #ffffff;
        padding: 2rem;
    }
    
    /* Content wrapper for better structure */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Sidebar styling - clean light background */
    section[data-testid="stSidebar"] {
        background: #f8f9fa !important;
        border-right: 1px solid #e0e0e0;
    }
    
    section[data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #1a252f !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #2c3e50 !important;
    }
    
    /* Navigation items styling - clean design */
    section[data-testid="stSidebar"] .stRadio > label {
        background: #ffffff !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin: 6px 0 !important;
        transition: all 0.2s ease !important;
        border: 1px solid #e0e0e0 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #2c3e50 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label * {
        color: #2c3e50 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label:hover {
        background: #f0f4ff !important;
        border-color: #2563eb !important;
        transform: translateX(4px) !important;
    }
    
    /* Active navigation item - clean blue accent */
    section[data-testid="stSidebar"] .stRadio > label[data-checked="true"] {
        background: #2563eb !important;
        border-color: #2563eb !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label[data-checked="true"] * {
        color: #ffffff !important;
    }
    
    /* Header styles - clean, readable */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding: 1rem 1.5rem;
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
    }
    
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Card styles - clean white cards */
    .card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Metric cards - clean design */
    .metric-card {
        background: #2563eb;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #dbeafe;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Alert boxes with better visibility */
    .stAlert {
        border-radius: 8px;
        border: 1px solid;
        padding: 1rem 1.25rem;
        font-weight: 500;
    }
    
    /* Info alerts */
    div[data-baseweb="notification"][kind="info"] {
        background-color: #eff6ff !important;
        border-color: #3b82f6 !important;
        color: #1e40af !important;
    }
    
    div[data-baseweb="notification"][kind="info"] * {
        color: #1e40af !important;
    }
    
    /* Success alerts */
    div[data-baseweb="notification"][kind="success"] {
        background-color: #f0fdf4 !important;
        border-color: #22c55e !important;
        color: #166534 !important;
    }
    
    div[data-baseweb="notification"][kind="success"] * {
        color: #166534 !important;
    }
    
    /* Warning alerts */
    div[data-baseweb="notification"][kind="warning"] {
        background-color: #fffbeb !important;
        border-color: #f59e0b !important;
        color: #92400e !important;
    }
    
    div[data-baseweb="notification"][kind="warning"] * {
        color: #92400e !important;
    }
    
    /* Error alerts */
    div[data-baseweb="notification"][kind="error"] {
        background-color: #fef2f2 !important;
        border-color: #ef4444 !important;
        color: #991b1b !important;
    }
    
    div[data-baseweb="notification"][kind="error"] * {
        color: #991b1b !important;
    }
    
    /* Button styling - clean solid colors */
    .stButton > button {
        background: #2563eb;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #2563eb;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 6px;
        border: 1px solid #d1d5db;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus,
    .stMultiSelect > div > div:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8fafc;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2563eb;
        color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #2563eb;
        background: #f8fafc;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: #2563eb;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #2563eb;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 8px;
        border: 2px dashed #cbd5e0;
        padding: 2rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2563eb;
        background: #f8fafc;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.875rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    /* Checkbox and Radio */
    .stCheckbox, .stRadio {
        background: transparent;
        padding: 0.5rem;
        border-radius: 6px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = DataCleaner()
    st.session_state.data_loaded = False
    st.session_state.duplicates_detected = False
    st.session_state.duplicate_groups = None
    st.session_state.cleaning_complete = False

def main():
    """Main application function."""
    
    # Header with clean design
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(f'<div class="main-header">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Efficiently clean your CSV data and detect duplicates with advanced AI-powered algorithms</p>', unsafe_allow_html=True)
    
    # Sidebar with clean styling and step indicators
    with st.sidebar:
        st.markdown(f"## {APP_ICON} Data Cleaning")
        st.markdown("---")
        st.markdown("### 📍 Step-by-Step Process")
        
        # Determine current step status
        step_status = {
            "Upload": "✅" if st.session_state.data_loaded else "1️⃣",
            "Detect": "✅" if st.session_state.duplicates_detected else ("2️⃣" if st.session_state.data_loaded else "⏸️"),
            "Clean": "✅" if st.session_state.cleaning_complete else ("3️⃣" if st.session_state.duplicates_detected else "⏸️"),
            "Report": "4️⃣" if st.session_state.data_loaded else "⏸️",
            "Export": "5️⃣" if st.session_state.data_loaded else "⏸️"
        }
        
        page = st.radio(
            "Select Page",
            [
                f"{step_status['Upload']} Upload Data",
                f"{step_status['Detect']} Detect Duplicates",
                f"{step_status['Clean']} Clean Data",
                f"{step_status['Report']} Reports & Analytics",
                f"{step_status['Export']} Export Data"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if st.session_state.data_loaded:
            summary = st.session_state.cleaner.get_data_summary()
            
            # Clean styled metrics in sidebar
            st.markdown(f"""
            <div style="background: #ffffff; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e2e8f0;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">TOTAL RECORDS</div>
                <div style="color: #1e293b; font-size: 1.5rem; font-weight: 700;">{summary.get('total_rows', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #ffffff; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e2e8f0;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">COLUMNS</div>
                <div style="color: #1e293b; font-size: 1.5rem; font-weight: 700;">{summary.get('total_columns', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.duplicates_detected:
                report = st.session_state.cleaner.get_cleaning_report()
                st.markdown(f"""
                <div style="background: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">DUPLICATES FOUND</div>
                    <div style="color: #dc2626; font-size: 1.5rem; font-weight: 700;">{report.get('duplicates_found', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                <div style="color: #475569; font-size: 0.95rem;">💡 No data loaded yet</div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">Upload data to see stats</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content based on selected page
    if "Upload Data" in page:
        upload_data_page()
    elif "Detect Duplicates" in page:
        detect_duplicates_page()
    elif "Clean Data" in page:
        clean_data_page()
    elif "Reports & Analytics" in page:
        reports_page()
    elif "Export Data" in page:
        export_data_page()

def upload_data_page():
    """Page for uploading CSV data."""
    st.markdown('<div class="sub-header">📤 Step 1: Upload Your Data</div>', unsafe_allow_html=True)
    
    # Progress indicator
    st.markdown("""
    <div class="card" style="background: #eff6ff; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">🎯 Current Step: Data Upload</h4>
        <p style="color: #1e293b; margin: 0; font-size: 1rem;">
            <strong>What to do:</strong> Upload your CSV file or generate sample data to begin the cleaning process.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p style="font-size: 1.1rem; color: #475569;">
            Upload a CSV file containing user data. Our AI-powered system will analyze it for duplicates and help you clean the data efficiently.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="Maximum file size: 200MB")
    
    if uploaded_file is not None:
        try:
            # Show progress indicator
            with st.spinner("📁 Loading CSV file..."):
                # Optimize CSV reading with low_memory and dtype inference optimization
                df = pd.read_csv(uploaded_file, low_memory=False)
            
            st.session_state.cleaner.load_dataframe(df)
            st.session_state.data_loaded = True
            st.session_state.duplicates_detected = False
            st.session_state.cleaning_complete = False
            
            st.success(f"✨ File uploaded successfully! Loaded **{len(df):,}** records with **{len(df.columns)}** columns.")
            
            # Display data preview with modern card
            st.markdown("### 📋 Data Preview")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True, height=400)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display column information in modern layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Column Information")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                # Optimize: Use isna() instead of isnull() and calculate once
                with st.spinner("📊 Analyzing data quality..."):
                    null_counts = df.isna().sum()
                    total_rows = len(df)
                    
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': total_rows - null_counts,
                        'Null %': (null_counts / total_rows * 100).round(2)
                    })
                
                st.dataframe(col_info, use_container_width=True, height=400)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Data Quality Metrics")
                
                # Optimize: Calculate metrics once
                total_cells = total_rows * len(df.columns)
                null_cells = null_counts.sum()
                completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
                
                # Modern metric cards
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Completeness</div>
                        <div class="metric-value">{completeness:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Cells</div>
                        <div class="metric-value">{total_cells:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="card" style="margin-top: 1rem; text-align: center;">
                    <p style="color: #4a5568; margin-bottom: 0.5rem;">Missing Values</p>
                    <p style="font-size: 2rem; font-weight: 700; color: {'#e53e3e' if null_cells > 0 else '#38a169'}; margin: 0;">
                        {null_cells:,}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualize missing data - only if there are missing values and not too many columns
                if null_cells > 0 and len(df.columns) <= 50:
                    fig = px.bar(
                        x=null_counts.index,
                        y=null_counts.values,
                        labels={'x': 'Column', 'y': 'Missing Values'},
                        title='Missing Values by Column',
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter, sans-serif')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif len(df.columns) > 50:
                    st.info("📊 Chart hidden for datasets with >50 columns to improve performance")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Sample data generator with modern design
    st.markdown("---")
    st.markdown("### 🎲 Don't have a CSV file? Generate sample data")
    if st.button("🚀 Generate Sample Data", use_container_width=False):
        sample_df = generate_sample_data()
        st.session_state.cleaner.load_dataframe(sample_df)
        st.session_state.data_loaded = True
        st.session_state.duplicates_detected = False
        st.session_state.cleaning_complete = False
        st.success("✅ Sample data generated successfully!")
        st.dataframe(sample_df.head(10), use_container_width=True)

def detect_duplicates_page():
    """Page for detecting duplicates."""
    st.markdown('<div class="sub-header">🔍 Step 2: Detect Duplicates</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="card" style="background: #fffbeb; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 0.5rem 0;">⚠️ Data Required</h4>
            <p style="color: #92400e; margin: 0; font-size: 1rem;">
                Please upload data first from <strong>Step 1: Upload Data</strong> page.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Progress indicator
    st.markdown("""
    <div class="card" style="background: #eff6ff; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">🎯 Current Step: Duplicate Detection</h4>
        <p style="color: #1e293b; margin: 0; font-size: 1rem;">
            <strong>What to do:</strong> Choose a detection method and configure parameters to find duplicate records.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    cleaner = st.session_state.cleaner
    df = cleaner.df
    
    st.markdown("""
    <div class="card">
        <p style="font-size: 1.1rem; color: #475569;">
            Choose a duplicate detection method and configure the parameters. Our AI-powered algorithms will analyze your data intelligently.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detection method selection with modern design
    st.markdown("### 🎯 Select Detection Method")
    detection_method = st.radio(
        "detection_method",
        ["ML Advanced (Learning)", "Smart AI (Automatic)", "Exact Match", "Fuzzy Match (AI-Powered)"],
        help="ML Advanced uses machine learning that learns and improves with data. Smart AI automatically detects duplicates like a human would. Exact Match finds identical records. Fuzzy Match uses AI to find similar records.",
        label_visibility="collapsed"
    )
    
    # Column selection
    st.markdown("### ⚙️ Configure Detection")
    available_columns = list(df.columns)
    
    if detection_method == "ML Advanced (Learning)":
        # Check if ML dependencies are available
        if not ML_AVAILABLE:
            st.error("""
            ⚠️ **ML Advanced features are not available**
            
            The advanced ML dependencies are not installed. This feature requires additional packages 
            that may need compilation on Windows systems.
            
            **To enable ML Advanced features:**
            
            1. **On Linux/Mac:**
               ```bash
               pip install -r requirements-ml.txt
               ```
            
            2. **On Windows:**
               - Install Visual Studio Build Tools first
               - Or use Windows Subsystem for Linux (WSL)
               - Or use pre-compiled wheels if available
            
            **Alternative:** Use "Smart AI (Automatic)" detection which provides excellent results 
            without requiring advanced ML packages!
            """)
            
            if not RECORDLINKAGE_AVAILABLE:
                st.warning(f"Note: Record linkage is also unavailable - {RECORDLINKAGE_IMPORT_ERROR}")
            
            return
        
        st.markdown("""
        <div class="card" style="background: #eff6ff; border: 1px solid #3b82f6;">
            <h3 style="color: #1e40af; margin-top: 0;">🧠 ML Advanced Detection</h3>
            <p style="color: #1e293b;">Uses cutting-edge machine learning algorithms that learn patterns from your data and get better with each use!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show optimization status
        optimization_status = []
        if PERFORMANCE_LIBS_AVAILABLE:
            optimization_status.append("✅ Performance Optimization Active (2-10x faster)")
        else:
            optimization_status.append("⚠️ Performance optimization not installed (slower processing)")
        
        if RETAIL_LIBS_AVAILABLE:
            optimization_status.append("✅ Retail Data Optimization Active (better accuracy for customer data)")
        else:
            optimization_status.append("💡 Install retail libraries for better accuracy: pip install nameparser email-validator")
        
        if optimization_status:
            st.info("\n".join(optimization_status))
        
        st.markdown("""
        **Advanced Features:**
        - 🎯 TF-IDF vectorization for semantic similarity
        - 🔬 DBSCAN clustering for pattern detection
        - 🗣️ Phonetic matching (Soundex, Metaphone) for name variations
        - 🔗 Record linkage algorithms
        - 📈 Learns from each cleaning session and improves over time
        - 🏪 Retail-specific normalization (names, emails, phones)
        - ⚡ Parallel processing for large datasets
        """)
        
        # Show capacity information
        if PERFORMANCE_LIBS_AVAILABLE:
            st.success("🚀 **Performance Mode**: Can process up to 100,000 rows efficiently")
        else:
            st.warning("⚡ **Standard Mode**: Can process up to 50,000 rows. Install performance libraries for 2x capacity: `pip install numba joblib`")
        
        if not RECORDLINKAGE_AVAILABLE:
            st.info("ℹ️ Record linkage algorithms are not available but other ML features will work.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            learn_mode = st.checkbox("Learn from this data", value=True,
                                    help="AI will learn patterns from this data to improve future performance")
        with col2:
            use_phonetic = st.checkbox("Use phonetic matching", value=True,
                                      help="Match names that sound similar (e.g., 'Smith' and 'Smythe')")
        with col3:
            use_clustering = st.checkbox("Use ML clustering", value=True,
                                        help="Use advanced clustering algorithms to find duplicate patterns")
        
        if st.button("🚀 Run ML Advanced Detection", type="primary"):
            with st.spinner("🧠 ML is analyzing your data and learning patterns..."):
                try:
                    duplicate_groups, num_duplicates, uncleaned_df = cleaner.detect_duplicates_ml_advanced(
                        learn_from_data=learn_mode,
                        use_phonetic=use_phonetic,
                        use_clustering=use_clustering
                    )
                    
                    st.session_state.duplicates_detected = True
                    st.session_state.duplicate_groups = duplicate_groups
                    st.session_state.uncleaned_data = uncleaned_df
                    st.session_state.detection_method = 'ml_advanced'
                    
                    report = cleaner.get_cleaning_report()
                    
                    if num_duplicates > 0:
                        st.success(f"✅ ML Advanced found {len(duplicate_groups)} groups containing {num_duplicates} duplicate records!")
                        
                        # Show learning stats
                        if report.get('learning_stats'):
                            st.info(f"📊 ML Stats: {report['learning_stats']['patterns_learned']} patterns learned | Session #{report['learning_stats']['total_sessions']}")
                        
                        # Show what columns were analyzed
                        st.info(f"📊 Analyzed columns: {', '.join(report.get('columns_analyzed', []))}")
                        
                        # Show confidence distribution
                        if report.get('confidence_scores'):
                            avg_confidence = np.mean(report['confidence_scores'])
                            st.metric("Average ML Confidence", f"{avg_confidence:.1f}%")
                        
                        # Show duplicate groups
                        st.markdown("### Duplicate Groups (ML Detected)")
                        high_conf_groups = [g for g in duplicate_groups if g.get('group_confidence', 0) >= 65]
                        low_conf_groups = [g for g in duplicate_groups if g.get('group_confidence', 0) < 65]
                        
                        if high_conf_groups:
                            st.markdown(f"#### 🟢 High Confidence ML Matches ({len(high_conf_groups)} groups)")
                            for idx, group in enumerate(high_conf_groups, 1):
                                records = group['records']
                                confidence = group['group_confidence']
                                method = group.get('detection_method', 'ML')
                                with st.expander(f"Group {idx} - {len(records)} records (Confidence: {confidence:.1f}% | Method: {method})"):
                                    group_df = pd.DataFrame([r['record'] for r in records])
                                    similarities = [f"{r.get('similarity', 100):.1f}%" for r in records[1:]]
                                    
                                    st.write(f"**ML Similarity Scores:** {similarities}")
                                    
                                    # Show phonetic info if available
                                    if 'phonetic_similarity' in records[0]:
                                        phonetic_scores = [f"{r.get('phonetic_similarity', 100):.1f}%" for r in records[1:]]
                                        st.write(f"**Phonetic Match Scores:** {phonetic_scores}")
                                    
                                    st.dataframe(group_df, use_container_width=True)
                        
                        if low_conf_groups:
                            st.markdown(f"#### 🟡 Lower Confidence ML Matches ({len(low_conf_groups)} groups) - Review Recommended")
                            for idx, group in enumerate(low_conf_groups, 1):
                                records = group['records']
                                confidence = group['group_confidence']
                                with st.expander(f"Group {idx} - {len(records)} records (Confidence: {confidence:.1f}%)"):
                                    group_df = pd.DataFrame([r['record'] for r in records])
                                    st.warning("⚠️ Lower ML confidence - please review before cleaning")
                                    st.dataframe(group_df, use_container_width=True)
                        
                        # Show uncleaned data info
                        if not uncleaned_df.empty:
                            st.warning(f"⚠️ {len(uncleaned_df)} records flagged for manual review (low ML confidence)")
                    else:
                        st.info("🎉 No duplicates found with ML Advanced detection!")
                        
                        # Still show learning info
                        if learn_mode:
                            st.success("✅ ML model has learned patterns from this data for future use!")
                
                except Exception as e:
                    st.error(f"Error during ML Advanced detection: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    elif detection_method == "Smart AI (Automatic)":
        st.info("🤖 Smart AI will automatically identify the best columns and strategies for duplicate detection, similar to how a human would analyze the data.")
        
        # Show performance warning for large datasets
        if len(df) > LARGE_DATASET_WARNING_THRESHOLD:
            est_time_min = (len(df) // 1000) * PERF_ESTIMATE_MIN_PER_1K
            est_time_max = (len(df) // 1000) * PERF_ESTIMATE_MAX_PER_1K
            st.warning(f"""
            ⚠️ **Large Dataset Detected ({len(df):,} records)**
            
            Smart AI duplicate detection may take some time with large datasets. 
            
            **Recommendations:**
            - Use **Exact Match** first for faster duplicate removal
            - Or filter your data to focus on specific subsets
            - Consider using external tools for very large datasets (>50k records)
            
            Processing time estimate: ~{est_time_min} to {est_time_max} minutes depending on data complexity.
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            auto_detect = st.checkbox("Auto-detect key columns", value=True, 
                                     help="Let AI automatically identify which columns are most important for duplicate detection (IDs, names, emails, etc.)")
        with col2:
            smart_threshold = st.slider(
                "Detection Sensitivity",
                min_value=60,
                max_value=95,
                value=80,
                help="Lower values detect more potential duplicates but may include false positives. 80% is balanced."
            )
        
        if not auto_detect:
            selected_columns = st.multiselect(
                "Manual column selection",
                available_columns,
                default=available_columns[:min(3, len(available_columns))]
            )
        else:
            selected_columns = None
        
        if st.button("🧠 Run Smart AI Detection", type="primary"):
            # Add progress message for large datasets
            if len(df) > 5000:
                progress_text = f"🤖 Analyzing {len(df):,} records... This may take a few minutes for large datasets."
            else:
                progress_text = "🤖 AI is analyzing your data with human-like intelligence..."
            
            with st.spinner(progress_text):
                try:
                    duplicate_groups, num_duplicates, uncleaned_df = cleaner.detect_duplicates_smart_ai(
                        auto_detect_columns=auto_detect,
                        custom_columns=selected_columns,
                        threshold=smart_threshold
                    )
                    
                    st.session_state.duplicates_detected = True
                    st.session_state.duplicate_groups = duplicate_groups
                    st.session_state.uncleaned_data = uncleaned_df
                    st.session_state.detection_method = 'smart_ai'
                    
                    report = cleaner.get_cleaning_report()
                    
                    if num_duplicates > 0:
                        st.success(f"✅ Smart AI found {len(duplicate_groups)} groups containing {num_duplicates} duplicate records!")
                        
                        # Show what columns were analyzed
                        st.info(f"📊 Analyzed columns: {', '.join(report.get('columns_analyzed', []))}")
                        
                        # Show confidence distribution
                        if report.get('confidence_scores'):
                            avg_confidence = np.mean(report['confidence_scores'])
                            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                        
                        # Show duplicate groups
                        st.markdown("### Duplicate Groups")
                        high_conf_groups = [g for g in duplicate_groups if g.get('group_confidence', 0) >= 70]
                        low_conf_groups = [g for g in duplicate_groups if g.get('group_confidence', 0) < 70]
                        
                        if high_conf_groups:
                            st.markdown(f"#### 🟢 High Confidence Matches ({len(high_conf_groups)} groups)")
                            for idx, group in enumerate(high_conf_groups, 1):
                                records = group['records']
                                confidence = group['group_confidence']
                                with st.expander(f"Group {idx} - {len(records)} similar records (Confidence: {confidence:.1f}%)"):
                                    group_df = pd.DataFrame([r['record'] for r in records])
                                    similarities = [r.get('similarity', 100) for r in records[1:]]
                                    
                                    st.write(f"**Similarity Scores:** {[f'{s:.1f}%' for s in similarities]}")
                                    st.dataframe(group_df, use_container_width=True)
                        
                        if low_conf_groups:
                            st.markdown(f"#### 🟡 Lower Confidence Matches ({len(low_conf_groups)} groups) - Review Recommended")
                            for idx, group in enumerate(low_conf_groups, 1):
                                records = group['records']
                                confidence = group['group_confidence']
                                with st.expander(f"Group {idx} - {len(records)} records (Confidence: {confidence:.1f}%)"):
                                    group_df = pd.DataFrame([r['record'] for r in records])
                                    st.warning("⚠️ Lower confidence match - please review before cleaning")
                                    st.dataframe(group_df, use_container_width=True)
                        
                        # Show uncleaned data info
                        if not uncleaned_df.empty:
                            st.warning(f"⚠️ {len(uncleaned_df)} records flagged for manual review (low confidence duplicates)")
                    else:
                        st.info("🎉 No duplicates found with Smart AI detection!")
                
                except Exception as e:
                    st.error(f"Error during Smart AI detection: {str(e)}")
    
    elif detection_method == "Exact Match":
        selected_columns = st.multiselect(
            "Select columns for exact matching (leave empty to use all columns)",
            available_columns,
            default=[]
        )
        
        if st.button("🔍 Detect Exact Duplicates", type="primary"):
            with st.spinner("Detecting duplicates..."):
                cols_to_use = selected_columns if selected_columns else None
                duplicates_df, num_duplicates = cleaner.detect_duplicates_exact(columns=cols_to_use)
                
                st.session_state.duplicates_detected = True
                st.session_state.duplicate_groups = None
                st.session_state.detection_method = 'exact'
                
                if num_duplicates > 0:
                    st.success(f"✅ Found {num_duplicates} duplicate records!")
                    
                    st.markdown("### Duplicate Records")
                    st.dataframe(duplicates_df, use_container_width=True)
                else:
                    st.info("🎉 No exact duplicates found in the data!")
    
    else:  # Fuzzy Match
        # Show performance warning for large datasets
        if len(df) > LARGE_DATASET_WARNING_THRESHOLD:
            est_time_min = (len(df) // 1000) * PERF_ESTIMATE_MIN_PER_1K
            est_time_max = (len(df) // 1000) * PERF_ESTIMATE_MAX_PER_1K
            st.warning(f"""
            ⚠️ **Large Dataset Detected ({len(df):,} records)**
            
            Fuzzy matching may take significant time with large datasets due to comparing each record with others.
            
            **Recommendations:**
            - Use **Exact Match** first to remove obvious duplicates
            - Or use **Smart AI** which has optimizations for large datasets  
            - Consider filtering/sampling your data before fuzzy matching
            
            Processing time estimate: ~{est_time_min} to {est_time_max} minutes depending on column count.
            """)
        
        selected_columns = st.multiselect(
            "Select columns for fuzzy matching (required)",
            available_columns,
            default=available_columns[:min(3, len(available_columns))]
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold (%)",
            min_value=50,
            max_value=100,
            value=85,
            help="Higher values require closer matches. 85% is recommended for most cases."
        )
        
        if st.button("🤖 Detect Fuzzy Duplicates (AI)", type="primary"):
            if not selected_columns:
                st.error("Please select at least one column for comparison.")
            else:
                # Add progress message for large datasets
                if len(df) > 5000:
                    progress_text = f"🤖 Analyzing {len(df):,} records with fuzzy matching... This may take several minutes."
                else:
                    progress_text = "AI is analyzing your data for similar records..."
                
                with st.spinner(progress_text):
                    duplicate_groups, num_duplicates = cleaner.detect_duplicates_fuzzy(
                        columns=selected_columns,
                        threshold=similarity_threshold
                    )
                    
                    st.session_state.duplicates_detected = True
                    st.session_state.duplicate_groups = duplicate_groups
                    st.session_state.detection_method = 'fuzzy'
                    
                    if num_duplicates > 0:
                        st.success(f"✅ Found {len(duplicate_groups)} groups containing {num_duplicates} duplicate records!")
                        
                        st.markdown("### Duplicate Groups")
                        for idx, group in enumerate(duplicate_groups, 1):
                            with st.expander(f"Group {idx} - {len(group)} similar records"):
                                group_df = pd.DataFrame([r['record'] for r in group])
                                similarities = [r.get('similarity', 100) for r in group]
                                
                                st.write(f"**Similarity Scores:** {similarities[1:]}")
                                st.dataframe(group_df, use_container_width=True)
                    else:
                        st.info("🎉 No fuzzy duplicates found based on the specified threshold!")

def clean_data_page():
    """Page for cleaning data."""
    st.markdown('<div class="sub-header">🧹 Step 3: Clean Your Data</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="card" style="background: #fffbeb; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 0.5rem 0;">⚠️ Data Required</h4>
            <p style="color: #92400e; margin: 0; font-size: 1rem;">
                Please upload data first from <strong>Step 1: Upload Data</strong> page.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Progress indicator
    st.markdown("""
    <div class="card" style="background: #eff6ff; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">🎯 Current Step: Data Cleaning</h4>
        <p style="color: #1e293b; margin: 0; font-size: 1rem;">
            <strong>What to do:</strong> Remove duplicates, handle missing values, and standardize your data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    cleaner = st.session_state.cleaner
    
    st.info("Configure cleaning operations and apply them to your data.")
    
    tab1, tab2, tab3 = st.tabs(["Remove Duplicates", "Handle Missing Values", "Standardize Data"])
    
    with tab1:
        st.markdown("### Remove Duplicate Records")
        
        if not st.session_state.duplicates_detected:
            st.warning("⚠️ Please detect duplicates first from the 'Detect Duplicates' page.")
        else:
            report = cleaner.get_cleaning_report()
            st.info(f"Detected {report.get('duplicates_found', 0)} duplicates using {report.get('cleaning_method', 'N/A')}")
            
            if report.get('duplicates_found', 0) > 0:
                detection_method = st.session_state.get('detection_method', 'unknown')
                
                # Special handling for ML Advanced and Smart AI
                if detection_method in ['smart_ai', 'ml_advanced']:
                    st.markdown("#### AI/ML Cleaning Options")
                    
                    clean_option = st.radio(
                        "Select cleaning strategy",
                        ["High confidence only (Recommended)", "All duplicates"],
                        help="High confidence only removes duplicates the AI is very sure about. Low confidence matches will be exported separately for manual review."
                    )
                    
                    high_conf_only = (clean_option == "High confidence only (Recommended)")
                    
                    button_label = "🗑️ Remove Duplicates with ML" if detection_method == 'ml_advanced' else "🗑️ Remove Duplicates with Smart AI"
                    
                    if st.button(button_label, type="primary"):
                        with st.spinner("Removing duplicates..."):
                            cleaned_data, uncleaned_data = cleaner.remove_smart_ai_duplicates(
                                st.session_state.duplicate_groups,
                                high_confidence_only=high_conf_only
                            )
                            
                            st.session_state.cleaning_complete = True
                            st.session_state.has_uncleaned = not uncleaned_data.empty
                            
                            new_report = cleaner.get_cleaning_report()
                            st.success(f"✅ Removed {new_report['records_removed']} duplicate records!")
                            st.info(f"✨ Cleaned dataset now has {len(cleaned_data)} records.")
                            
                            if not uncleaned_data.empty:
                                st.warning(f"⚠️ {len(uncleaned_data)} records require manual review (low confidence duplicates)")
                                st.info("💡 You can export both cleaned and uncleaned data separately from the Export page.")
                            
                            if detection_method == 'ml_advanced':
                                st.success("🧠 ML model has been updated and will perform better on future data!")
                else:
                    # Original logic for exact/fuzzy
                    if st.button("🗑️ Remove Duplicates", type="primary"):
                        with st.spinner("Removing duplicates..."):
                            if st.session_state.duplicate_groups:
                                cleaner.remove_fuzzy_duplicates(st.session_state.duplicate_groups)
                            else:
                                cols = report.get('columns_analyzed')
                                cleaner.remove_exact_duplicates(columns=cols if cols else None)
                            
                            st.session_state.cleaning_complete = True
                            new_report = cleaner.get_cleaning_report()
                            st.success(f"✅ Removed {new_report['records_removed']} duplicate records!")
                            st.info(f"Dataset now has {new_report['final_records']} records.")
    
    with tab2:
        st.markdown("### Handle Missing Values")
        
        df = cleaner.df
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.warning(f"⚠️ Found {missing_count} missing values in the dataset.")
            
            strategy = st.selectbox(
                "Select Missing Value Strategy",
                ["drop", "fill_empty", "fill_mean", "fill_mode"],
                format_func=lambda x: {
                    "drop": "Drop rows with missing values",
                    "fill_empty": "Fill with empty string",
                    "fill_mean": "Fill numeric columns with mean",
                    "fill_mode": "Fill categorical columns with mode"
                }[x]
            )
            
            if st.button("Apply Missing Value Strategy"):
                with st.spinner("Handling missing values..."):
                    cleaner.handle_missing_values(strategy=strategy)
                    st.success("✅ Missing values handled successfully!")
                    st.info(f"Dataset now has {len(cleaner.df)} records.")
    
    with tab3:
        st.markdown("### Standardize Data")
        
        df = cleaner.df
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not text_columns:
            st.info("No text columns available for standardization.")
        else:
            selected_cols = st.multiselect(
                "Select columns to standardize",
                text_columns,
                default=[]
            )
            
            operation = st.selectbox(
                "Select standardization operation",
                ["lowercase", "uppercase", "titlecase", "strip"],
                format_func=lambda x: {
                    "lowercase": "Convert to lowercase",
                    "uppercase": "Convert to UPPERCASE",
                    "titlecase": "Convert To Title Case",
                    "strip": "Remove leading/trailing spaces"
                }[x]
            )
            
            if selected_cols and st.button("Apply Standardization"):
                with st.spinner("Standardizing data..."):
                    cleaner.standardize_data(columns=selected_cols, operation=operation)
                    st.success("✅ Data standardized successfully!")

def reports_page():
    """Page for viewing reports and analytics."""
    st.markdown('<div class="sub-header">📊 Step 4: Reports & Analytics</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="card" style="background: #fffbeb; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 0.5rem 0;">⚠️ Data Required</h4>
            <p style="color: #92400e; margin: 0; font-size: 1rem;">
                Please upload data first from <strong>Step 1: Upload Data</strong> page.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Progress indicator
    st.markdown("""
    <div class="card" style="background: #eff6ff; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">🎯 Current Step: View Reports</h4>
        <p style="color: #1e293b; margin: 0; font-size: 1rem;">
            <strong>What to do:</strong> Review data quality metrics and cleaning analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    cleaner = st.session_state.cleaner
    report = cleaner.get_cleaning_report()
    summary = cleaner.get_data_summary()
    
    # Overview metrics
    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Records", report.get('total_records', 0))
    with col2:
        st.metric("Current Records", summary.get('total_rows', 0))
    with col3:
        st.metric("Duplicates Found", report.get('duplicates_found', 0))
    with col4:
        st.metric("Records Removed", report.get('records_removed', 0))
    
    # Cleaning report
    st.markdown("### Cleaning Report")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Process Details")
        details = {
            "Timestamp": report.get('timestamp', 'N/A'),
            "Cleaning Method": report.get('cleaning_method', 'Not applied'),
            "Columns Analyzed": ', '.join(report.get('columns_analyzed', [])) if report.get('columns_analyzed') else 'All',
            "Total Columns": summary.get('total_columns', 0),
            "Memory Usage": f"{summary.get('memory_usage', 0):.2f} MB"
        }
        for key, value in details.items():
            st.text(f"{key}: {value}")
    
    with col2:
        st.markdown("#### Data Quality")
        
        # Calculate data quality score
        completeness = 0
        if report.get('total_records', 0) > 0:
            completeness = (summary.get('total_rows', 0) / report.get('total_records', 1)) * 100
        
        quality_score = completeness
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Data Quality Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualizations
    st.markdown("### Data Visualizations")
    
    df = cleaner.df
    
    # Missing values chart
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Records Distribution")
            fig = go.Figure(data=[
                go.Bar(
                    x=['Original', 'After Cleaning', 'Removed'],
                    y=[
                        report.get('total_records', 0),
                        summary.get('total_rows', 0),
                        report.get('records_removed', 0)
                    ],
                    marker_color=['blue', 'green', 'red']
                )
            ])
            fig.update_layout(title="Data Cleaning Impact")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Column Data Types")
            dtype_counts = df.dtypes.value_counts()
            fig = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index.astype(str),
                title="Data Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed report download
    st.markdown("### Download Detailed Report")
    if st.button("Generate JSON Report"):
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def export_data_page():
    """Page for exporting cleaned data."""
    st.markdown('<div class="sub-header">⬇️ Step 5: Export Cleaned Data</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="card" style="background: #fffbeb; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 0.5rem 0;">⚠️ Data Required</h4>
            <p style="color: #92400e; margin: 0; font-size: 1rem;">
                Please upload data first from <strong>Step 1: Upload Data</strong> page.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Progress indicator
    st.markdown("""
    <div class="card" style="background: #eff6ff; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">🎯 Current Step: Export Data</h4>
        <p style="color: #1e293b; margin: 0; font-size: 1rem;">
            <strong>What to do:</strong> Download your cleaned data in CSV or Excel format.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    cleaner = st.session_state.cleaner
    has_uncleaned = st.session_state.get('has_uncleaned', False)
    
    st.info("Export your cleaned data in various formats. " + 
            ("Uncleaned data (requiring manual review) can also be exported separately." if has_uncleaned else ""))
    
    # Check if we have separate cleaned/uncleaned data
    if cleaner.cleaned_data is not None:
        df = cleaner.cleaned_data
        st.success("✅ Showing cleaned data from Smart AI processing")
    else:
        df = cleaner.df
    
    # Preview current data
    st.markdown("### Cleaned Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    # Export options
    st.markdown("### Export Options")
    
    export_format = st.radio("Select Export Format", ["CSV", "Excel"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cleaned Data")
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        else:  # Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
            st.download_button(
                label="📥 Download Cleaned Excel",
                data=buffer.getvalue(),
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
    
    with col2:
        if has_uncleaned and cleaner.uncleaned_data is not None and not cleaner.uncleaned_data.empty:
            st.markdown("#### Uncleaned Data (Manual Review Required)")
            st.warning(f"⚠️ {len(cleaner.uncleaned_data)} records need review")
            
            if export_format == "CSV":
                uncleaned_csv = cleaner.uncleaned_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Uncleaned CSV",
                    data=uncleaned_csv,
                    file_name=f"uncleaned_data_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:  # Excel
                buffer2 = io.BytesIO()
                with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
                    cleaner.uncleaned_data.to_excel(writer, index=False, sheet_name='Requires Review')
                
                st.download_button(
                    label="📥 Download Uncleaned Excel",
                    data=buffer2.getvalue(),
                    file_name=f"uncleaned_data_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("No uncleaned data - all records were processed successfully!")
    
    # Summary
    st.markdown("### Export Summary")
    report = cleaner.get_cleaning_report()
    
    summary_rows = [
        ["Original Records", report.get('total_records', 0)],
        ["Exported Records (Cleaned)", len(df)],
        ["Duplicates Removed", report.get('records_removed', 0)],
        ["Cleaning Method", report.get('cleaning_method', 'Not applied')],
    ]
    
    if has_uncleaned and cleaner.uncleaned_data is not None:
        summary_rows.append(["Records Requiring Review", len(cleaner.uncleaned_data)])
        summary_rows.append(["Data Quality", "High (with manual review recommended)"])
    else:
        summary_rows.append(["Data Quality", "High" if report.get('records_removed', 0) > 0 else "Not cleaned"])
    
    summary_data = {
        "Metric": [row[0] for row in summary_rows],
        "Value": [row[1] for row in summary_rows]
    }
    
    st.table(pd.DataFrame(summary_data))
    
    
    st.table(pd.DataFrame(summary_data))

def generate_sample_data():
    """Generate sample data with duplicates for testing."""
    data = {
        'User ID': list(range(1, 21)),
        'Name': [
            'John Smith', 'Jane Doe', 'John Smith', 'Bob Johnson', 'Alice Williams',
            'john smith', 'Jane Doe', 'Charlie Brown', 'David Miller', 'Emma Davis',
            'John Smithh', 'Frank Wilson', 'Grace Lee', 'Bob Johnson', 'Helen Clark',
            'Alice Williams', 'Ian Martinez', 'Julia Anderson', 'Kevin Taylor', 'Laura Moore'
        ],
        'Email': [
            'john@email.com', 'jane@email.com', 'john@email.com', 'bob@email.com', 'alice@email.com',
            'john@email.com', 'jane@email.com', 'charlie@email.com', 'david@email.com', 'emma@email.com',
            'john@email.com', 'frank@email.com', 'grace@email.com', 'bob@email.com', 'helen@email.com',
            'alice@email.com', 'ian@email.com', 'julia@email.com', 'kevin@email.com', 'laura@email.com'
        ],
        'Phone': [
            '555-0001', '555-0002', '555-0001', '555-0004', '555-0005',
            '555-0001', '555-0002', '555-0008', '555-0009', '555-0010',
            '555-0001', '555-0012', '555-0013', '555-0004', '555-0015',
            '555-0005', '555-0017', '555-0018', '555-0019', '555-0020'
        ],
        'City': [
            'New York', 'Los Angeles', 'New York', 'Chicago', 'Houston',
            'New York', 'Los Angeles', 'Phoenix', 'Philadelphia', 'San Antonio',
            'New York', 'San Diego', 'Dallas', 'Chicago', 'San Jose',
            'Houston', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus'
        ]
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
