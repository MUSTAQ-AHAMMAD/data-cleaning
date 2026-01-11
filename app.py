"""
AI CRM - Data Cleaning Application
A user-friendly web interface for cleaning CSV files and detecting duplicates.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
from data_cleaner import DataCleaner

# Page configuration
st.set_page_config(
    page_title="AI CRM - Data Cleaning",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
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
    
    # Header
    st.markdown('<div class="main-header">🧹 AI CRM - Data Cleaning System</div>', unsafe_allow_html=True)
    st.markdown("### Efficiently clean your CSV data and remove duplicate users with AI-powered detection")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["📤 Upload Data", "🔍 Detect Duplicates", "🧹 Clean Data", "📊 Reports & Analytics", "⬇️ Export Data"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        if st.session_state.data_loaded:
            summary = st.session_state.cleaner.get_data_summary()
            st.metric("Total Records", summary.get('total_rows', 0))
            st.metric("Total Columns", summary.get('total_columns', 0))
            if st.session_state.duplicates_detected:
                report = st.session_state.cleaner.get_cleaning_report()
                st.metric("Duplicates Found", report.get('duplicates_found', 0))
        else:
            st.info("No data loaded yet")
    
    # Main content based on selected page
    if page == "📤 Upload Data":
        upload_data_page()
    elif page == "🔍 Detect Duplicates":
        detect_duplicates_page()
    elif page == "🧹 Clean Data":
        clean_data_page()
    elif page == "📊 Reports & Analytics":
        reports_page()
    elif page == "⬇️ Export Data":
        export_data_page()

def upload_data_page():
    """Page for uploading CSV data."""
    st.markdown('<div class="sub-header">📤 Upload Your Data</div>', unsafe_allow_html=True)
    
    st.info("Upload a CSV file containing user data. The system will analyze it for duplicates and help you clean the data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.cleaner.load_dataframe(df)
            st.session_state.data_loaded = True
            st.session_state.duplicates_detected = False
            st.session_state.cleaning_complete = False
            
            st.success(f"✅ File uploaded successfully! Loaded {len(df)} records with {len(df.columns)} columns.")
            
            # Display data preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display column information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
            
            with col2:
                st.markdown("### Data Quality Metrics")
                total_cells = len(df) * len(df.columns)
                null_cells = df.isnull().sum().sum()
                completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
                
                st.metric("Data Completeness", f"{completeness:.2f}%")
                st.metric("Total Cells", f"{total_cells:,}")
                st.metric("Missing Values", f"{null_cells:,}")
                
                # Visualize missing data
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    fig = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        labels={'x': 'Column', 'y': 'Missing Values'},
                        title='Missing Values by Column'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Sample data generator
    st.markdown("---")
    st.markdown("### Don't have a CSV file? Generate sample data")
    if st.button("Generate Sample Data"):
        sample_df = generate_sample_data()
        st.session_state.cleaner.load_dataframe(sample_df)
        st.session_state.data_loaded = True
        st.session_state.duplicates_detected = False
        st.session_state.cleaning_complete = False
        st.success("✅ Sample data generated successfully!")
        st.dataframe(sample_df.head(10), use_container_width=True)

def detect_duplicates_page():
    """Page for detecting duplicates."""
    st.markdown('<div class="sub-header">🔍 Detect Duplicates</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first from the 'Upload Data' page.")
        return
    
    cleaner = st.session_state.cleaner
    df = cleaner.df
    
    st.info("Choose a duplicate detection method and configure the parameters.")
    
    # Detection method selection
    detection_method = st.radio(
        "Select Detection Method",
        ["Exact Match", "Fuzzy Match (AI-Powered)"],
        help="Exact Match finds identical records. Fuzzy Match uses AI to find similar records."
    )
    
    # Column selection
    st.markdown("### Select Columns to Compare")
    available_columns = list(df.columns)
    
    if detection_method == "Exact Match":
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
                
                if num_duplicates > 0:
                    st.success(f"✅ Found {num_duplicates} duplicate records!")
                    
                    st.markdown("### Duplicate Records")
                    st.dataframe(duplicates_df, use_container_width=True)
                else:
                    st.info("🎉 No exact duplicates found in the data!")
    
    else:  # Fuzzy Match
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
                with st.spinner("AI is analyzing your data for similar records..."):
                    duplicate_groups, num_duplicates = cleaner.detect_duplicates_fuzzy(
                        columns=selected_columns,
                        threshold=similarity_threshold
                    )
                    
                    st.session_state.duplicates_detected = True
                    st.session_state.duplicate_groups = duplicate_groups
                    
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
    st.markdown('<div class="sub-header">🧹 Clean Your Data</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first from the 'Upload Data' page.")
        return
    
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
    st.markdown('<div class="sub-header">📊 Reports & Analytics</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first from the 'Upload Data' page.")
        return
    
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
    st.markdown('<div class="sub-header">⬇️ Export Cleaned Data</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first from the 'Upload Data' page.")
        return
    
    cleaner = st.session_state.cleaner
    df = cleaner.df
    
    st.info("Export your cleaned data in various formats.")
    
    # Preview current data
    st.markdown("### Current Data Preview")
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
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
    
    with col2:
        if export_format == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
    
    # Summary
    st.markdown("### Export Summary")
    report = cleaner.get_cleaning_report()
    
    summary_data = {
        "Metric": [
            "Original Records",
            "Exported Records",
            "Duplicates Removed",
            "Cleaning Method",
            "Data Quality"
        ],
        "Value": [
            report.get('total_records', 0),
            len(df),
            report.get('records_removed', 0),
            report.get('cleaning_method', 'Not applied'),
            "High" if report.get('records_removed', 0) > 0 else "Not cleaned"
        ]
    }
    
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
