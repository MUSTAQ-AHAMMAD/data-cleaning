# Quick Start Guide

Get up and running with AI CRM Data Cleaning in 3 minutes!

## Method 1: One-Command Start (Recommended)

### Linux/Mac
```bash
./start.sh
```

### Windows
```bash
start.bat
```

That's it! The application will open in your browser automatically at http://localhost:8501

## Method 2: Manual Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run app.py
```

### Step 3: Open Browser
Navigate to: http://localhost:8501

## Method 3: Using Virtual Environment (Best Practice)

### Create and Activate Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## First-Time Usage

### Option A: Use Sample Data (Fastest)
1. Click "Upload Data" in sidebar
2. Click "Generate Sample Data" button
3. Navigate through the pages to explore features

### Option B: Upload Your Own CSV
1. Click "Upload Data" in sidebar
2. Click "Browse files" and select your CSV
3. Follow the workflow:
   - Upload Data → Detect Duplicates → Clean Data → Reports → Export

## Common Workflows

### Quick Cleanup
```
1. Upload CSV
2. Click "Detect Duplicates" → "Exact Match" → Detect
3. Click "Clean Data" → "Remove Duplicates"
4. Click "Export Data" → Download CSV
```

### Deep Cleaning
```
1. Upload CSV
2. Detect Duplicates (Fuzzy Match, 85% threshold)
3. Clean Data:
   - Remove duplicates
   - Handle missing values
   - Standardize text
4. Review Reports
5. Export cleaned data
```

## Need Help?

- **Detailed Guide**: See [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Technical Details**: See [README.md](README.md)
- **Project Info**: See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Modern web browser
- Internet connection (for first-time dependency installation)

## Troubleshooting

### Application won't start?
```bash
pip install --upgrade -r requirements.txt
```

### Port already in use?
```bash
streamlit run app.py --server.port 8502
```

### Dependencies not installing?
Make sure you have pip updated:
```bash
pip install --upgrade pip
```

## That's All!

You're ready to start cleaning your data efficiently! 🧹

For more detailed instructions, check out [USAGE_GUIDE.md](USAGE_GUIDE.md)
