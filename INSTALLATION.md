# Installation Guide

This guide will help you install the AI CRM Data Cleaning application on different platforms.

## Quick Start (Core Features Only)

The core features work on all platforms without any special setup:

```bash
# 1. Clone the repository
git clone https://github.com/MUSTAQ-AHAMMAD/data-cleaning.git
cd data-cleaning

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
```

This gives you access to:
- ✅ Exact Match duplicate detection
- ✅ Fuzzy Match duplicate detection (AI-powered)
- ✅ Smart AI (Automatic) duplicate detection
- ✅ Data cleaning and standardization
- ✅ Excel and CSV export

## Advanced ML Features (Optional)

For the **ML Advanced (Learning)** feature, you need additional packages. Follow the platform-specific instructions below:

### Linux / macOS

ML dependencies install easily on Linux and macOS:

```bash
# Install ML dependencies
pip install -r requirements-ml.txt

# Run the application
streamlit run app.py
```

### Windows

On Windows, some ML packages require C++ compilation. You have several options:

#### Option 1: Windows Subsystem for Linux (Recommended)

The easiest way to use all features on Windows:

1. Install WSL2: https://learn.microsoft.com/en-us/windows/wsl/install
2. Install Ubuntu from Microsoft Store
3. Open Ubuntu terminal and follow Linux instructions above

#### Option 2: Install Visual Studio Build Tools

If you prefer native Windows installation:

1. Download and install Visual Studio Build Tools:
   - https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"
   
2. Install core dependencies first:
   ```bash
   pip install -r requirements.txt
   ```

3. Try installing ML dependencies:
   ```bash
   pip install -r requirements-ml.txt
   ```

If it fails, continue without ML Advanced features - the other features still work great!

#### Option 3: Use Core Features Only

The application works perfectly with just core dependencies:
- Smart AI detection provides excellent duplicate detection
- Fuzzy matching handles most use cases
- No compilation required

```bash
# Just install core dependencies
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting

### Issue: "streamlit: command not found"

This means the installation didn't complete successfully. Try:

```bash
# Check Python and pip versions
python --version
pip --version

# Upgrade pip
pip install --upgrade pip

# Try installing again
pip install -r requirements.txt
```

### Issue: "Building wheel for affinegap failed" (Windows)

These packages need C++ compilation:
- `affinegap`
- `dedupe-Levenshtein-search`
- `PyLBFGS`

**Solution**: Use core features only. Install without ML dependencies:

```bash
pip install -r requirements.txt
```

The app will work with Smart AI, Fuzzy Match, and Exact Match detection.

### Issue: Import errors when running the app

If you see import errors for ML packages:
- The app will automatically disable ML Advanced features
- Other detection methods will still work
- You can continue using the application

### Issue: "No module named 'streamlit'"

The installation didn't succeed. Try:

```bash
# Install core dependencies one by one to find the issue
pip install streamlit
pip install pandas
pip install numpy
pip install python-Levenshtein
pip install fuzzywuzzy
pip install rapidfuzz
pip install openpyxl
pip install plotly

# Run the app
streamlit run app.py
```

## Verification

To verify your installation:

```bash
# Check if streamlit is installed
streamlit --version

# Check installed packages
pip list | grep streamlit
pip list | grep pandas
pip list | grep scikit-learn  # Only if ML dependencies installed
```

## Feature Availability

| Feature | Core Install | With ML Install |
|---------|-------------|-----------------|
| Exact Match | ✅ | ✅ |
| Fuzzy Match (AI) | ✅ | ✅ |
| Smart AI (Automatic) | ✅ | ✅ |
| ML Advanced (Learning) | ❌ | ✅ |
| Data Cleaning | ✅ | ✅ |
| Export (CSV/Excel) | ✅ | ✅ |
| Record Linkage | ❌ | ✅ |
| Phonetic Matching | ❌ | ✅ |

## Python Version

- Minimum: Python 3.8
- Recommended: Python 3.10 or higher
- Tested on: Python 3.12

## Getting Help

If you continue to have issues:

1. Check that you're using Python 3.8 or higher
2. Try creating a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Open an issue on GitHub with:
   - Your operating system
   - Python version
   - Error messages
   - Output of `pip list`

## Quick Reference

```bash
# Core installation (works everywhere)
pip install -r requirements.txt

# ML features (Linux/Mac)
pip install -r requirements-ml.txt

# Development dependencies
pip install -r requirements-dev.txt

# Run the application
streamlit run app.py

# Run tests
pytest

# Access the app
# Browser will open automatically or navigate to:
# http://localhost:8501
```

## Upgrading

To upgrade to the latest version:

```bash
cd data-cleaning
git pull
pip install -r requirements.txt --upgrade
# Optional: pip install -r requirements-ml.txt --upgrade
```
