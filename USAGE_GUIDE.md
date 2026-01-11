# AI CRM Data Cleaning - Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Using the Application](#using-the-application)
3. [Features Guide](#features-guide)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

## Getting Started

### Quick Start (Recommended)

#### Linux/Mac:
```bash
./start.sh
```

#### Windows:
```
start.bat
```

### Manual Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

## Using the Application

### 1. Upload Data Page

**Purpose**: Load your CSV file into the system

**Steps**:
1. Click on "Upload Data" in the sidebar
2. Click "Browse files" or drag and drop your CSV file
3. Review the data preview and column information
4. Check data quality metrics

**Alternative**: Click "Generate Sample Data" to test with demo data

**What You'll See**:
- Data preview (first 10 rows)
- Column information (data types, null counts)
- Data quality metrics (completeness percentage)
- Missing values visualization

### 2. Detect Duplicates Page

**Purpose**: Find duplicate or similar records in your data

#### Method 1: Exact Match

**When to Use**: 
- Find identical records
- Fast duplicate detection
- Perfect for clean data

**Steps**:
1. Select "Exact Match" method
2. Choose columns to compare (optional - leave empty to use all columns)
3. Click "Detect Exact Duplicates"
4. Review the duplicate records found

**Example Use Case**: Finding exact duplicate customer entries

#### Method 2: Fuzzy Match (AI-Powered)

**When to Use**:
- Find similar records with typos
- Detect variations in names/data
- Handle messy real-world data

**Steps**:
1. Select "Fuzzy Match (AI-Powered)" method
2. Choose columns to compare (at least one required)
3. Adjust similarity threshold (50-100%):
   - 90-100%: Very strict matching (almost identical)
   - 80-90%: Balanced approach (recommended: 85%)
   - 70-80%: Lenient matching (more variations)
4. Click "Detect Fuzzy Duplicates (AI)"
5. Review duplicate groups with similarity scores

**Example Use Cases**:
- "John Smith" vs "john smith" vs "John Smithh"
- "alice@email.com" vs "alice@email.com " (with space)
- "Bob Johnson" vs "Bob Jonson" (typo)

**Understanding Results**:
- **Groups**: Collections of similar records
- **Similarity Scores**: How similar each record is (higher = more similar)
- First record in each group is kept by default

### 3. Clean Data Page

**Purpose**: Apply cleaning operations to your data

#### Tab 1: Remove Duplicates

**Prerequisites**: Must detect duplicates first (Step 2)

**Steps**:
1. Review the number of duplicates detected
2. Click "Remove Duplicates" button
3. System keeps the first record from each duplicate group
4. See confirmation with number of records removed

**What Happens**:
- Exact duplicates: Removes identical rows
- Fuzzy duplicates: Removes similar records based on AI detection

#### Tab 2: Handle Missing Values

**Available Strategies**:

1. **Drop rows with missing values**
   - Removes any row containing null/empty values
   - Use when: Data completeness is critical

2. **Fill with empty string**
   - Replaces null values with ""
   - Use when: You want to keep all records

3. **Fill numeric columns with mean**
   - Calculates average for numeric columns
   - Use when: You need statistical balance

4. **Fill categorical columns with mode**
   - Uses most frequent value
   - Use when: Working with categories/labels

**Steps**:
1. Check missing value count
2. Select appropriate strategy
3. Click "Apply Missing Value Strategy"
4. Review results

#### Tab 3: Standardize Data

**Purpose**: Normalize text data for consistency

**Available Operations**:
- **Lowercase**: Convert text to lowercase
- **UPPERCASE**: Convert text to uppercase
- **Title Case**: First Letter Of Each Word Capitalized
- **Strip**: Remove leading/trailing spaces

**Steps**:
1. Select columns to standardize
2. Choose operation type
3. Click "Apply Standardization"

**Common Use Cases**:
- Lowercase emails: alice@EMAIL.com → alice@email.com
- Title case names: john smith → John Smith
- Strip extra spaces: "  John  " → "John"

### 4. Reports & Analytics Page

**Purpose**: View comprehensive statistics and insights

**Sections**:

1. **Overview Metrics**
   - Original record count
   - Current record count
   - Duplicates found
   - Records removed

2. **Process Details**
   - Timestamp of operations
   - Cleaning method used
   - Columns analyzed
   - Memory usage

3. **Data Quality Score**
   - Visual gauge showing data quality (0-100%)
   - Based on completeness and cleaning

4. **Visualizations**
   - Records distribution chart
   - Data types distribution
   - Cleaning impact analysis

5. **Download Report**
   - Generate detailed JSON report
   - Contains all cleaning statistics
   - Timestamped for record-keeping

### 5. Export Data Page

**Purpose**: Download your cleaned data

**Steps**:
1. Review data preview
2. Check record and column counts
3. Select export format:
   - **CSV**: Universal format, smaller size
   - **Excel**: Formatted spreadsheet, better for presentations
4. Click download button
5. File saved with timestamp

**Export Summary**:
- Original vs exported record count
- Number of duplicates removed
- Cleaning method used
- Data quality assessment

## Features Guide

### AI-Powered Duplicate Detection

The fuzzy matching algorithm uses:
- **Levenshtein Distance**: Measures character-level similarity
- **Multi-Column Analysis**: Compares across multiple fields
- **Weighted Scoring**: Averages similarity across selected columns

**How It Works**:
1. System compares each record with all others
2. Calculates similarity score (0-100%) for each pair
3. Groups records above threshold
4. Provides similarity scores for review

### Data Quality Metrics

**Completeness**: `(Non-null cells / Total cells) × 100%`
- 100%: Perfect, no missing values
- 90-99%: Excellent
- 80-89%: Good
- <80%: Needs attention

**Memory Usage**: Amount of RAM used by dataset
- Helps plan for large file processing

### Smart Column Detection

System automatically:
- Detects data types (text, numeric, date)
- Identifies columns with missing values
- Suggests relevant columns for comparison

## Best Practices

### For Best Results

1. **Start with Sample Data**
   - Test the system with sample data first
   - Understand the workflow before processing real data

2. **Backup Your Data**
   - Always keep original CSV file
   - Export cleaned data with different name

3. **Choose Right Columns**
   - For exact match: Use all unique identifier columns
   - For fuzzy match: Use 2-3 most important columns (Name, Email, Phone)

4. **Set Appropriate Threshold**
   - Start with 85% for fuzzy matching
   - Increase if too many false positives
   - Decrease if missing obvious duplicates

5. **Review Before Removing**
   - Always check detected duplicates
   - Verify similarity scores
   - Understand what will be removed

6. **Incremental Cleaning**
   - Remove duplicates first
   - Then handle missing values
   - Finally standardize data

7. **Export Regularly**
   - Export after each major cleaning step
   - Keep multiple versions for safety

### Common Workflows

#### Workflow 1: Customer Database Cleanup
```
1. Upload customer CSV
2. Detect fuzzy duplicates on Name + Email (85% threshold)
3. Review duplicate groups
4. Remove duplicates
5. Standardize Name (Title Case) and Email (Lowercase)
6. Fill missing phone numbers with empty string
7. Export as CSV
```

#### Workflow 2: User Registration Cleanup
```
1. Upload registration data
2. Detect exact duplicates on Email
3. Remove exact duplicates
4. Detect fuzzy duplicates on Name (80% threshold)
5. Review and remove fuzzy duplicates
6. Standardize all text to lowercase
7. Drop rows with critical missing values
8. Export as Excel
```

#### Workflow 3: Marketing List Deduplication
```
1. Upload marketing list
2. Standardize Email to lowercase first
3. Detect exact duplicates on Email
4. Remove duplicates
5. Standardize Name to Title Case
6. Fill missing cities with mode
7. Export as CSV for campaign
```

## Troubleshooting

### Common Issues

#### Issue: "No data loaded" warning
**Solution**: 
- Go to "Upload Data" page first
- Upload CSV file or generate sample data

#### Issue: "Please select at least one column"
**Solution**: 
- For fuzzy matching, you must select columns
- Choose the most relevant columns (Name, Email, etc.)

#### Issue: Too many duplicates detected
**Solution**:
- Increase similarity threshold (try 90-95%)
- Select fewer columns for comparison
- Use exact match instead of fuzzy match

#### Issue: Not enough duplicates detected
**Solution**:
- Decrease similarity threshold (try 75-80%)
- Select more columns for comparison
- Check if data has typos/variations

#### Issue: File upload fails
**Solution**:
- Check file format is CSV
- Ensure file has headers
- Try different encoding if special characters fail
- System will attempt multiple encodings automatically

#### Issue: Application won't start
**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Try starting manually
streamlit run app.py
```

#### Issue: Slow performance with large files
**Solution**:
- Use exact match for very large files (faster)
- Select fewer columns for fuzzy matching
- Consider processing in batches
- Close other applications to free memory

### Getting Help

If you encounter issues not covered here:
1. Check the GitHub Issues page
2. Review the README.md file
3. Open a new issue with:
   - Your operating system
   - Python version
   - Error message (if any)
   - Steps to reproduce

## Advanced Tips

### Working with Large Files

For files >10,000 rows:
- Use exact matching when possible
- Limit fuzzy match to 2-3 key columns
- Process in smaller batches if needed
- Export intermediate results

### Customizing Thresholds

Start with these guidelines:
- Names: 85-90% (names have variations)
- Emails: 95-100% (emails should be exact)
- Phone numbers: 90-100% (some formatting differences)
- Addresses: 80-85% (many variations possible)

### Data Preparation Tips

Before uploading:
- Ensure first row has column headers
- Remove any merged cells (Excel)
- Check for special characters
- Verify date formats are consistent

### Interpreting Similarity Scores

- 100%: Exact match
- 95-99%: Very similar (1-2 character difference)
- 85-94%: Similar with variations
- 70-84%: Somewhat similar
- <70%: Different records

---

## Quick Reference

### Keyboard Shortcuts (in browser)
- `Ctrl/Cmd + R`: Refresh application
- `Ctrl/Cmd + -`: Zoom out
- `Ctrl/Cmd + +`: Zoom in

### File Formats Supported
- **Import**: CSV
- **Export**: CSV, Excel (XLSX)

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for large files)
- Modern web browser (Chrome, Firefox, Edge, Safari)

---

**Need more help?** Check the README.md or open an issue on GitHub!
