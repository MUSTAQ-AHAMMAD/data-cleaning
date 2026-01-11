# 🧹 AI CRM - Data Cleaning System

A powerful, user-friendly Python-based CRM application for efficient CSV data cleaning with AI-powered duplicate detection.

## 🌟 Features

### 🎯 Core Capabilities
- **AI-Powered Duplicate Detection**: Uses fuzzy matching algorithms to detect similar records
- **Exact Match Detection**: Find identical duplicate records across specified columns
- **Smart Data Cleaning**: Remove duplicates while preserving data integrity
- **Missing Value Handling**: Multiple strategies for handling missing data
- **Data Standardization**: Normalize text data (lowercase, uppercase, title case, trim)
- **Real-time Reporting**: Comprehensive analytics and insights

### 📊 User Interface
- **Web-Based Dashboard**: Built with Streamlit for easy access
- **Interactive Data Preview**: View and analyze your data before cleaning
- **Visual Analytics**: Charts and graphs showing data quality metrics
- **Multi-Format Export**: Export cleaned data as CSV or Excel
- **Progress Tracking**: Real-time updates on cleaning operations

### 🔍 Duplicate Detection Methods

1. **Exact Match**
   - Identifies identical records
   - Fast and efficient
   - Perfect for finding exact duplicates

2. **Fuzzy Match (AI-Powered)**
   - Uses Levenshtein distance algorithm
   - Detects similar records with typos or variations
   - Configurable similarity threshold (50-100%)
   - Ideal for real-world messy data

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MUSTAQ-AHAMMAD/data-cleaning.git
cd data-cleaning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to:
```
http://localhost:8501
```

## 📖 Usage Guide

### 1. Upload Data
- Navigate to the "Upload Data" page
- Upload your CSV file or generate sample data
- Review data preview and quality metrics

### 2. Detect Duplicates
- Go to "Detect Duplicates" page
- Choose detection method:
  - **Exact Match**: For identical records
  - **Fuzzy Match**: For similar records with variations
- Select columns to compare
- Configure similarity threshold (for fuzzy matching)
- Click "Detect" to find duplicates

### 3. Clean Data
- Navigate to "Clean Data" page
- Remove detected duplicates
- Handle missing values with various strategies:
  - Drop rows with missing values
  - Fill with empty strings
  - Fill numeric columns with mean
  - Fill categorical columns with mode
- Standardize text data (lowercase, uppercase, etc.)

### 4. View Reports
- Check "Reports & Analytics" page for:
  - Cleaning statistics
  - Data quality score
  - Visual analytics
  - Detailed processing logs

### 5. Export Data
- Go to "Export Data" page
- Choose export format (CSV or Excel)
- Download your cleaned data

## 🎨 Features in Detail

### Data Quality Metrics
- **Data Completeness**: Percentage of non-null values
- **Missing Values Analysis**: Column-wise missing data visualization
- **Data Type Distribution**: Overview of data types in your dataset

### Cleaning Report
- Original record count
- Current record count
- Duplicates found and removed
- Cleaning method used
- Columns analyzed
- Processing timestamp

### Export Options
- **CSV Format**: Standard comma-separated values
- **Excel Format**: XLSX with formatted sheets
- Timestamped filenames for version control

## 🔧 Technical Details

### Technology Stack
- **Python 3.8+**: Core programming language
- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation and analysis
- **FuzzyWuzzy**: Fuzzy string matching
- **python-Levenshtein**: Fast string similarity
- **Plotly**: Interactive visualizations
- **OpenPyXL**: Excel file handling

### Algorithms Used

#### Exact Match Detection
- Uses pandas `duplicated()` function
- O(n) time complexity
- Compares all columns or specified subset

#### Fuzzy Match Detection
- Levenshtein distance algorithm
- Configurable similarity threshold
- Groups similar records together
- Returns similarity scores for each match

### Data Processing Pipeline
1. **Load**: Read CSV with automatic encoding detection
2. **Analyze**: Calculate data quality metrics
3. **Detect**: Find duplicates using selected method
4. **Clean**: Remove duplicates and handle missing values
5. **Standardize**: Normalize text data
6. **Export**: Save cleaned data in desired format

## 📁 Project Structure

```
data-cleaning/
├── app.py                 # Main Streamlit application
├── data_cleaner.py        # Core data cleaning module
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**MUSTAQ AHAMMAD**

## 🐛 Bug Reports & Feature Requests

If you encounter any issues or have suggestions for improvements, please open an issue on GitHub.

## 📚 Example Use Cases

1. **Customer Database Cleanup**
   - Remove duplicate customer records
   - Standardize names and addresses
   - Handle incomplete contact information

2. **User Registration Data**
   - Detect duplicate user accounts
   - Clean email addresses
   - Standardize phone numbers

3. **Marketing Lists**
   - Deduplicate contact lists
   - Merge similar records
   - Clean and standardize data for campaigns

4. **Sales Data**
   - Remove duplicate transactions
   - Clean customer information
   - Prepare data for analysis

## 🎓 Tips for Best Results

- **Column Selection**: Select the most relevant columns for duplicate detection
- **Fuzzy Threshold**: Start with 85% and adjust based on results
  - Higher (90-100%): Very strict matching
  - Medium (80-90%): Balanced approach
  - Lower (70-80%): More lenient, may catch more variations
- **Preview First**: Always review detected duplicates before removing
- **Backup Data**: Keep a copy of original data before cleaning

## 🔮 Future Enhancements

- Machine learning-based duplicate detection
- Batch processing for multiple files
- Scheduled cleaning jobs
- API integration
- More export formats (JSON, XML)
- Advanced data validation rules
- Custom cleaning scripts support

## 📞 Support

For support and questions, please open an issue on GitHub or contact the maintainer.

---

Made with ❤️ for better data quality
