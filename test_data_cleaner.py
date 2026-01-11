"""
Test suite for Data Cleaner module
"""

import pandas as pd
import pytest
from data_cleaner import DataCleaner
import tempfile
import os


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def setup_method(self):
        """Set up test data before each test."""
        self.cleaner = DataCleaner()
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'Name': ['John Smith', 'Jane Doe', 'John Smith', 'Bob Johnson', 'john smith'],
            'Email': ['john@email.com', 'jane@email.com', 'john@email.com', 'bob@email.com', 'john@email.com'],
            'Phone': ['555-0001', '555-0002', '555-0001', '555-0003', '555-0001'],
            'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'New York']
        })
    
    def test_load_dataframe(self):
        """Test loading data from DataFrame."""
        df = self.cleaner.load_dataframe(self.test_data)
        assert df is not None
        assert len(df) == 5
        assert self.cleaner.cleaning_report['total_records'] == 5
    
    def test_detect_duplicates_exact(self):
        """Test exact duplicate detection."""
        self.cleaner.load_dataframe(self.test_data)
        duplicates, count = self.cleaner.detect_duplicates_exact()
        
        # Should find at least 1 exact duplicate
        assert count >= 1
        assert len(duplicates) >= 1
    
    def test_detect_duplicates_exact_with_columns(self):
        """Test exact duplicate detection with specific columns."""
        self.cleaner.load_dataframe(self.test_data)
        duplicates, count = self.cleaner.detect_duplicates_exact(columns=['Email'])
        
        # Should find duplicates based on email only
        assert count >= 2
    
    def test_detect_duplicates_fuzzy(self):
        """Test fuzzy duplicate detection."""
        self.cleaner.load_dataframe(self.test_data)
        duplicate_groups, count = self.cleaner.detect_duplicates_fuzzy(
            columns=['Name'],
            threshold=80
        )
        
        # Should find 'John Smith', 'john smith' as similar
        assert count > 0
        assert len(duplicate_groups) > 0
    
    def test_remove_exact_duplicates(self):
        """Test removing exact duplicates."""
        self.cleaner.load_dataframe(self.test_data)
        self.cleaner.detect_duplicates_exact()
        cleaned_df = self.cleaner.remove_exact_duplicates()
        
        # Should have fewer records after removing duplicates
        assert len(cleaned_df) < len(self.test_data)
        assert self.cleaner.cleaning_report['records_removed'] > 0
    
    def test_handle_missing_values_drop(self):
        """Test handling missing values by dropping."""
        # Add some NaN values
        test_data_with_nan = self.test_data.copy()
        test_data_with_nan.loc[0, 'City'] = None
        
        self.cleaner.load_dataframe(test_data_with_nan)
        cleaned_df = self.cleaner.handle_missing_values(strategy='drop')
        
        # Should have dropped the row with NaN
        assert len(cleaned_df) < len(test_data_with_nan)
    
    def test_handle_missing_values_fill_empty(self):
        """Test handling missing values by filling with empty string."""
        test_data_with_nan = self.test_data.copy()
        test_data_with_nan.loc[0, 'City'] = None
        
        self.cleaner.load_dataframe(test_data_with_nan)
        cleaned_df = self.cleaner.handle_missing_values(strategy='fill_empty')
        
        # Should have filled NaN with empty string
        assert cleaned_df.loc[0, 'City'] == ''
        assert len(cleaned_df) == len(test_data_with_nan)
    
    def test_standardize_data_lowercase(self):
        """Test data standardization to lowercase."""
        self.cleaner.load_dataframe(self.test_data)
        cleaned_df = self.cleaner.standardize_data(columns=['Name'], operation='lowercase')
        
        # All names should be lowercase
        assert all(cleaned_df['Name'].str.islower())
    
    def test_standardize_data_uppercase(self):
        """Test data standardization to uppercase."""
        self.cleaner.load_dataframe(self.test_data)
        cleaned_df = self.cleaner.standardize_data(columns=['Name'], operation='uppercase')
        
        # All names should be uppercase
        assert all(cleaned_df['Name'].str.isupper())
    
    def test_get_data_summary(self):
        """Test getting data summary."""
        self.cleaner.load_dataframe(self.test_data)
        summary = self.cleaner.get_data_summary()
        
        assert summary['total_rows'] == 5
        assert summary['total_columns'] == 4
        assert 'columns' in summary
        assert 'missing_values' in summary
    
    def test_get_cleaning_report(self):
        """Test getting cleaning report."""
        self.cleaner.load_dataframe(self.test_data)
        report = self.cleaner.get_cleaning_report()
        
        assert 'total_records' in report
        assert report['total_records'] == 5
        assert 'timestamp' in report
    
    def test_export_cleaned_data_csv(self):
        """Test exporting data to CSV."""
        self.cleaner.load_dataframe(self.test_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name
        
        try:
            output_path = self.cleaner.export_cleaned_data(tmp_path, format='csv')
            assert os.path.exists(output_path)
            
            # Verify the exported data
            exported_df = pd.read_csv(output_path)
            assert len(exported_df) == len(self.test_data)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_load_csv_file(self):
        """Test loading CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as tmp:
            self.test_data.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            df = self.cleaner.load_csv(tmp_path)
            assert df is not None
            assert len(df) == len(self.test_data)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_fuzzy_matching_threshold(self):
        """Test fuzzy matching with different thresholds."""
        self.cleaner.load_dataframe(self.test_data)
        
        # High threshold - should find fewer matches
        groups_high, count_high = self.cleaner.detect_duplicates_fuzzy(
            columns=['Name'],
            threshold=95
        )
        
        # Low threshold - should find more matches
        groups_low, count_low = self.cleaner.detect_duplicates_fuzzy(
            columns=['Name'],
            threshold=70
        )
        
        # Lower threshold should find at least as many duplicates as high threshold
        assert count_low >= count_high


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
