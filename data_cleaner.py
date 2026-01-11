"""
AI CRM Data Cleaning Module
Handles CSV file processing, duplicate detection, and data cleaning operations.
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple, Optional
import hashlib
from datetime import datetime


class DataCleaner:
    """Main class for data cleaning operations with AI-powered duplicate detection."""
    
    def __init__(self):
        self.df = None
        self.original_df = None
        self.duplicates = []
        self.cleaning_report = {
            'total_records': 0,
            'duplicates_found': 0,
            'records_removed': 0,
            'cleaning_method': '',
            'timestamp': None,
            'columns_analyzed': [],
            'match_details': []
        }
    
    def load_csv(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """Load CSV file into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(file_path, encoding=encoding)
            self.original_df = self.df.copy()
            self.cleaning_report['total_records'] = len(self.df)
            self.cleaning_report['timestamp'] = datetime.now()
            return self.df
        except UnicodeDecodeError:
            # Try different encodings
            for enc in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    self.df = pd.read_csv(file_path, encoding=enc)
                    self.original_df = self.df.copy()
                    self.cleaning_report['total_records'] = len(self.df)
                    self.cleaning_report['timestamp'] = datetime.now()
                    return self.df
                except:
                    continue
            raise
    
    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load data from an existing DataFrame."""
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_report['total_records'] = len(self.df)
        self.cleaning_report['timestamp'] = datetime.now()
        return self.df
    
    def detect_duplicates_exact(self, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
        """
        Detect exact duplicates based on specified columns.
        If no columns specified, uses all columns.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        if columns is None:
            subset = None
            self.cleaning_report['columns_analyzed'] = list(self.df.columns)
        else:
            subset = columns
            self.cleaning_report['columns_analyzed'] = columns
        
        # Find duplicates
        duplicate_mask = self.df.duplicated(subset=subset, keep='first')
        self.duplicates = self.df[duplicate_mask].copy()
        
        num_duplicates = len(self.duplicates)
        self.cleaning_report['duplicates_found'] = num_duplicates
        self.cleaning_report['cleaning_method'] = 'Exact Match'
        
        return self.duplicates, num_duplicates
    
    def detect_duplicates_fuzzy(self, columns: List[str], threshold: int = 85) -> Tuple[List[Dict], int]:
        """
        Detect fuzzy duplicates using Levenshtein distance.
        Returns potential duplicate groups based on similarity threshold.
        
        Args:
            columns: List of column names to compare
            threshold: Similarity threshold (0-100), default 85
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        duplicate_groups = []
        checked_indices = set()
        
        self.cleaning_report['columns_analyzed'] = columns
        self.cleaning_report['cleaning_method'] = f'Fuzzy Match (threshold={threshold}%)'
        
        # Convert dataframe to list for faster iteration
        df_records = self.df.to_dict('records')
        
        for i in range(len(df_records)):
            if i in checked_indices:
                continue
            
            current_record = df_records[i]
            similar_records = [{'index': i, 'record': current_record}]
            
            for j in range(i + 1, len(df_records)):
                if j in checked_indices:
                    continue
                
                compare_record = df_records[j]
                
                # Calculate similarity across specified columns
                total_similarity = 0
                valid_comparisons = 0
                
                for col in columns:
                    val1 = str(current_record.get(col, '')).strip().lower()
                    val2 = str(compare_record.get(col, '')).strip().lower()
                    
                    if val1 and val2:
                        similarity = fuzz.ratio(val1, val2)
                        total_similarity += similarity
                        valid_comparisons += 1
                
                if valid_comparisons > 0:
                    avg_similarity = total_similarity / valid_comparisons
                    
                    if avg_similarity >= threshold:
                        similar_records.append({
                            'index': j, 
                            'record': compare_record,
                            'similarity': avg_similarity
                        })
                        checked_indices.add(j)
            
            if len(similar_records) > 1:
                duplicate_groups.append(similar_records)
                checked_indices.add(i)
        
        # Count total duplicates (all records except the first one in each group)
        num_duplicates = sum(len(group) - 1 for group in duplicate_groups)
        self.cleaning_report['duplicates_found'] = num_duplicates
        self.cleaning_report['match_details'] = [
            {
                'group_id': idx,
                'group_size': len(group),
                'similarity_scores': [r.get('similarity', 100) for r in group[1:]]
            }
            for idx, group in enumerate(duplicate_groups)
        ]
        
        return duplicate_groups, num_duplicates
    
    def remove_exact_duplicates(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove exact duplicates and return cleaned DataFrame."""
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        original_count = len(self.df)
        
        if columns is None:
            self.df = self.df.drop_duplicates(keep='first')
        else:
            self.df = self.df.drop_duplicates(subset=columns, keep='first')
        
        records_removed = original_count - len(self.df)
        self.cleaning_report['records_removed'] = records_removed
        
        return self.df
    
    def remove_fuzzy_duplicates(self, duplicate_groups: List[Dict]) -> pd.DataFrame:
        """
        Remove fuzzy duplicates based on detected groups.
        Keeps the first record in each group.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        indices_to_remove = []
        for group in duplicate_groups:
            # Keep first record, remove others
            indices_to_remove.extend([record['index'] for record in group[1:]])
        
        self.df = self.df.drop(self.df.index[indices_to_remove])
        self.df = self.df.reset_index(drop=True)
        
        self.cleaning_report['records_removed'] = len(indices_to_remove)
        
        return self.df
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: 'drop' (remove rows), 'fill_mean' (numeric), 'fill_mode' (categorical), 'fill_empty' (empty string)
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill_mean':
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        elif strategy == 'fill_mode':
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    mode_value = self.df[col].mode()
                    if len(mode_value) > 0:
                        self.df[col].fillna(mode_value[0], inplace=True)
        elif strategy == 'fill_empty':
            self.df = self.df.fillna('')
        
        return self.df
    
    def standardize_data(self, columns: List[str], operation: str = 'lowercase') -> pd.DataFrame:
        """
        Standardize data in specified columns.
        
        Args:
            columns: List of columns to standardize
            operation: 'lowercase', 'uppercase', 'titlecase', 'strip'
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        for col in columns:
            if col in self.df.columns:
                if operation == 'lowercase':
                    self.df[col] = self.df[col].astype(str).str.lower()
                elif operation == 'uppercase':
                    self.df[col] = self.df[col].astype(str).str.upper()
                elif operation == 'titlecase':
                    self.df[col] = self.df[col].astype(str).str.title()
                elif operation == 'strip':
                    self.df[col] = self.df[col].astype(str).str.strip()
        
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        """Return the cleaning report with statistics."""
        report = self.cleaning_report.copy()
        if self.df is not None:
            report['final_records'] = len(self.df)
            report['records_kept'] = len(self.df)
        return report
    
    def export_cleaned_data(self, file_path: str, format: str = 'csv') -> str:
        """
        Export cleaned data to file.
        
        Args:
            file_path: Output file path
            format: 'csv' or 'excel'
        """
        if self.df is None:
            raise ValueError("No data to export.")
        
        if format == 'csv':
            self.df.to_csv(file_path, index=False)
        elif format == 'excel':
            self.df.to_excel(file_path, index=False)
        
        return file_path
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the current dataset."""
        if self.df is None:
            return {}
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return summary
