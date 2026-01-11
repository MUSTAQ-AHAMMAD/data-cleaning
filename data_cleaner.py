"""
AI CRM Data Cleaning Module
Handles CSV file processing, duplicate detection, and data cleaning operations.
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rapid_fuzz, process
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re


class DataCleaner:
    """Main class for data cleaning operations with AI-powered duplicate detection."""
    
    def __init__(self):
        self.df = None
        self.original_df = None
        self.duplicates = []
        self.cleaned_data = None
        self.uncleaned_data = None
        self.cleaning_report = {
            'total_records': 0,
            'duplicates_found': 0,
            'records_removed': 0,
            'cleaning_method': '',
            'timestamp': None,
            'columns_analyzed': [],
            'match_details': [],
            'confidence_scores': [],
            'auto_detection_results': {}
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
    
    def _identify_key_columns(self) -> Dict[str, List[str]]:
        """
        Automatically identify key columns for duplicate detection based on data patterns.
        Returns dictionary with column categories.
        """
        if self.df is None:
            return {}
        
        column_types = {
            'identifiers': [],  # ID, email, phone-like columns
            'names': [],  # Name columns
            'addresses': [],  # Address-like columns
            'numeric': [],  # Numeric columns
            'dates': [],  # Date columns
            'other': []  # Other text columns
        }
        
        for col in self.df.columns:
            col_lower = col.lower()
            sample_data = self.df[col].dropna().astype(str).head(10)
            
            # Check for ID patterns
            if any(keyword in col_lower for keyword in ['id', 'identifier', 'uid', 'key']):
                column_types['identifiers'].append(col)
            # Check for email patterns
            elif 'email' in col_lower or sample_data.str.contains('@', regex=False).any():
                column_types['identifiers'].append(col)
            # Check for phone patterns
            elif 'phone' in col_lower or sample_data.str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}', regex=True).any():
                column_types['identifiers'].append(col)
            # Check for name patterns
            elif any(keyword in col_lower for keyword in ['name', 'first', 'last', 'fname', 'lname', 'full']):
                column_types['names'].append(col)
            # Check for address patterns
            elif any(keyword in col_lower for keyword in ['address', 'street', 'city', 'state', 'zip', 'postal', 'country']):
                column_types['addresses'].append(col)
            # Check for date columns
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]) or 'date' in col_lower:
                column_types['dates'].append(col)
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                column_types['numeric'].append(col)
            else:
                column_types['other'].append(col)
        
        return column_types
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better comparison."""
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common punctuation
        text = re.sub(r'[.,;:!?]', '', text)
        return text
    
    def _calculate_field_similarity(self, val1: str, val2: str, field_type: str = 'general') -> float:
        """
        Calculate similarity score based on field type.
        Uses different algorithms for different field types.
        """
        val1 = self._normalize_text(val1)
        val2 = self._normalize_text(val2)
        
        if not val1 or not val2:
            return 0.0
        
        # For exact identifiers (emails, IDs), require exact match
        if field_type == 'identifier':
            return 100.0 if val1 == val2 else 0.0
        
        # For names, use token sort ratio (handles "John Smith" vs "Smith, John")
        elif field_type == 'name':
            return rapid_fuzz.token_sort_ratio(val1, val2)
        
        # For addresses, use token set ratio (handles different word orders)
        elif field_type == 'address':
            return rapid_fuzz.token_set_ratio(val1, val2)
        
        # General text comparison
        else:
            # Use average of multiple similarity metrics for robustness
            ratio = rapid_fuzz.ratio(val1, val2)
            partial_ratio = rapid_fuzz.partial_ratio(val1, val2)
            token_ratio = rapid_fuzz.token_ratio(val1, val2)
            return (ratio + partial_ratio + token_ratio) / 3.0
    
    def detect_duplicates_smart_ai(self, auto_detect_columns: bool = True, 
                                   custom_columns: Optional[List[str]] = None,
                                   threshold: int = 80) -> Tuple[List[Dict], int, pd.DataFrame]:
        """
        Advanced AI-powered duplicate detection that automatically:
        1. Identifies the most relevant columns for comparison
        2. Applies different matching strategies based on field types
        3. Uses weighted scoring based on column importance
        4. Separates cleaned data from potentially problematic data
        
        Args:
            auto_detect_columns: Whether to automatically detect key columns
            custom_columns: Manual column selection (overrides auto-detection)
            threshold: Overall similarity threshold (0-100), default 80
            
        Returns:
            Tuple of (duplicate_groups, num_duplicates, uncleaned_data_df)
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        # Identify column types
        column_types = self._identify_key_columns()
        
        # Determine which columns to analyze
        if custom_columns:
            analyze_columns = custom_columns
            weights = {col: 1.0 for col in custom_columns}
        elif auto_detect_columns:
            # Prioritize identifiers and names for duplicate detection
            analyze_columns = (column_types['identifiers'] + 
                             column_types['names'] + 
                             column_types['addresses'][:2])  # Limit addresses
            
            if not analyze_columns:
                # Fallback to first 3 text columns
                text_cols = [col for col in self.df.columns 
                           if self.df[col].dtype == 'object']
                analyze_columns = text_cols[:min(3, len(text_cols))]
            
            # Assign weights (identifiers are most important)
            weights = {}
            for col in analyze_columns:
                if col in column_types['identifiers']:
                    weights[col] = 2.0  # Higher weight for identifiers
                elif col in column_types['names']:
                    weights[col] = 1.5  # Medium-high weight for names
                else:
                    weights[col] = 1.0  # Standard weight
        else:
            analyze_columns = list(self.df.columns)
            weights = {col: 1.0 for col in analyze_columns}
        
        self.cleaning_report['columns_analyzed'] = analyze_columns
        self.cleaning_report['cleaning_method'] = f'Smart AI Detection (auto-threshold={threshold}%)'
        self.cleaning_report['auto_detection_results'] = column_types
        
        # Detect duplicates with weighted scoring
        duplicate_groups = []
        checked_indices = set()
        confidence_scores = []
        
        df_records = self.df.to_dict('records')
        
        for i in range(len(df_records)):
            if i in checked_indices:
                continue
            
            current_record = df_records[i]
            similar_records = [{'index': i, 'record': current_record, 'confidence': 100.0}]
            
            for j in range(i + 1, len(df_records)):
                if j in checked_indices:
                    continue
                
                compare_record = df_records[j]
                
                # Calculate weighted similarity
                total_weighted_similarity = 0
                total_weight = 0
                field_scores = {}
                
                for col in analyze_columns:
                    val1 = current_record.get(col, '')
                    val2 = compare_record.get(col, '')
                    
                    # Determine field type for appropriate comparison
                    if col in column_types['identifiers']:
                        field_type = 'identifier'
                    elif col in column_types['names']:
                        field_type = 'name'
                    elif col in column_types['addresses']:
                        field_type = 'address'
                    else:
                        field_type = 'general'
                    
                    similarity = self._calculate_field_similarity(val1, val2, field_type)
                    weight = weights.get(col, 1.0)
                    
                    field_scores[col] = similarity
                    total_weighted_similarity += similarity * weight
                    total_weight += weight
                
                if total_weight > 0:
                    avg_weighted_similarity = total_weighted_similarity / total_weight
                    
                    # Calculate confidence (how sure we are about the match)
                    confidence = self._calculate_confidence(field_scores, column_types)
                    
                    if avg_weighted_similarity >= threshold:
                        similar_records.append({
                            'index': j,
                            'record': compare_record,
                            'similarity': avg_weighted_similarity,
                            'confidence': confidence,
                            'field_scores': field_scores
                        })
                        checked_indices.add(j)
            
            if len(similar_records) > 1:
                # Calculate group confidence
                group_confidence = np.mean([r['confidence'] for r in similar_records])
                duplicate_groups.append({
                    'records': similar_records,
                    'group_confidence': group_confidence
                })
                confidence_scores.append(group_confidence)
                checked_indices.add(i)
        
        # Separate high-confidence from low-confidence matches
        high_confidence_groups = []
        low_confidence_groups = []
        
        for group in duplicate_groups:
            if group['group_confidence'] >= 70:  # High confidence threshold
                high_confidence_groups.append(group)
            else:
                low_confidence_groups.append(group)
        
        # Create uncleaned data dataframe (low confidence duplicates need manual review)
        uncleaned_indices = []
        for group in low_confidence_groups:
            uncleaned_indices.extend([r['index'] for r in group['records']])
        
        if uncleaned_indices:
            self.uncleaned_data = self.df.iloc[uncleaned_indices].copy()
            self.uncleaned_data['reason'] = 'Low confidence duplicate detection - requires manual review'
        else:
            self.uncleaned_data = pd.DataFrame()
        
        # Count total duplicates
        all_groups = [g['records'] for g in duplicate_groups]
        num_duplicates = sum(len(group) - 1 for group in all_groups)
        
        self.cleaning_report['duplicates_found'] = num_duplicates
        self.cleaning_report['confidence_scores'] = confidence_scores
        self.cleaning_report['match_details'] = [
            {
                'group_id': idx,
                'group_size': len(group['records']),
                'group_confidence': group['group_confidence'],
                'similarity_scores': [r.get('similarity', 100) for r in group['records'][1:]]
            }
            for idx, group in enumerate(duplicate_groups)
        ]
        
        return duplicate_groups, num_duplicates, self.uncleaned_data
    
    def _calculate_confidence(self, field_scores: Dict[str, float], 
                            column_types: Dict[str, List[str]]) -> float:
        """
        Calculate confidence score based on field agreement.
        Higher confidence when key identifiers match.
        """
        if not field_scores:
            return 0.0
        
        confidence = 0.0
        weight_sum = 0.0
        
        for field, score in field_scores.items():
            # Higher confidence weight for identifiers
            if any(field in column_types.get(key, []) for key in ['identifiers']):
                weight = 3.0
            elif any(field in column_types.get(key, []) for key in ['names']):
                weight = 2.0
            else:
                weight = 1.0
            
            confidence += score * weight
            weight_sum += weight
        
        if weight_sum > 0:
            confidence = confidence / weight_sum
        
        return confidence
    
    def remove_smart_ai_duplicates(self, duplicate_groups: List[Dict], 
                                   high_confidence_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove duplicates detected by Smart AI, keeping the first record in each group.
        Optionally only remove high-confidence duplicates.
        
        Args:
            duplicate_groups: Groups from detect_duplicates_smart_ai
            high_confidence_only: If True, only remove high-confidence matches
            
        Returns:
            Tuple of (cleaned_data, uncleaned_data requiring manual review)
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        indices_to_remove = []
        
        for group in duplicate_groups:
            # Check if this is a dict with 'records' key (from smart AI)
            if isinstance(group, dict) and 'records' in group:
                records = group['records']
                group_confidence = group.get('group_confidence', 100)
                
                if high_confidence_only and group_confidence < 70:
                    continue  # Skip low confidence groups
                
                # Keep first record, remove others
                indices_to_remove.extend([r['index'] for r in records[1:]])
            else:
                # Legacy format support
                indices_to_remove.extend([record['index'] for record in group[1:]])
        
        # Create cleaned dataframe
        self.cleaned_data = self.df.drop(self.df.index[indices_to_remove]).reset_index(drop=True)
        self.cleaning_report['records_removed'] = len(indices_to_remove)
        
        return self.cleaned_data, self.uncleaned_data if self.uncleaned_data is not None else pd.DataFrame()
    
    def export_separated_data(self, cleaned_path: str, uncleaned_path: str = None, 
                             format: str = 'csv') -> Dict[str, str]:
        """
        Export cleaned and uncleaned data to separate files.
        
        Args:
            cleaned_path: Path for cleaned data
            uncleaned_path: Path for uncleaned data (optional)
            format: 'csv' or 'excel'
            
        Returns:
            Dictionary with file paths
        """
        result = {}
        
        # Export cleaned data
        if self.cleaned_data is not None and not self.cleaned_data.empty:
            if format == 'csv':
                self.cleaned_data.to_csv(cleaned_path, index=False)
            elif format == 'excel':
                self.cleaned_data.to_excel(cleaned_path, index=False)
            result['cleaned'] = cleaned_path
        
        # Export uncleaned data
        if uncleaned_path and self.uncleaned_data is not None and not self.uncleaned_data.empty:
            if format == 'csv':
                self.uncleaned_data.to_csv(uncleaned_path, index=False)
            elif format == 'excel':
                self.uncleaned_data.to_excel(uncleaned_path, index=False)
            result['uncleaned'] = uncleaned_path
        
        return result
    
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
