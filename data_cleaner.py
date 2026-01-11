"""
AI CRM Data Cleaning Module
Handles CSV file processing, duplicate detection, and data cleaning operations.
Advanced ML-based duplicate detection that learns and improves with data.
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rapid_fuzz, process
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re
import pickle
import os

# Advanced ML imports - These are optional for basic functionality
# If not available, ML Advanced features will be disabled
ML_AVAILABLE = True
ML_IMPORT_ERROR = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    import jellyfish
    import phonetics
    import ftfy
except ImportError as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)
    # Provide stub objects to prevent immediate crashes
    TfidfVectorizer = None
    DBSCAN = None
    StandardScaler = None

# Record linkage is optional - requires native compilation on Windows
RECORDLINKAGE_AVAILABLE = True
RECORDLINKAGE_IMPORT_ERROR = None

try:
    import recordlinkage
    from recordlinkage.index import Block
except ImportError as e:
    RECORDLINKAGE_AVAILABLE = False
    RECORDLINKAGE_IMPORT_ERROR = str(e)


class DataCleaner:
    """Main class for data cleaning operations with AI-powered duplicate detection."""
    
    def __init__(self):
        self.df = None
        self.original_df = None
        self.duplicates = []
        self.cleaned_data = None
        self.uncleaned_data = None
        
        # ML model components that learn and improve
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.learned_patterns = []
        self.model_path = 'ml_models'
        self.performance_history = []
        
        self.cleaning_report = {
            'total_records': 0,
            'duplicates_found': 0,
            'records_removed': 0,
            'cleaning_method': '',
            'timestamp': None,
            'columns_analyzed': [],
            'match_details': [],
            'confidence_scores': [],
            'auto_detection_results': {},
            'ml_model_version': None,
            'learning_stats': {}
        }
        
        # Load existing ML models if available
        self._load_ml_models()
    
    def _load_ml_models(self):
        """Load pre-trained ML models if they exist."""
        try:
            if os.path.exists(os.path.join(self.model_path, 'tfidf_vectorizer.pkl')):
                with open(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
            
            if os.path.exists(os.path.join(self.model_path, 'learned_patterns.pkl')):
                with open(os.path.join(self.model_path, 'learned_patterns.pkl'), 'rb') as f:
                    self.learned_patterns = pickle.load(f)
            
            if os.path.exists(os.path.join(self.model_path, 'performance_history.pkl')):
                with open(os.path.join(self.model_path, 'performance_history.pkl'), 'rb') as f:
                    self.performance_history = pickle.load(f)
        except Exception as e:
            # If loading fails, start fresh
            pass
    
    def _save_ml_models(self):
        """Save ML models for future use (learning persistence)."""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            if self.tfidf_vectorizer is not None:
                with open(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
            
            if self.learned_patterns:
                with open(os.path.join(self.model_path, 'learned_patterns.pkl'), 'wb') as f:
                    pickle.dump(self.learned_patterns, f)
            
            if self.performance_history:
                with open(os.path.join(self.model_path, 'performance_history.pkl'), 'wb') as f:
                    pickle.dump(self.performance_history, f)
        except Exception as e:
            # If saving fails, continue without saving
            pass
    
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
    
    def detect_duplicates_ml_advanced(self, learn_from_data: bool = True, 
                                      use_phonetic: bool = True,
                                      use_clustering: bool = True) -> Tuple[List[Dict], int, pd.DataFrame]:
        """
        Advanced ML-based duplicate detection that learns and improves with data.
        Uses multiple advanced techniques:
        - TF-IDF vectorization for semantic similarity
        - DBSCAN clustering for pattern detection
        - Phonetic matching (Soundex, Metaphone) for name variations
        - Record linkage algorithms
        - Learning from previous cleaning sessions
        
        Args:
            learn_from_data: If True, trains/updates ML models on this data
            use_phonetic: Use phonetic algorithms for name matching
            use_clustering: Use ML clustering to find duplicate groups
            
        Returns:
            Tuple of (duplicate_groups, num_duplicates, uncleaned_data_df)
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load a CSV file first.")
        
        # Check if ML dependencies are available
        if not ML_AVAILABLE:
            raise ImportError(
                f"ML Advanced features require additional dependencies that are not installed.\n"
                f"Error: {ML_IMPORT_ERROR}\n\n"
                f"To use ML Advanced features, install the optional ML dependencies:\n"
                f"  pip install -r requirements-ml.txt\n\n"
                f"Or use 'Smart AI (Automatic)' detection which works with core dependencies."
            )
        
        self.cleaning_report['cleaning_method'] = 'ML Advanced (Learning-Based)'
        
        # Step 1: Auto-identify columns
        column_types = self._identify_key_columns()
        analyze_columns = (column_types['identifiers'] + 
                          column_types['names'] + 
                          column_types['addresses'][:2])
        
        if not analyze_columns:
            text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
            analyze_columns = text_cols[:min(3, len(text_cols))]
        
        self.cleaning_report['columns_analyzed'] = analyze_columns
        
        # Step 2: Clean and normalize text data
        cleaned_texts = []
        for idx, row in self.df.iterrows():
            text_parts = []
            for col in analyze_columns:
                val = str(row.get(col, ''))
                # Use ftfy to fix text encoding issues
                val = ftfy.fix_text(val)
                val = self._normalize_text(val)
                text_parts.append(val)
            cleaned_texts.append(' '.join(text_parts))
        
        # Step 3: TF-IDF Vectorization for semantic similarity
        if self.tfidf_vectorizer is None or learn_from_data:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 3),
                analyzer='char_wb'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
            if learn_from_data:
                self._save_ml_models()
        else:
            try:
                tfidf_matrix = self.tfidf_vectorizer.transform(cleaned_texts)
            except:
                # If transformation fails, retrain
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=100,
                    ngram_range=(1, 3),
                    analyzer='char_wb'
                )
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        
        # Step 4: Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Step 5: Use DBSCAN clustering for pattern detection
        duplicate_groups = []
        
        if use_clustering:
            # Convert similarity to distance (ensure non-negative)
            distance_matrix = np.clip(1 - similarity_matrix, 0, None)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.25, min_samples=2, metric='precomputed')
            labels = clustering.fit_predict(distance_matrix)
            
            # Group by clusters
            cluster_groups = {}
            for idx, label in enumerate(labels):
                if label != -1:  # -1 means noise/no cluster
                    if label not in cluster_groups:
                        cluster_groups[label] = []
                    cluster_groups[label].append(idx)
            
            # Convert clusters to duplicate groups
            for cluster_id, indices in cluster_groups.items():
                if len(indices) > 1:
                    group_records = []
                    for idx in indices:
                        record = self.df.iloc[idx].to_dict()
                        
                        # Calculate average similarity within group
                        similarities = [similarity_matrix[idx][other_idx] 
                                      for other_idx in indices if other_idx != idx]
                        avg_similarity = np.mean(similarities) * 100 if similarities else 100
                        
                        # Add phonetic matching if enabled
                        phonetic_score = 100
                        if use_phonetic and column_types['names']:
                            phonetic_score = self._calculate_phonetic_similarity(
                                idx, indices[0], column_types['names']
                            )
                        
                        # Combined confidence score
                        confidence = (avg_similarity * 0.7 + phonetic_score * 0.3)
                        
                        group_records.append({
                            'index': idx,
                            'record': record,
                            'similarity': avg_similarity,
                            'phonetic_similarity': phonetic_score,
                            'confidence': confidence
                        })
                    
                    # Calculate group confidence
                    group_confidence = np.mean([r['confidence'] for r in group_records])
                    duplicate_groups.append({
                        'records': group_records,
                        'group_confidence': group_confidence,
                        'detection_method': 'ML_Clustering'
                    })
        
        # Step 6: Apply record linkage for additional verification
        if len(self.df) < 10000:  # Record linkage is slow on large datasets
            additional_groups = self._apply_record_linkage(column_types)
            
            # Merge with existing groups (avoid duplicates)
            existing_indices = set()
            for group in duplicate_groups:
                for record in group['records']:
                    existing_indices.add(record['index'])
            
            for new_group in additional_groups:
                new_indices = {r['index'] for r in new_group['records']}
                if not new_indices.intersection(existing_indices):
                    duplicate_groups.append(new_group)
                    existing_indices.update(new_indices)
        
        # Step 7: Separate high and low confidence matches
        high_conf_groups = []
        low_conf_groups = []
        
        for group in duplicate_groups:
            if group['group_confidence'] >= 65:  # Lower threshold for ML
                high_conf_groups.append(group)
            else:
                low_conf_groups.append(group)
        
        # Create uncleaned data
        uncleaned_indices = []
        for group in low_conf_groups:
            uncleaned_indices.extend([r['index'] for r in group['records']])
        
        if uncleaned_indices:
            self.uncleaned_data = self.df.iloc[uncleaned_indices].copy()
            self.uncleaned_data['reason'] = 'Low ML confidence - requires manual review'
        else:
            self.uncleaned_data = pd.DataFrame()
        
        # Step 8: Learn from this cleaning session
        if learn_from_data:
            self._learn_from_cleaning_session(duplicate_groups, column_types)
        
        # Count duplicates
        num_duplicates = sum(len(group['records']) - 1 for group in duplicate_groups)
        
        self.cleaning_report['duplicates_found'] = num_duplicates
        self.cleaning_report['ml_model_version'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.cleaning_report['learning_stats'] = {
            'total_sessions': len(self.performance_history) + 1,
            'patterns_learned': len(self.learned_patterns),
            'high_confidence_groups': len(high_conf_groups),
            'low_confidence_groups': len(low_conf_groups)
        }
        
        return duplicate_groups, num_duplicates, self.uncleaned_data
    
    def _calculate_phonetic_similarity(self, idx1: int, idx2: int, name_columns: List[str]) -> float:
        """Calculate phonetic similarity for name fields."""
        similarities = []
        
        for col in name_columns:
            if col not in self.df.columns:
                continue
            
            name1 = str(self.df.iloc[idx1][col]).strip()
            name2 = str(self.df.iloc[idx2][col]).strip()
            
            if not name1 or not name2:
                continue
            
            try:
                # Use multiple phonetic algorithms (with error handling)
                soundex_match = False
                metaphone_match = False
                
                try:
                    # Clean names for phonetic matching (only alphabetic)
                    clean_name1 = ''.join(c for c in name1 if c.isalpha() or c.isspace())
                    clean_name2 = ''.join(c for c in name2 if c.isalpha() or c.isspace())
                    
                    if clean_name1 and clean_name2:
                        soundex_match = phonetics.soundex(clean_name1) == phonetics.soundex(clean_name2)
                        metaphone_match = phonetics.metaphone(clean_name1) == phonetics.metaphone(clean_name2)
                except:
                    pass
                
                # Jaro-Winkler for name similarity
                jaro_sim = jellyfish.jaro_winkler_similarity(name1, name2) * 100
                
                # Combined score
                phonetic_score = (
                    (100 if soundex_match else 0) * 0.3 +
                    (100 if metaphone_match else 0) * 0.3 +
                    jaro_sim * 0.4
                )
                similarities.append(phonetic_score)
            except Exception as e:
                # If phonetic matching fails, use jaro-winkler only
                try:
                    jaro_sim = jellyfish.jaro_winkler_similarity(name1, name2) * 100
                    similarities.append(jaro_sim)
                except:
                    pass
        
        return np.mean(similarities) if similarities else 100
    
    def _apply_record_linkage(self, column_types: Dict[str, List[str]]) -> List[Dict]:
        """Apply record linkage algorithms for additional duplicate detection."""
        groups = []
        
        # Check if recordlinkage is available
        if not RECORDLINKAGE_AVAILABLE:
            # Skip record linkage if not available
            return groups
        
        try:
            # Create indexer
            indexer = recordlinkage.Index()
            
            # Use blocking on key fields if available
            if column_types['identifiers']:
                for col in column_types['identifiers'][:1]:  # Use first identifier
                    if col in self.df.columns:
                        indexer.add(Block(col))
            else:
                indexer.add(recordlinkage.index.Full())
            
            # Generate candidate pairs
            candidate_pairs = indexer.index(self.df)
            
            if len(candidate_pairs) == 0:
                return groups
            
            # Compare records
            compare = recordlinkage.Compare()
            
            # Add comparison features
            for col in column_types['names']:
                if col in self.df.columns:
                    compare.string(col, col, method='jarowinkler', label=f'{col}_sim')
            
            for col in column_types['identifiers']:
                if col in self.df.columns:
                    compare.exact(col, col, label=f'{col}_match')
            
            # Compute features
            features = compare.compute(candidate_pairs, self.df)
            
            # Find matches (average similarity > 0.85)
            matches = features[features.mean(axis=1) > 0.85]
            
            # Convert to groups
            processed = set()
            for idx1, idx2 in matches.index:
                if idx1 not in processed and idx2 not in processed:
                    group_records = [
                        {
                            'index': idx1,
                            'record': self.df.iloc[idx1].to_dict(),
                            'similarity': features.loc[(idx1, idx2)].mean() * 100,
                            'confidence': features.loc[(idx1, idx2)].mean() * 100
                        },
                        {
                            'index': idx2,
                            'record': self.df.iloc[idx2].to_dict(),
                            'similarity': features.loc[(idx1, idx2)].mean() * 100,
                            'confidence': features.loc[(idx1, idx2)].mean() * 100
                        }
                    ]
                    
                    groups.append({
                        'records': group_records,
                        'group_confidence': features.loc[(idx1, idx2)].mean() * 100,
                        'detection_method': 'RecordLinkage'
                    })
                    
                    processed.add(idx1)
                    processed.add(idx2)
        
        except Exception as e:
            # If record linkage fails, return empty
            pass
        
        return groups
    
    def _learn_from_cleaning_session(self, duplicate_groups: List[Dict], 
                                    column_types: Dict[str, List[str]]):
        """Learn patterns from this cleaning session to improve future performance."""
        # Extract patterns from detected duplicates
        for group in duplicate_groups:
            if group['group_confidence'] >= 80:  # Only learn from high-confidence matches
                pattern = {
                    'column_types': column_types,
                    'group_size': len(group['records']),
                    'avg_confidence': group['group_confidence'],
                    'detection_method': group.get('detection_method', 'unknown'),
                    'timestamp': datetime.now()
                }
                self.learned_patterns.append(pattern)
        
        # Keep only recent patterns (last 1000)
        if len(self.learned_patterns) > 1000:
            self.learned_patterns = self.learned_patterns[-1000:]
        
        # Track performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'duplicates_found': len(duplicate_groups),
            'avg_confidence': np.mean([g['group_confidence'] for g in duplicate_groups]) if duplicate_groups else 0,
            'data_size': len(self.df)
        })
        
        # Save models
        self._save_ml_models()
    
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
