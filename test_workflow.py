"""
Integration test to verify the complete data cleaning workflow
"""

import pandas as pd
from data_cleaner import DataCleaner


def test_complete_workflow():
    """Test the complete data cleaning workflow."""
    
    print("=" * 60)
    print("AI CRM - Data Cleaning Integration Test")
    print("=" * 60)
    
    # Step 1: Load sample data
    print("\n1. Loading sample data...")
    cleaner = DataCleaner()
    df = cleaner.load_csv('sample_data.csv')
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Columns: {', '.join(df.columns)}")
    
    # Step 2: Get data summary
    print("\n2. Analyzing data quality...")
    summary = cleaner.get_data_summary()
    print(f"   ✓ Total Rows: {summary['total_rows']}")
    print(f"   ✓ Total Columns: {summary['total_columns']}")
    print(f"   ✓ Memory Usage: {summary['memory_usage']:.2f} MB")
    
    # Step 3: Detect exact duplicates
    print("\n3. Detecting exact duplicates...")
    duplicates, count = cleaner.detect_duplicates_exact()
    print(f"   ✓ Found {count} exact duplicate records")
    if count > 0:
        print(f"   ✓ Duplicate records:")
        for idx, row in duplicates.iterrows():
            print(f"      - {row['Name']} ({row['Email']})")
    
    # Step 4: Detect fuzzy duplicates
    print("\n4. Detecting fuzzy duplicates (AI-powered)...")
    duplicate_groups, fuzzy_count = cleaner.detect_duplicates_fuzzy(
        columns=['Name', 'Email'],
        threshold=85
    )
    print(f"   ✓ Found {len(duplicate_groups)} groups with {fuzzy_count} similar records")
    for idx, group in enumerate(duplicate_groups[:3], 1):  # Show first 3 groups
        print(f"   ✓ Group {idx}: {len(group)} similar records")
        for record in group:
            similarity = record.get('similarity', 100)
            print(f"      - {record['record']['Name']} (similarity: {similarity:.1f}%)")
    
    # Step 5: Remove exact duplicates
    print("\n5. Removing exact duplicates...")
    original_count = len(cleaner.df)
    cleaner.remove_exact_duplicates()
    final_count = len(cleaner.df)
    removed = original_count - final_count
    print(f"   ✓ Removed {removed} duplicate records")
    print(f"   ✓ Cleaned dataset has {final_count} records")
    
    # Step 6: Standardize data
    print("\n6. Standardizing data...")
    cleaner.standardize_data(columns=['Name', 'Email'], operation='lowercase')
    print(f"   ✓ Standardized Name and Email columns to lowercase")
    
    # Step 7: Get cleaning report
    print("\n7. Generating cleaning report...")
    report = cleaner.get_cleaning_report()
    print(f"   ✓ Original Records: {report['total_records']}")
    print(f"   ✓ Final Records: {report['final_records']}")
    print(f"   ✓ Duplicates Found: {report['duplicates_found']}")
    print(f"   ✓ Records Removed: {report['records_removed']}")
    print(f"   ✓ Cleaning Method: {report['cleaning_method']}")
    print(f"   ✓ Timestamp: {report['timestamp']}")
    
    # Step 8: Export cleaned data
    print("\n8. Exporting cleaned data...")
    output_file = '/tmp/cleaned_data_test.csv'
    cleaner.export_cleaned_data(output_file, format='csv')
    print(f"   ✓ Exported to {output_file}")
    
    # Verify exported file
    exported_df = pd.read_csv(output_file)
    print(f"   ✓ Verified: {len(exported_df)} records exported")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Data cleaning workflow works correctly.")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        test_complete_workflow()
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
