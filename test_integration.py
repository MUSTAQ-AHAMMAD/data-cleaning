#!/usr/bin/env python3
"""
Integration test to verify the fix for Windows installation issues.
Tests that the application works with and without ML dependencies.
"""

import sys
import subprocess
import importlib


def test_core_dependencies():
    """Test that core dependencies are available."""
    print("=" * 60)
    print("Testing Core Dependencies")
    print("=" * 60)
    
    required_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'Levenshtein',
        'fuzzywuzzy',
        'rapidfuzz',
        'openpyxl',
        'plotly',
        'pytest',
        'chardet'
    ]
    
    missing = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing core dependencies: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All core dependencies available")
        return True


def test_ml_dependencies():
    """Test if ML dependencies are available (optional)."""
    print("\n" + "=" * 60)
    print("Testing ML Dependencies (Optional)")
    print("=" * 60)
    
    ml_modules = [
        'sklearn',
        'scipy',
        'jellyfish',
        'phonetics',
        'ftfy',
        'recordlinkage'
    ]
    
    available = []
    missing = []
    for module in ml_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
            available.append(module)
        except ImportError:
            print(f"✗ {module}")
            missing.append(module)
    
    if available:
        print(f"\n✅ ML dependencies available: {len(available)}/{len(ml_modules)}")
    if missing:
        print(f"ℹ️  ML dependencies not installed: {', '.join(missing)}")
        print("   (This is okay - ML Advanced features will be disabled)")
    
    return True  # ML dependencies are optional


def test_data_cleaner_import():
    """Test that DataCleaner can be imported and used."""
    print("\n" + "=" * 60)
    print("Testing DataCleaner Import")
    print("=" * 60)
    
    try:
        from data_cleaner import DataCleaner, ML_AVAILABLE, RECORDLINKAGE_AVAILABLE
        print("✓ DataCleaner imported successfully")
        print(f"  - ML_AVAILABLE: {ML_AVAILABLE}")
        print(f"  - RECORDLINKAGE_AVAILABLE: {RECORDLINKAGE_AVAILABLE}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import DataCleaner: {e}")
        return False


def test_basic_functionality():
    """Test basic data cleaning functionality."""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    try:
        from data_cleaner import DataCleaner
        import pandas as pd
        
        # Create instance
        cleaner = DataCleaner()
        print("✓ DataCleaner instance created")
        
        # Create test data
        df = pd.DataFrame({
            'Name': ['John', 'John', 'Jane'],
            'Email': ['john@test.com', 'john@test.com', 'jane@test.com']
        })
        cleaner.load_dataframe(df)
        print("✓ Test data loaded")
        
        # Test exact duplicate detection
        duplicates_df, num_duplicates = cleaner.detect_duplicates_exact()
        print(f"✓ Exact duplicate detection works (found {num_duplicates} duplicates)")
        
        # Test fuzzy duplicate detection
        groups, num_dups = cleaner.detect_duplicates_fuzzy(['Name'], threshold=80)
        print(f"✓ Fuzzy duplicate detection works (found {num_dups} duplicates)")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_advanced_graceful_failure():
    """Test that ML Advanced fails gracefully when dependencies are missing."""
    print("\n" + "=" * 60)
    print("Testing ML Advanced Graceful Failure")
    print("=" * 60)
    
    try:
        from data_cleaner import DataCleaner, ML_AVAILABLE
        import pandas as pd
        
        cleaner = DataCleaner()
        df = pd.DataFrame({
            'Name': ['John', 'Jane'],
            'Email': ['john@test.com', 'jane@test.com']
        })
        cleaner.load_dataframe(df)
        
        if not ML_AVAILABLE:
            # Should raise ImportError with helpful message
            try:
                cleaner.detect_duplicates_ml_advanced()
                print("✗ ML Advanced should have raised ImportError")
                return False
            except ImportError as e:
                if "requirements-ml.txt" in str(e):
                    print("✓ ML Advanced shows helpful error message when dependencies missing")
                    return True
                else:
                    print(f"✗ Error message not helpful: {e}")
                    return False
        else:
            # ML is available, test it works
            groups, num_dups, uncleaned = cleaner.detect_duplicates_ml_advanced(learn_from_data=False)
            print(f"✓ ML Advanced works (found {num_dups} duplicates)")
            return True
            
    except Exception as e:
        print(f"✗ Test failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Windows Installation Fix")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test core dependencies
    results.append(("Core Dependencies", test_core_dependencies()))
    
    # Test ML dependencies (optional)
    results.append(("ML Dependencies", test_ml_dependencies()))
    
    # Test DataCleaner import
    results.append(("DataCleaner Import", test_data_cleaner_import()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test ML Advanced graceful failure
    results.append(("ML Advanced Graceful Failure", test_ml_advanced_graceful_failure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The fix is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
