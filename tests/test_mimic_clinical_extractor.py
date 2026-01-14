"""
Quick test of mimic_clinical_extractor.py
Run this to verify the module loads and basic functionality works.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from colab_src.data_pipeline.mimic_clinical_extractor import MIMICClinicalExtractor

def test_import():
    """Test that the module imports successfully."""
    print("✅ Module imported successfully")
    return True

def test_initialization():
    """Test that the class initializes without errors."""
    extractor = MIMICClinicalExtractor(
        mimic_clinical_dir=Path('dummy_path'),
        cache_dir=Path('data/cache')
    )
    print("✅ MIMICClinicalExtractor initialized")
    print(f"   Cache dir: {extractor.cache_dir}")
    return True

def test_cache_dir_creation():
    """Test that cache directory is created."""
    cache_dir = Path('data/cache')
    extractor = MIMICClinicalExtractor(
        mimic_clinical_dir=Path('dummy_path'),
        cache_dir=cache_dir
    )
    if cache_dir.exists():
        print(f"✅ Cache directory created: {cache_dir}")
        return True
    else:
        print(f"❌ Cache directory not found")
        return False

if __name__ == '__main__':
    print("Testing mimic_clinical_extractor.py\n")
    
    try:
        test_import()
        test_initialization()
        test_cache_dir_creation()
        print("\n✅ All basic tests passed!")
        print("\nNext steps:")
        print("1. Review the mimic_clinical_extractor.py implementation")
        print("2. Prepare to test with actual MIMIC CSV files")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
