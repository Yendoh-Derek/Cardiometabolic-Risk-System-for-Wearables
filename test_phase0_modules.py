"""
Comprehensive test suite for Phase 0 modules.
Tests import, initialization, and basic functionality of all components.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from colab_src.data_pipeline import (
    MIMICClinicalExtractor,
    ClinicalLabelExtractor,
    WaveformClinicalLinker,
    DemographicProcessor,
    DatasetAssembler
)


def test_module_imports():
    """Test that all Phase 0 modules import successfully."""
    print("\n" + "="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    try:
        assert MIMICClinicalExtractor is not None
        assert ClinicalLabelExtractor is not None
        assert WaveformClinicalLinker is not None
        assert DemographicProcessor is not None
        assert DatasetAssembler is not None
        
        print("✅ All Phase 0 modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_clinical_extractor_init():
    """Test MIMICClinicalExtractor initialization."""
    print("\n" + "="*80)
    print("TEST 2: MIMICClinicalExtractor Initialization")
    print("="*80)
    
    try:
        extractor = MIMICClinicalExtractor(
            mimic_clinical_dir=Path('dummy'),
            cache_dir=Path('data/cache')
        )
        
        assert extractor.clinical_dir == Path('dummy')
        assert extractor.cache_dir == Path('data/cache')
        assert extractor.cache_dir.exists()
        
        print("✅ MIMICClinicalExtractor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_label_extractor_prefix_matching():
    """Test ICD-9 prefix matching logic."""
    print("\n" + "="*80)
    print("TEST 3: Clinical Label Extractor - ICD-9 Matching")
    print("="*80)
    
    try:
        extractor = ClinicalLabelExtractor()
        
        # Test diabetes codes
        assert extractor._match_icd9_prefix('250.02', ['250']) == True
        assert extractor._match_icd9_prefix('250.9', ['250']) == True
        assert extractor._match_icd9_prefix('251.1', ['250']) == False
        
        # Test hypertension codes
        assert extractor._match_icd9_prefix('401.9', ['401', '402', '403']) == True
        assert extractor._match_icd9_prefix('402.1', ['401', '402', '403']) == True
        assert extractor._match_icd9_prefix('404.0', ['401', '402', '403']) == False
        
        # Test obesity codes
        assert extractor._match_icd9_prefix('278.00', ['278']) == True
        assert extractor._match_icd9_prefix('279.0', ['278']) == False
        
        print("✅ ICD-9 prefix matching logic works correctly")
        return True
    except Exception as e:
        print(f"❌ Prefix matching test failed: {e}")
        return False


def test_label_extraction_with_sample_data():
    """Test label extraction with sample diagnosis data."""
    print("\n" + "="*80)
    print("TEST 4: Label Extraction with Sample Data")
    print("="*80)
    
    try:
        # Create sample diagnoses
        sample_diagnoses = pd.DataFrame({
            'hadm_id': [1, 1, 2, 2, 3],
            'icd9_code': ['250.02', '401.9', '278.00', '401.9', '401.9'],
            'seq_num': [1, 2, 1, 2, 1]
        })
        
        extractor = ClinicalLabelExtractor()
        labels = extractor.extract_labels_per_admission(sample_diagnoses)
        
        # Verify output
        assert len(labels) == 3, f"Expected 3 admissions, got {len(labels)}"
        assert 'hadm_id' in labels.columns
        assert 'diabetes' in labels.columns
        assert 'hypertension' in labels.columns
        assert 'obesity' in labels.columns
        
        # Check values
        admission_1_labels = labels[labels['hadm_id'] == 1].iloc[0]
        assert admission_1_labels['diabetes'] == 1, "Admission 1 should have diabetes"
        assert admission_1_labels['hypertension'] == 1, "Admission 1 should have hypertension"
        assert admission_1_labels['obesity'] == 0, "Admission 1 should not have obesity"
        
        print("✅ Label extraction with sample data successful")
        print(f"   Labels extracted: {labels}")
        return True
    except Exception as e:
        print(f"❌ Label extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_waveform_linker_init():
    """Test WaveformClinicalLinker initialization."""
    print("\n" + "="*80)
    print("TEST 5: WaveformClinicalLinker Initialization")
    print("="*80)
    
    try:
        linker = WaveformClinicalLinker(n_jobs=-1, batch_size=1000)
        
        assert linker.n_jobs == -1
        assert linker.batch_size == 1000
        
        print("✅ WaveformClinicalLinker initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_demographic_processor_init():
    """Test DemographicProcessor initialization."""
    print("\n" + "="*80)
    print("TEST 6: DemographicProcessor Initialization")
    print("="*80)
    
    try:
        processor = DemographicProcessor(n_jobs=-1)
        
        assert processor.n_jobs == -1
        
        print("✅ DemographicProcessor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_dataset_assembler_init():
    """Test DatasetAssembler initialization and directory creation."""
    print("\n" + "="*80)
    print("TEST 7: DatasetAssembler Initialization")
    print("="*80)
    
    try:
        assembler = DatasetAssembler(output_dir=Path('data/processed'))
        
        assert assembler.output_dir == Path('data/processed')
        assert assembler.output_dir.exists()
        
        print("✅ DatasetAssembler initialized successfully")
        print(f"   Output directory: {assembler.output_dir}")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_charlson_computation():
    """Test Charlson Index computation."""
    print("\n" + "="*80)
    print("TEST 8: Charlson Index Computation")
    print("="*80)
    
    try:
        # Create sample diagnoses
        sample_diagnoses = pd.DataFrame({
            'hadm_id': [1, 1, 2, 2, 3],
            'icd9_code': ['250.02', '410.1', '585.3', '401.9', '042.0'],
            'seq_num': [1, 2, 1, 2, 1]
        })
        
        extractor = ClinicalLabelExtractor()
        cci = extractor.compute_charlson_index(sample_diagnoses)
        
        # Verify output
        assert len(cci) == 3, f"Expected 3 admissions, got {len(cci)}"
        assert 'hadm_id' in cci.columns
        assert 'cci_score' in cci.columns
        
        # Check values
        admission_1_cci = cci[cci['hadm_id'] == 1].iloc[0]['cci_score']
        assert admission_1_cci > 0, "Admission 1 should have non-zero CCI"
        
        print("✅ Charlson Index computation successful")
        print(f"   CCI scores: {cci}")
        return True
    except Exception as e:
        print(f"❌ CCI computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 0 tests."""
    print("\n" + "="*80)
    print("PHASE 0 MODULE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Clinical Extractor Init", test_clinical_extractor_init),
        ("ICD-9 Prefix Matching", test_label_extractor_prefix_matching),
        ("Label Extraction", test_label_extraction_with_sample_data),
        ("Waveform Linker Init", test_waveform_linker_init),
        ("Demographic Processor Init", test_demographic_processor_init),
        ("Dataset Assembler Init", test_dataset_assembler_init),
        ("Charlson Index", test_charlson_computation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8s} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 0 is ready for implementation.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
