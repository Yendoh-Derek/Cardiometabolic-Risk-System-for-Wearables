#!/usr/bin/env python3
"""
Phase 5A Completion Verification Script
Validates all critical fixes and readiness for Phase 5B
"""

import sys
from pathlib import Path

def verify_files_exist():
    """Verify all Phase 5A files exist."""
    files = {
        'Core Models': [
            'colab_src/models/ssl/encoder.py',
            'colab_src/models/ssl/decoder.py',
            'colab_src/models/ssl/config.py',
            'colab_src/models/ssl/losses.py',
            'colab_src/models/ssl/dataloader.py',
            'colab_src/models/ssl/trainer.py',
        ],
        'Data Pipeline': [
            'colab_src/data_pipeline/generate_mimic_windows.py',
        ],
        'Configuration': [
            'configs/ssl_pretraining.yaml',
        ],
        'Tests': [
            'tests/test_phase5a_comprehensive.py',
        ],
        'Documentation': [
            'docs/PHASE_5A_COMPLETION.md',
            'docs/PHASE_5B_QUICKREF.md',
            'docs/PHASE_5A_5B_INDEX.md',
            'docs/PHASE_5A_SUMMARY.md',
        ]
    }
    
    print("\n" + "="*70)
    print("VERIFYING PHASE 5A FILES".center(70))
    print("="*70)
    
    all_exist = True
    for category, file_list in files.items():
        print(f"\n{category}:")
        for filepath in file_list:
            path = Path(filepath)
            exists = path.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {filepath}")
            if not exists:
                all_exist = False
    
    return all_exist

def verify_config():
    """Verify critical fixes in config."""
    print("\n" + "="*70)
    print("VERIFYING CONFIG CRITICAL FIXES".center(70))
    print("="*70)
    
    try:
        from colab_src.models.ssl.config import SSLConfig
        config = SSLConfig.from_yaml('configs/ssl_pretraining.yaml')
        
        checks = {
            "signal_length == 1250": config.data.signal_length == 1250,
            "num_blocks == 3": config.model.num_blocks == 3,
            "batch_size == 128": config.training.batch_size == 128,
            "fft_pad_size == 2048": config.loss.fft_pad_size == 2048,
            "temporal_shift == 0.02": config.augmentation.temporal_shift_range == 0.02,
            "normalize_per_window == True": config.normalize_per_window == True,
            "sqi_threshold_train == 0.4": config.sqi_threshold_train == 0.4,
            "sqi_threshold_eval == 0.7": config.sqi_threshold_eval == 0.7,
        }
        
        all_pass = True
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}")
            if not result:
                all_pass = False
        
        return all_pass
    except Exception as e:
        print(f"  ❌ Error loading config: {e}")
        return False

def verify_imports():
    """Verify all critical modules import successfully."""
    print("\n" + "="*70)
    print("VERIFYING MODULE IMPORTS".center(70))
    print("="*70)
    
    imports = {
        "Encoder": "from colab_src.models.ssl.encoder import Encoder",
        "Decoder": "from colab_src.models.ssl.decoder import Decoder",
        "Config": "from colab_src.models.ssl.config import SSLConfig",
        "Losses": "from colab_src.models.ssl.losses import SSLLoss",
        "DataLoader": "from colab_src.models.ssl.dataloader import SSLDataLoader",
        "Trainer": "from colab_src.models.ssl.trainer import SSLTrainer",
        "Window Generator": "from colab_src.data_pipeline.generate_mimic_windows import MIMICWindowGenerator",
    }
    
    all_import = True
    for name, import_stmt in imports.items():
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")
            all_import = False
    
    return all_import

def verify_tests():
    """Check if test file exists and is valid."""
    print("\n" + "="*70)
    print("VERIFYING TEST SUITE".center(70))
    print("="*70)
    
    test_file = Path('tests/test_phase5a_comprehensive.py')
    if not test_file.exists():
        print("  ❌ Test file not found")
        return False
    
    print(f"  ✅ Test file exists ({test_file.stat().st_size / 1024:.1f} KB)")
    
    try:
        # Try to parse the test file
        with open(test_file) as f:
            content = f.read()
        
        test_count = content.count('def test_')
        print(f"  ✅ Found {test_count} test methods")
        
        return test_count == 11
    except Exception as e:
        print(f"  ❌ Error reading test file: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + "PHASE 5A COMPLETION VERIFICATION".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    checks = [
        ("Files Exist", verify_files_exist),
        ("Config Critical Fixes", verify_config),
        ("Module Imports", verify_imports),
        ("Test Suite", verify_tests),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error during {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY".center(70))
    print("="*70)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_pass = all(results.values())
    print("\n" + "="*70)
    if all_pass:
        print("✅ PHASE 5A VERIFICATION COMPLETE - READY FOR PHASE 5B".center(70))
    else:
        print("❌ PHASE 5A VERIFICATION FAILED - FIX ISSUES ABOVE".center(70))
    print("="*70 + "\n")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
