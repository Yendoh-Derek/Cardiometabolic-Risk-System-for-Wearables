#!/usr/bin/env python
"""Test runner script for SSL components."""
import sys
from pathlib import Path
import argparse
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_unit_tests(verbose=False):
    """Run unit tests."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", "not integration",
        "-v" if verbose else "",
        "--tb=short"
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("\n" + "="*80)
    print("RUNNING INTEGRATION TESTS")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration.py",
        "-v" if verbose else "",
        "--tb=short"
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_smoke.py",
        "tests/test_losses.py",
        "tests/test_config.py",
        "-v" if verbose else "",
        "--tb=short"
    ]
    
    if coverage:
        cmd.extend(["--cov=colab_src/models/ssl", "--cov-report=html"])
    
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_specific_test(test_name, verbose=False):
    """Run specific test."""
    print(f"\n" + "="*80)
    print(f"RUNNING TEST: {test_name}")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/test_{test_name}.py",
        "-v" if verbose else "",
        "--tb=short"
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SSL component tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit                   # Run unit tests only
  python run_tests.py --integration            # Run integration tests only
  python run_tests.py --test encoder           # Run encoder tests
  python run_tests.py --all --verbose          # Run all tests with verbose output
  python run_tests.py --all --coverage         # Generate coverage report
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test (e.g., 'encoder', 'decoder', 'losses')"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    # Ensure pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Install with: pip install pytest")
        return 1
    
    # Determine which tests to run
    if args.test:
        return_code = run_specific_test(args.test, args.verbose)
    elif args.unit:
        return_code = run_unit_tests(args.verbose)
    elif args.integration:
        return_code = run_integration_tests(args.verbose)
    elif args.all or not any([args.all, args.unit, args.integration, args.test]):
        return_code = run_all_tests(args.verbose, args.coverage)
    
    # Print summary
    print("\n" + "="*80)
    if return_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
