"""Test configuration and execution management for the AI architecture.

This module provides test configuration, fixtures, and execution utilities.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pytest


def run_test_suite(test_type: str = "all", 
                  verbose: bool = True,
                  coverage: bool = True,
                  markers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run specific test suite with configuration."""
    
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests"
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test directory
    if test_type == "unit":
        cmd.append(str(test_dir / "unit"))
    elif test_type == "integration":  
        cmd.append(str(test_dir / "integration"))
    elif test_type == "e2e":
        cmd.append(str(test_dir / "e2e"))
    elif test_type == "performance":
        cmd.append(str(test_dir / "performance"))
    else:  # all
        cmd.append(str(test_dir))
    
    # Add verbose output
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage reporting
    if coverage:
        cmd.extend([
            "--cov=ai_architect_demo",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Additional pytest options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker validation  
        "--durations=10",  # Show 10 slowest tests
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "command": " ".join(cmd)
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Test execution timed out after 30 minutes",
            "returncode": -1,
            "command": " ".join(cmd)
        }


def run_quick_tests() -> Dict[str, Any]:
    """Run quick test suite (unit tests only, no slow markers)."""
    return run_test_suite(
        test_type="unit",
        verbose=True,
        coverage=False,
        markers=["not slow"]
    )


def run_full_test_suite() -> Dict[str, Any]:
    """Run complete test suite including performance tests."""
    return run_test_suite(
        test_type="all",
        verbose=True,
        coverage=True,
        markers=None
    )


def run_smoke_tests() -> Dict[str, Any]:
    """Run smoke tests to verify basic functionality."""
    return run_test_suite(
        test_type="all",
        verbose=True,
        coverage=False,
        markers=["smoke"]
    )


def run_performance_tests() -> Dict[str, Any]:
    """Run performance and load tests."""
    return run_test_suite(
        test_type="performance",
        verbose=True,
        coverage=False,
        markers=["performance or load or stress or benchmark"]
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AI Architecture test suites")
    parser.add_argument(
        "suite",
        choices=["unit", "integration", "e2e", "performance", "all", "quick", "smoke"],
        default="quick",
        help="Test suite to run"
    )
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--markers", nargs="+", help="Pytest markers to include")
    
    args = parser.parse_args()
    
    # Map suite names to functions
    suite_functions = {
        "quick": run_quick_tests,
        "smoke": run_smoke_tests,
        "performance": run_performance_tests,
        "all": run_full_test_suite
    }
    
    if args.suite in suite_functions:
        result = suite_functions[args.suite]()
    else:
        result = run_test_suite(
            test_type=args.suite,
            verbose=not args.quiet,
            coverage=not args.no_coverage,
            markers=args.markers
        )
    
    # Print results
    print("\n" + "="*80)
    print(f"TEST SUITE: {args.suite.upper()}")
    print("="*80)
    
    if result["success"]:
        print("✅ TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
    
    print(f"\nReturn code: {result['returncode']}")
    
    if result["stdout"]:
        print("\nSTDOUT:")
        print(result["stdout"])
    
    if result["stderr"]:
        print("\nSTDERR:")
        print(result["stderr"])
    
    # Exit with test result code
    sys.exit(0 if result["success"] else 1)