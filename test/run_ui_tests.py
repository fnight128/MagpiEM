#!/usr/bin/env python3
"""
Simple script to run all automated UI tests.
"""

from test_dash_ui import run_tests

if __name__ == "__main__":
    print("Starting automated Dash UI tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1)
