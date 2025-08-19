#!/usr/bin/env python3
"""
Simple script to run only the TestWorkflow test class.
"""

import unittest
from test_dash_ui import TestWorkflow

if __name__ == "__main__":
    print("Starting TestWorkflow test...")
    
    # Create test suite with only TestWorkflow
    test_suite = unittest.TestSuite()
    tests = unittest.TestLoader().loadTestsFromTestCase(TestWorkflow)
    test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("\n✅ TestWorkflow test passed!")
    else:
        print("\n❌ TestWorkflow test failed!")
    
    exit(0 if result.wasSuccessful() else 1)
