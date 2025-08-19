# Automated UI Testing Framework for MagpiEM

This directory contains automated UI tests for the MagpiEM Dash application using Selenium WebDriver.

## Prerequisites

1. **Python Dependencies**: Install the required packages:
   ```bash
   pip install -r requirements_test.txt
   ```

2. **Chrome Browser**: The tests use Chrome WebDriver, so Chrome must be installed on your system.

3. **Test Data**: Ensure `WT_CA_2nd.mat` is present in the `test/` directory.

## Running Tests

### Run All Tests
```bash
python run_ui_tests.py
```

### Run Only Workflow Test
```bash
python run_workflow_test.py
```

### Run Specific Test Class
```bash
python -m unittest test_dash_ui.TestWorkflow
```

## Test Categories

### TestWorkflow
- **Purpose**: Tests the complete workflow from file upload to cleaning
- **Actions**: Upload file → Read 5 tomograms → Run cleaning → Verify completion
- **Browser**: Visible (not headless) for debugging

### TestBasicUI
- **Purpose**: Tests basic UI functionality
- **Actions**: Page loading, element presence, dropdown functionality

### TestFileUpload
- **Purpose**: Tests file upload functionality
- **Actions**: Upload test files, verify file handling

### TestGraphControls
- **Purpose**: Tests graph control interactions
- **Actions**: Toggle switches, set cone size, show/hide removed particles

### TestCleaningParameters
- **Purpose**: Tests cleaning parameter controls
- **Actions**: Set distance, orientation, curvature parameters

### TestNavigation
- **Purpose**: Tests navigation between tomograms
- **Actions**: Next/previous button functionality

### TestSaveFunctionality
- **Purpose**: Tests save functionality
- **Actions**: Set save filename, toggle keep particles

## Framework Features

### Debugging Tools
- **Screenshots**: Automatic screenshots on test failures
- **Page Source Analysis**: Detailed HTML inspection for element presence
- **Element Verification**: Comprehensive element existence checking
- **Meaningful Error Messages**: Descriptive assertion failures

### Robust Element Interaction
- **JavaScript Fallbacks**: Uses JavaScript for unreliable element interactions
- **Scrolling**: Automatically scrolls elements into view before interaction
- **Wait Strategies**: Multiple wait strategies for dynamic content
- **Error Recovery**: Graceful handling of interaction failures

### Server Management
- **Automatic Startup**: Starts Dash server in background process
- **Health Checks**: Verifies server accessibility before tests
- **Cleanup**: Proper server shutdown after tests

## Configuration

### Browser Options
- **Window Size**: 1920x1080
- **Headless Mode**: Disabled for debugging (can be enabled in `setup_driver()`)
- **Chrome Options**: Optimized for stability

### Timeouts
- **Server Startup**: 30 seconds
- **Page Load**: 30 seconds
- **Element Wait**: 10 seconds
- **Cleaning Process**: 5 minutes

## Troubleshooting

### Common Issues

1. **ChromeDriver Not Found**
   - Solution: The framework uses `webdriver-manager` to automatically download ChromeDriver
   - Ensure Chrome browser is installed

2. **Server Not Starting**
   - Check if port 8050 is available
   - Verify Python environment has all required packages
   - Check server logs in test output

3. **Elements Not Found**
   - Check screenshots in `test/screenshots/` directory
   - Review page source analysis in test output
   - Verify Dash app is running correctly

4. **Test Timeouts**
   - Increase timeout values in test configuration
   - Check system performance during tests
   - Verify network connectivity

### Debug Information

The framework provides extensive debugging information:
- Current URL and page title
- Element presence status
- Page source preview
- Screenshot paths
- Detailed error messages

### Screenshots

Screenshots are automatically saved to `test/screenshots/` directory:
- `debug_page_state.png`: Initial page state
- `missing_elements_error.png`: When elements are not found
- `*_error.png`: Various error conditions during tests

## Adding New Tests

1. **Create Test Class**: Inherit from `DashUITestCase`
2. **Use Helper Methods**: Leverage existing interaction methods
3. **Add Meaningful Assertions**: Include descriptive error messages
4. **Handle Errors Gracefully**: Use try-catch blocks for robust testing
5. **Add to Test Suite**: Update `run_tests()` function

## Example Test Structure

```python
class TestNewFeature(DashUITestCase):
    def test_new_functionality(self):
        """Test description."""
        try:
            # Test actions
            self.click_button("button-id")
            self.set_input_value("input-id", "value")
            
            # Verifications
            self.assertTrue(
                self.check_element_exists("element-id"),
                "Element should be present after action"
            )
            
        except Exception as e:
            screenshot_path = self.take_screenshot("test_failure.png")
            print(f"Test failed: {e}")
            print(f"Screenshot: {screenshot_path}")
            raise
```
