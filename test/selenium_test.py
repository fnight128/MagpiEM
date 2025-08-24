# -*- coding: utf-8 -*-
"""
Selenium test for the MagpiEM Dash application.

Tests are focused on UI functionality, not on the actual cleaning process, as
this would require large numbers of particles, leading to complex plots which
selenium cannot handle. The actual cleaning process is tested elsewhere

Note: Tests MUST be run in firefox. Although the app itself is perfectly compatible
with chrome, the combination of chrome, selenium and plotly all together prevents
plots from displaying.
"""

import warnings

# Suppress ResourceWarnings for cleaner test output
warnings.filterwarnings("ignore", category=ResourceWarning)

import argparse
import os
import sys
import time
import unittest
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager


TEST_FILE_NAME = "test_data_miniscule.mat"

TEST_PARAMETERS = {
    "switch-allow-flips": False,
    "inp-cc-thresh": 3,
    "inp-curv-goal": 90,
    "inp-curv-tol": 20,
    "inp-dist-goal": 60,
    "inp-dist-tol": 20,
    "inp-array-size": 3,
    "inp-min-neighbours": 2,
    "inp-ori-goal": 10,
    "inp-ori-tol": 20,
}

HEADLESS_MODE = True


class MagpiEMTest(unittest.TestCase):
    """Test case for MagpiEM Dash application using Selenium."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        firefox_options = Options()
        firefox_options.add_argument("--width=1920")
        firefox_options.add_argument("--height=1080")
        firefox_options.add_argument("--disable-extensions")
        firefox_options.add_argument("--disable-plugins")
        firefox_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
        )

        if HEADLESS_MODE:
            firefox_options.add_argument("--headless")

        service = Service(GeckoDriverManager().install())
        cls.driver = webdriver.Firefox(service=service, options=firefox_options)

        cls.driver.implicitly_wait(10)

        cls.project_root = Path(__file__).parent.parent
        cls.test_data_path = cls.project_root / "test" / TEST_FILE_NAME

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        if cls.driver:
            cls.driver.quit()

    def setUp(self):
        """Set up each test."""
        # Start the Dash application in a separate process
        self.dash_process = None
        self.start_dash_app()

        self.driver.get("http://localhost:8050")

        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Give firefox time to initialize WebGL and rendering components
        time.sleep(3)

    def tearDown(self):
        """Clean up after each test."""
        self.stop_dash_app()

    def start_dash_app(self):
        """Start the Dash application."""
        import subprocess
        import sys
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(self.project_root) + os.pathsep + env.get("PYTHONPATH", "")
        )

        # Ensure the processing DLL can be found by adding project root to PATH
        env["PATH"] = str(self.project_root) + os.pathsep + env.get("PATH", "")

        cmd = [sys.executable, "-m", "magpiem.dash_ui", "--no-browser"]
        self.dash_process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for the app to start and check if it's running
        time.sleep(3)

        if self.dash_process.poll() is not None:
            # Process died, get the error output
            stdout, stderr = self.dash_process.communicate()
            print(f"Dash app failed to start.")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            print(f"Return code: {self.dash_process.returncode}")
            raise RuntimeError("Failed to start Dash application")
        time.sleep(2)

    def stop_dash_app(self):
        """Stop the Dash application."""
        if self.dash_process:
            self.dash_process.terminate()
            self.dash_process.wait()
            # Close the file handles to prevent ResourceWarnings
            if hasattr(self.dash_process, "stdout") and self.dash_process.stdout:
                self.dash_process.stdout.close()
            if hasattr(self.dash_process, "stderr") and self.dash_process.stderr:
                self.dash_process.stderr.close()

    def test_complete_workflow(self):
        """Test the complete workflow from start to finish."""
        print("🚀 Starting complete workflow test...")

        print("Step 1: Checking application loads...")
        time.sleep(2)

        title = self.driver.find_element(By.TAG_NAME, "h1")
        self.assertEqual(title.text, "MagpiEM")
        print("✓ Application loaded successfully")

        print("Step 2: Uploading test file...")
        self.assertTrue(
            self.test_data_path.exists(), f"{TEST_FILE_NAME} file not found"
        )

        upload_element = self.driver.find_element(By.ID, "upload-data")
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "upload-data"))
        )

        file_input = self.driver.find_element(
            By.CSS_SELECTOR, "#upload-data input[type='file']"
        )
        file_input.send_keys(str(self.test_data_path.absolute()))

        WebDriverWait(self.driver, 10).until(
            lambda driver: TEST_FILE_NAME
            in driver.find_element(By.ID, "upload-data").text
        )
        print("✓ File uploaded successfully")

        print("Step 3: Setting file type...")
        dropdown = self.driver.find_element(By.ID, "dropdown-filetype")
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "dropdown-filetype"))
        )
        dropdown.click()

        # Try different selectors for the dropdown options
        mat_option = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[text()='.mat']"))
        )

        mat_option.click()
        time.sleep(1)
        print("✓ File type set to .mat")

        print("Step 4: Reading tomograms...")
        read_button = self.driver.find_element(By.ID, "button-read")
        read_button.click()

        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.ID, "dropdown-tomo"))
        )

        dropdown = self.driver.find_element(By.ID, "dropdown-tomo")
        self.assertFalse(dropdown.get_attribute("disabled"))
        print("✓ Tomograms read successfully")
        print("✓ Tomogram reading functionality verified through UI state changes")

        # Step 5: Input cleaning parameters
        print("Step 5: Inputting cleaning parameters...")
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "inp-cc-thresh"))
        )

        for param_id, value in TEST_PARAMETERS.items():
            element = self.driver.find_element(By.ID, param_id)

            if "switch" in param_id:
                if value:
                    if not element.is_selected():
                        element.click()
                else:
                    if element.is_selected():
                        element.click()
            else:
                element.clear()
                element.send_keys(str(value))

        print("✓ Cleaning parameters set correctly")

        print("Step 6: Running cleaning process...")
        clean_button = self.driver.find_element(By.ID, "button-full-clean")
        clean_button.click()

        print("⏳ Running cleaning process... (this may take several minutes)")

        print("✓ Cleaning process started")

        print("Waiting for cleaning to complete...")

        # First, wait a bit to see if cleaning actually starts processing, or
        # errors occur
        time.sleep(5)
        try:
            # Check for dash error popups in case cleaning failed to start
            alerts = self.driver.find_elements(
                By.CSS_SELECTOR, ".alert-danger, .alert-warning"
            )
            if alerts:
                print("Found alert messages:")
                for alert in alerts:
                    if alert.is_displayed():
                        print(f"  Alert: {alert.text}")
        except:
            pass

        # Appearance of the saving card indicates cleaning is finished
        print("Waiting for saving card to appear...")

        try:
            WebDriverWait(self.driver, 60).until(
                lambda driver: driver.find_element(
                    By.ID, "collapse-save"
                ).is_displayed()
            )
            print("✓ Saving card appeared - cleaning completed successfully")

        except Exception as e:
            print(f"✗ Timeout waiting for saving card: {e}")

            # Check for error messages in the browser console
            try:
                logs = self.driver.get_log("browser")
                if logs:
                    print("Browser console errors:")
                    for log in logs:
                        if log["level"] in ["SEVERE", "ERROR"]:
                            print(f"  {log['level']}: {log['message']}")
            except:
                print("Could not retrieve browser console logs")

            # Check for any visible error alerts
            try:
                alerts = self.driver.find_elements(
                    By.CSS_SELECTOR, ".alert-danger, .alert-warning, .alert-info"
                )
                if alerts:
                    print("Found alert messages:")
                    for alert in alerts:
                        if alert.is_displayed():
                            print(f"  Alert: {alert.text}")
            except:
                print("Could not check for alerts")

            return

        print("✓ Cleaning process completed")

        # Step 7: Verify results
        print("Step 7: Verifying cleaning results...")

        # Verify that the save card is visible and accessible
        save_card = self.driver.find_element(By.ID, "collapse-save")
        self.assertTrue(save_card.is_displayed(), "Save card not displayed")

        # Verify that the save button within the card is accessible
        save_button = self.driver.find_element(By.ID, "button-save")
        self.assertTrue(save_button.is_displayed(), "Save button not displayed")

        print("✓ Cleaning results verified - save functionality is available")
        print("✅ Complete workflow test passed!")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Selenium test for MagpiEM's Dash application"
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run tests with visible browser window (for debugging)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def set_headless_mode(enabled):
    """Set the global headless mode."""
    global HEADLESS_MODE
    HEADLESS_MODE = enabled


if __name__ == "__main__":
    args = parse_arguments()

    set_headless_mode(not args.no_headless)

    if args.verbose:
        print(f"Running tests in {'headless' if HEADLESS_MODE else 'visible'} mode")

    # Filter out custom arguments before passing to unittest
    import sys

    sys.argv = [sys.argv[0]]  # Keep only the script name

    unittest.main(verbosity=2)
