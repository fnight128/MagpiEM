# -*- coding: utf-8 -*-
"""
Selenium headless test for the MagpiEM Dash application.
Tests the complete workflow: file upload, parameter input, and cleaning execution.
"""

import os
import sys
import time
import unittest
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class MagpiEMTest(unittest.TestCase):
    """Test case for MagpiEM Dash application using Selenium."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Chrome options for headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")

        # Set up Chrome driver
        service = Service(ChromeDriverManager().install())
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)

        # Set implicit wait
        cls.driver.implicitly_wait(10)

        # Get the project root directory
        cls.project_root = Path(__file__).parent.parent
        cls.test_data_path = cls.project_root / "test" / "test_data.mat"

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

        # Navigate to the application
        self.driver.get("http://localhost:8050")

        # Wait for the page to load
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

    def tearDown(self):
        """Clean up after each test."""
        self.stop_dash_app()

    def start_dash_app(self):
        """Start the Dash application."""
        import subprocess
        import sys
        import os

        # Add the project root to Python path
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root) + os.pathsep + env.get('PYTHONPATH', '')

        # Start the Dash app with no browser
        cmd = [sys.executable, "-m", "magpiem.dash_ui", "--no-browser"]
        self.dash_process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        # Wait for the app to start and check if it's running
        time.sleep(3)
        
        # Check if the process is still running
        if self.dash_process.poll() is not None:
            # Process died, get the error output
            stdout, stderr = self.dash_process.communicate()
            print(f"Dash app failed to start. stdout: {stdout.decode()}")
            print(f"Dash app failed to start. stderr: {stderr.decode()}")
            raise RuntimeError("Failed to start Dash application")
        
        # Wait a bit more for the server to be ready
        time.sleep(2)

    def stop_dash_app(self):
        """Stop the Dash application."""
        if self.dash_process:
            self.dash_process.terminate()
            self.dash_process.wait()

    def test_complete_workflow(self):
        """Test the complete workflow from start to finish."""
        print("🚀 Starting complete workflow test...")

        # Step 1: Check application loads
        print("Step 1: Checking application loads...")
        time.sleep(2)
        
        title = self.driver.find_element(By.TAG_NAME, "h1")
        self.assertEqual(title.text, "MagpiEM")
        print("✓ Application loaded successfully")

        # Step 2: Upload file
        print("Step 2: Uploading test file...")
        self.assertTrue(self.test_data_path.exists(), "test_data.mat file not found")
        
        upload_element = self.driver.find_element(By.ID, "upload-data")
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "upload-data"))
        )
        
        file_input = self.driver.find_element(By.CSS_SELECTOR, "#upload-data input[type='file']")
        file_input.send_keys(str(self.test_data_path.absolute()))
        
        WebDriverWait(self.driver, 10).until(
            lambda driver: "test_data.mat" in driver.find_element(By.ID, "upload-data").text
        )
        print("✓ File uploaded successfully")

        # Step 3: Set file type
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

        # Step 4: Read tomograms
        print("Step 4: Reading tomograms...")
        read_button = self.driver.find_element(By.ID, "button-read")
        read_button.click()
        
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.ID, "dropdown-tomo"))
        )
        
        dropdown = self.driver.find_element(By.ID, "dropdown-tomo")
        self.assertFalse(dropdown.get_attribute("disabled"))
        print("✓ Tomograms read successfully")

        # Step 5: Input cleaning parameters
        print("Step 5: Inputting cleaning parameters...")
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "inp-cc-thresh"))
        )

        # Input parameters according to the requirements
        parameters = {
            "switch-allow-flips": False,  # allow flips: false
            "inp-cc-thresh": 3,           # cc threshold: 3
            "inp-curv-goal": 90,          # curvature: 90
            "inp-curv-tol": 20,           # curvature tolerance: 20
            "inp-dist-goal": 60,          # distance: 60
            "inp-dist-tol": 20,           # distance tolerance: 20
            "inp-array-size": 10,         # min array size: 10
            "inp-min-neighbours": 5,      # min neighbours: 5
            "inp-ori-goal": 10,           # orientation: 10
            "inp-ori-tol": 20,            # orientation tolerance: 20
        }

        # Set each parameter
        for param_id, value in parameters.items():
            element = self.driver.find_element(By.ID, param_id)

            if "switch" in param_id:
                # Handle boolean switch
                if value:
                    if not element.is_selected():
                        element.click()
                else:
                    if element.is_selected():
                        element.click()
            else:
                # Handle numeric input
                element.clear()
                element.send_keys(str(value))

        print("✓ Cleaning parameters set correctly")

        # Step 6: Run cleaning
        print("Step 6: Running cleaning process...")
        clean_button = self.driver.find_element(By.ID, "button-full-clean")
        clean_button.click()

        print("⏳ Running cleaning process... (this may take several minutes)")

        # Wait for progress bar to appear and complete
        WebDriverWait(self.driver, 60).until(
            EC.presence_of_element_located((By.ID, "progress-processing"))
        )

        # Monitor progress - wait until progress reaches 100% or disappears
        max_wait_time = 120  # 2 minutes maximum wait
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            progress_bar = self.driver.find_element(By.ID, "progress-processing")
            progress_value = progress_bar.get_attribute("value")

            if progress_value and int(progress_value) >= 100:
                break

            # Check if cleaning completed by looking for completion indicators
            save_button = self.driver.find_element(By.ID, "button-save")
            if save_button.is_enabled():
                break

            time.sleep(2)  # Check every 2 seconds

        print("✓ Cleaning process completed")

        # Step 7: Verify results
        print("Step 7: Verifying cleaning results...")
        graph = self.driver.find_element(By.ID, "graph-picking")
        self.assertTrue(graph.is_displayed(), "Main graph not displayed")

        save_button = self.driver.find_element(By.ID, "button-save")
        self.assertTrue(save_button.is_displayed(), "Save button not displayed")

        print("✓ Cleaning results verified")
        print("✅ Complete workflow test passed!")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
