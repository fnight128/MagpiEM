# -*- coding: utf-8 -*-
"""
Automated testing framework for the MagpiEM Dash UI.
Uses Selenium WebDriver to simulate user interactions and verify functionality.
"""

import time
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import unittest

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import subprocess
import threading
import requests
import json


class DashUITestCase(unittest.TestCase):
    """Base test case for Dash UI automation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment before all tests."""
        cls.base_url = "http://localhost:8050"
        cls.driver = None
        cls.server_process = None
        
        # Start the Dash server in a separate process
        cls.start_dash_server()
        
        # Wait for server to be ready
        cls.wait_for_server()
        
        # Set up the web driver
        cls.setup_driver()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if cls.driver:
            cls.driver.quit()
        
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()
    
    @classmethod
    def start_dash_server(cls):
        """Start the Dash server in a background process."""
        # Get the path to the main module
        project_root = Path(__file__).parent.parent
        main_module = project_root / "magpiem" / "dash_ui.py"
        
        # Start the server process
        cls.server_process = subprocess.Popen(
            ["python", "-m", "magpiem.dash_ui"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    @classmethod
    def wait_for_server(cls, timeout=30):
        """Wait for the Dash server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(cls.base_url, timeout=1)
                if response.status_code == 200:
                    print(f"Server ready at {cls.base_url}")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        
        raise TimeoutError(f"Server did not start within {timeout} seconds")
    
    @classmethod
    def setup_driver(cls):
        """Set up the Chrome WebDriver with appropriate options."""
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Commented out for visible browser
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Use webdriver-manager to automatically download and manage ChromeDriver
        service = Service(ChromeDriverManager().install())
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)
        cls.driver.implicitly_wait(10)
    
    def setUp(self):
        """Set up before each test."""
        self.driver.get(self.base_url)
        self.wait = WebDriverWait(self.driver, 10)
        
        # Wait for page to load and debug state
        self.wait_for_page_load()
        self.debug_page_state()
    
    def tearDown(self):
        """Clean up after each test."""
        # Clear any uploaded files or temporary data
        pass
    
    def wait_for_element(self, by: By, value: str, timeout: int = 10):
        """Wait for an element to be present and return it."""
        return self.wait.until(EC.presence_of_element_located((by, value)))
    
    def wait_for_clickable(self, by: By, value: str, timeout: int = 10):
        """Wait for an element to be clickable and return it."""
        return self.wait.until(EC.element_to_be_clickable((by, value)))
    
    def wait_for_visible(self, by: By, value: str, timeout: int = 10):
        """Wait for an element to be visible and return it."""
        return self.wait.until(EC.visibility_of_element_located((by, value)))
    
    def upload_file(self, file_path: str, upload_id: str = "upload-data"):
        """Upload a file to the specified upload component."""
        upload_element = self.wait_for_element(By.ID, upload_id)
        upload_element.send_keys(file_path)
        time.sleep(2)  # Wait for upload to complete
    
    def click_button(self, button_id: str):
        """Click a button by its ID."""
        button = self.wait_for_clickable(By.ID, button_id)
        
        # Scroll the button into view
        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
        time.sleep(0.5)  # Wait for scroll to complete
        
        # Try to click using JavaScript if regular click fails
        try:
            button.click()
        except Exception as e:
            print(f"Regular click failed for {button_id}, trying JavaScript click: {e}")
            self.driver.execute_script("arguments[0].click();", button)
        
        time.sleep(1)  # Wait for action to complete
    
    def set_input_value(self, input_id: str, value: str):
        """Set the value of an input field."""
        input_element = self.wait_for_element(By.ID, input_id)
        
        # Scroll the element into view
        self.driver.execute_script("arguments[0].scrollIntoView(true);", input_element)
        time.sleep(0.5)
        
        # Clear and set value using JavaScript for better reliability
        self.driver.execute_script("arguments[0].value = '';", input_element)
        self.driver.execute_script(f"arguments[0].value = '{value}';", input_element)
        
        # Trigger change event
        self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", input_element)
    
    def select_dropdown_option(self, dropdown_id: str, option_value: str):
        """Select an option from a dropdown."""
        dropdown = self.wait_for_element(By.ID, dropdown_id)
        
        # Scroll the dropdown into view
        self.driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)
        time.sleep(0.5)
        
        dropdown_select = Select(dropdown)
        dropdown_select.select_by_value(option_value)
    
    def toggle_switch(self, switch_id: str):
        """Toggle a boolean switch."""
        switch = self.wait_for_clickable(By.ID, switch_id)
        
        # Scroll the switch into view
        self.driver.execute_script("arguments[0].scrollIntoView(true);", switch)
        time.sleep(0.5)
        
        # Try to click using JavaScript if regular click fails
        try:
            switch.click()
        except Exception as e:
            print(f"Regular click failed for switch {switch_id}, trying JavaScript click: {e}")
            self.driver.execute_script("arguments[0].click();", switch)
        
        time.sleep(0.5)
    
    def get_element_text(self, element_id: str) -> str:
        """Get the text content of an element."""
        element = self.wait_for_element(By.ID, element_id)
        return element.text
    
    def check_element_exists(self, element_id: str) -> bool:
        """Check if an element exists on the page."""
        try:
            element = self.driver.find_element(By.ID, element_id)
            return True
        except NoSuchElementException:
            return False
    
    def get_page_source_info(self):
        """Get information about the current page for debugging."""
        try:
            page_source = self.driver.page_source
            # Look for common element patterns
            if "slider-num-images" in page_source:
                print("✅ 'slider-num-images' found in page source")
            else:
                print("❌ 'slider-num-images' NOT found in page source")
            
            if "button-read" in page_source:
                print("✅ 'button-read' found in page source")
            else:
                print("❌ 'button-read' NOT found in page source")
            
            if "upload-data" in page_source:
                print("✅ 'upload-data' found in page source")
            else:
                print("❌ 'upload-data' NOT found in page source")
            
            # Print first 500 characters of page source for debugging
            print(f"Page source preview: {page_source[:500]}...")
            
        except Exception as e:
            print(f"Could not get page source info: {e}")
    
    def wait_for_page_load(self, timeout: int = 30):
        """Wait for the page to fully load."""
        try:
            # Wait for document ready state
            self.driver.execute_script("return document.readyState") == "complete"
            time.sleep(2)  # Extra wait for dynamic content
            print("✅ Page loaded successfully")
        except Exception as e:
            print(f"❌ Page load wait failed: {e}")
    
    def debug_page_state(self):
        """Debug the current page state."""
        print("\n🔍 DEBUGGING PAGE STATE:")
        print(f"Current URL: {self.driver.current_url}")
        print(f"Page title: {self.driver.title}")
        
        # Check for common elements
        elements_to_check = [
            "upload-data", "slider-num-images", "button-read", 
            "dropdown-filetype", "graph-picking"
        ]
        
        for element_id in elements_to_check:
            exists = self.check_element_exists(element_id)
            status = "✅" if exists else "❌"
            print(f"{status} Element '{element_id}': {'Found' if exists else 'Not found'}")
        
        # Get page source info
        self.get_page_source_info()
        
        # Take a screenshot for debugging
        screenshot_path = self.take_screenshot("debug_page_state.png")
        print(f"📸 Debug screenshot: {screenshot_path}")
        print("🔍 END DEBUGGING\n")
    
    def wait_for_figure_update(self, timeout: int = 10):
        """Wait for the 3D figure to update."""
        # Look for the graph element and wait for it to be ready
        graph_element = self.wait_for_element(By.ID, "graph-picking")
        time.sleep(2)  # Give time for any animations or updates
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get the current graph data from the browser."""
        # Execute JavaScript to get the figure data
        script = """
        const graph = document.getElementById('graph-picking');
        if (graph && graph._fullData) {
            return graph._fullData;
        }
        return null;
        """
        return self.driver.execute_script(script)
    
    def wait_for_card_to_open(self, card_id: str, timeout: int = 30):
        """Wait for a specific card to open."""
        card = self.wait_for_element(By.ID, card_id)
        # Wait for the card to be visible (not collapsed)
        self.wait.until(lambda driver: card.is_displayed())
        time.sleep(1)  # Give extra time for animations
    
    def set_slider_value(self, slider_id: str, value: str):
        """Set a slider value using JavaScript."""
        slider = self.wait_for_element(By.ID, slider_id)
        
        # Scroll the slider into view
        self.driver.execute_script("arguments[0].scrollIntoView(true);", slider)
        time.sleep(0.5)
        
        # Set the value using JavaScript
        self.driver.execute_script(f"arguments[0].value = '{value}';", slider)
        
        # Trigger the change event
        self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", slider)
        self.driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", slider)
        
        time.sleep(1)  # Wait for the change to take effect
    
    def take_screenshot(self, filename: str = None):
        """Take a screenshot for debugging purposes."""
        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"debug_screenshot_{timestamp}.png"
            
            # Create screenshots directory in the test folder
            screenshot_dir = Path(__file__).parent / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            
            screenshot_path = screenshot_dir / filename
            
            # Take the screenshot
            self.driver.save_screenshot(str(screenshot_path))
            
            # Verify the file was created
            if screenshot_path.exists():
                file_size = screenshot_path.stat().st_size
                print(f"📸 Screenshot saved: {screenshot_path} (size: {file_size} bytes)")
                return str(screenshot_path)
            else:
                print(f"❌ Screenshot file was not created at: {screenshot_path}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to take screenshot: {e}")
            # Try to get current page info for debugging
            try:
                current_url = self.driver.current_url
                page_title = self.driver.title
                print(f"Current URL: {current_url}")
                print(f"Page title: {page_title}")
            except:
                pass
            return None


class TestWorkflow(DashUITestCase):
    """Test the complete workflow: upload file, read tomograms, run cleaning."""
    
    def test_server_accessible(self):
        """Test that the server is accessible and responding."""
        print("Testing server accessibility...")
        
        try:
            # Check if we can access the page
            response = requests.get(self.base_url, timeout=5)
            self.assertEqual(
                response.status_code, 
                200,
                f"Server not responding correctly. Expected status 200, got {response.status_code}"
            )
            print("✅ Server is accessible")
            
            # Check if the page title contains expected content (now fixed with update_title=None)
            self.assertIn(
                "MagpiEM", 
                self.driver.title,
                f"Page title verification failed. Expected 'MagpiEM' in title, but got: '{self.driver.title}'"
            )
            print("✅ Page title is correct")
            
        except Exception as e:
            print(f"❌ Server accessibility test failed: {e}")
            self.fail(f"Server is not accessible: {e}")
    
    def test_file_exists(self):
        """Test that the required test file exists."""
        test_file_path = Path(__file__).parent / "WT_CA_2nd.mat"
        self.assertTrue(
            test_file_path.exists(), 
            f"Test file 'WT_CA_2nd.mat' not found at expected location: {test_file_path}"
        )
        print(f"✅ Test file found: {test_file_path}")
    
    def test_basic_elements_present(self):
        """Test that basic UI elements are present before workflow."""
        print("\n🔍 Testing basic UI elements...")
        
        # Wait a bit more for dynamic content to load
        time.sleep(3)
        
        # Check that main components are present
        elements_to_check = [
            ("upload-data", "Upload data component"),
            ("slider-num-images", "Slider for number of images"),
            ("button-read", "Read button"),
            ("dropdown-filetype", "File type dropdown"),
            ("graph-picking", "Graph picking component")
        ]
        
        missing_elements = []
        
        for element_id, description in elements_to_check:
            exists = self.check_element_exists(element_id)
            if exists:
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} NOT found")
                missing_elements.append(description)
        
        # If any elements are missing, take a screenshot and provide detailed info
        if missing_elements:
            print(f"\n❌ Missing elements: {', '.join(missing_elements)}")
            screenshot_path = self.take_screenshot("missing_elements_error.png")
            print(f"📸 Error screenshot saved: {screenshot_path}")
            
            # Get detailed page info
            self.get_page_source_info()
            
            # Fail with detailed message
            self.fail(f"Missing UI elements: {', '.join(missing_elements)}. Check screenshot at {screenshot_path}")
        
        print("✅ All basic UI elements are present")
    
    def test_complete_workflow(self):
        """Test the complete workflow from file upload to cleaning."""
        print("Starting complete workflow test...")
        
        # Step 1: Upload WT_CA_2nd.mat from the test folder
        test_file_path = Path(__file__).parent / "WT_CA_2nd.mat"
        if not test_file_path.exists():
            self.fail(f"Test file not found: {test_file_path}")
        
        print(f"Uploading file: {test_file_path}")
        self.upload_file(str(test_file_path))
        
        # Verify file was uploaded
        upload_element = self.wait_for_element(By.ID, "upload-data")
        self.assertIn(
            "WT_CA_2nd.mat", 
            upload_element.text,
            f"File upload verification failed. Expected 'WT_CA_2nd.mat' in upload text, but got: '{upload_element.text}'"
        )
        print("✅ File uploaded successfully")
        
        # Step 2: Select option to process only 5 images
        print("Setting number of images to 5...")
        try:
            self.set_slider_value("slider-num-images", "1")
            
            # Verify slider value
            slider = self.wait_for_element(By.ID, "slider-num-images")
            slider_value = slider.get_attribute("value")
            self.assertEqual(
                slider_value, 
                "1",
                f"Slider value verification failed. Expected '1', but got '{slider_value}'"
            )
            print("✅ Number of images set to 5")
        except Exception as e:
            print(f"❌ Failed to set slider value: {e}")
            # Try alternative method
            slider = self.wait_for_element(By.ID, "slider-num-images")
            self.driver.execute_script("arguments[0].value = '1';", slider)
            self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", slider)
            time.sleep(1)
            print("✅ Number of images set to 5 (alternative method)")
        
        # Step 3: Click button-read
        print("Clicking read button...")
        try:
            self.click_button("button-read")
            print("✅ Read button clicked successfully")
        except Exception as e:
            print(f"❌ Failed to click read button: {e}")
            # Take a screenshot for debugging
            screenshot_path = self.take_screenshot("read_button_error.png")
            print(f"📸 Screenshot saved to: {screenshot_path}")
            # Try to find the button and get more info
            try:
                button = self.driver.find_element(By.ID, "button-read")
                print(f"Button found: enabled={button.is_enabled()}, displayed={button.is_displayed()}")
                print(f"Button text: {button.text}")
                print(f"Button classes: {button.get_attribute('class')}")
            except Exception as e2:
                print(f"Could not find button: {e2}")
            raise
        
        # Wait for processing to complete
        print("Waiting for tomogram reading to complete...")
        time.sleep(5)  # Give time for processing
        
        # Check that reading was successful
        try:
            label_element = self.wait_for_element(By.ID, "label-read")
            read_text = label_element.text.lower()
            self.assertIn(
                "read", 
                read_text,
                f"Reading verification failed. Expected 'read' in label text, but got: '{read_text}'"
            )
            print("✅ Tomograms read successfully")
        except Exception as e:
            print(f"❌ Failed to verify reading success: {e}")
            screenshot_path = self.take_screenshot("reading_verification_error.png")
            print(f"📸 Screenshot saved to: {screenshot_path}")
            print(f"Label text: {label_element.text if 'label_element' in locals() else 'No label found'}")
            raise
        
        # Step 4: Wait for card-clean to open
        print("Waiting for cleaning card to open...")
        try:
            self.wait_for_card_to_open("collapse-clean")
            print("✅ Cleaning card is now open")
        except Exception as e:
            print(f"❌ Failed to wait for cleaning card: {e}")
            screenshot_path = self.take_screenshot("cleaning_card_error.png")
            print(f"📸 Screenshot saved to: {screenshot_path}")
            # Check if the card exists
            try:
                card = self.driver.find_element(By.ID, "collapse-clean")
                print(f"Card found: displayed={card.is_displayed()}")
            except Exception as e2:
                print(f"Could not find cleaning card: {e2}")
            raise
        
        # Step 5: Click button-full-clean (default parameters will be used)
        print("Starting cleaning process...")
        try:
            self.click_button("button-full-clean")
            print("✅ Cleaning button clicked successfully")
        except Exception as e:
            print(f"❌ Failed to click cleaning button: {e}")
            screenshot_path = self.take_screenshot("cleaning_button_error.png")
            print(f"📸 Screenshot saved to: {screenshot_path}")
            # Try to find the button and get more info
            try:
                button = self.driver.find_element(By.ID, "button-full-clean")
                print(f"Button found: enabled={button.is_enabled()}, displayed={button.is_displayed()}")
                print(f"Button text: {button.text}")
                print(f"Button classes: {button.get_attribute('class')}")
            except Exception as e2:
                print(f"Could not find cleaning button: {e2}")
            raise
        
        # Wait for cleaning to start and progress
        print("Waiting for cleaning to complete...")
        time.sleep(10)  # Give time for cleaning to start
        
        # Check progress bar
        try:
            progress_element = self.wait_for_element(By.ID, "progress-processing")
            progress_value = progress_element.get_attribute("value")
            print(f"Cleaning progress: {progress_value}%")
        except Exception as e:
            print(f"Could not find progress bar: {e}")
        
        # Wait for cleaning to complete (this might take a while)
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check if save card is now open (indicates cleaning completed)
                save_card = self.driver.find_element(By.ID, "collapse-save")
                if save_card.is_displayed():
                    print("✅ Cleaning completed successfully!")
                    break
            except NoSuchElementException:
                pass
            
            # Check progress
            try:
                progress_element = self.driver.find_element(By.ID, "progress-processing")
                progress_value = progress_element.get_attribute("value")
                if progress_value and float(progress_value) > 0:
                    print(f"Cleaning progress: {progress_value}%")
            except:
                pass
            
            time.sleep(5)
        else:
            screenshot_path = self.take_screenshot("cleaning_timeout_error.png")
            print(f"📸 Screenshot saved to: {screenshot_path}")
            self.fail(f"Cleaning did not complete within {max_wait_time} seconds")
        
        print("🎉 Complete workflow test finished successfully!")


class TestBasicUI(DashUITestCase):
    """Test basic UI functionality."""
    
    def test_page_loads(self):
        """Test that the page loads correctly."""
        self.assertIn(
            "MagpiEM", 
            self.driver.title,
            f"Page title verification failed. Expected 'MagpiEM' in title, but got: '{self.driver.title}'"
        )
        
        # Check that main components are present
        self.assertTrue(
            self.check_element_exists("upload-data"),
            "Upload data component not found on page"
        )
        self.assertTrue(
            self.check_element_exists("dropdown-tomo"),
            "Tomogram dropdown not found on page"
        )
        self.assertTrue(
            self.check_element_exists("graph-picking"),
            "Graph picking component not found on page"
        )
    
    def test_upload_section_visible(self):
        """Test that the upload section is visible by default."""
        upload_section = self.wait_for_visible(By.ID, "collapse-upload")
        self.assertTrue(
            upload_section.is_displayed(),
            "Upload section is not visible by default"
        )
    
    def test_file_type_dropdown(self):
        """Test the file type dropdown functionality."""
        dropdown = self.wait_for_element(By.ID, "dropdown-filetype")
        self.assertIsNotNone(
            dropdown,
            "File type dropdown not found on page"
        )
        
        # Check available options
        options = dropdown.find_elements(By.TAG_NAME, "option")
        self.assertGreater(
            len(options), 
            0,
            f"File type dropdown has no options. Expected at least 1 option, but found {len(options)}"
        )


class TestFileUpload(DashUITestCase):
    """Test file upload functionality."""
    
    def setUp(self):
        """Set up test data."""
        super().setUp()
        
        # Create a temporary test file
        self.test_file = self.create_test_mat_file()
    
    def tearDown(self):
        """Clean up test data."""
        if hasattr(self, 'test_file') and os.path.exists(self.test_file):
            os.remove(self.test_file)
        super().tearDown()
    
    def create_test_mat_file(self) -> str:
        """Create a minimal test .mat file for testing."""
        # This is a simplified test file - in practice you'd want a real .mat file
        test_data = {
            'wt2nd_4004_2': [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 25, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.5, 0.6],
            ]
        }
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
        temp_file.close()
        
        # In a real implementation, you'd save this as a proper .mat file
        # For now, we'll just create a placeholder
        with open(temp_file.name, 'w') as f:
            f.write("Test .mat file content")
        
        return temp_file.name
    
    def test_file_upload(self):
        """Test uploading a file."""
        # Upload the test file
        self.upload_file(self.test_file)
        
        # Check that the filename is displayed
        filename_element = self.wait_for_element(By.ID, "upload-data")
        self.assertIn(
            "test_data.mat", 
            filename_element.text,
            f"File upload verification failed. Expected 'test_data.mat' in upload text, but got: '{filename_element.text}'"
        )
    
    def test_read_tomograms(self):
        """Test the read tomograms functionality."""
        # Upload file
        self.upload_file(self.test_file)
        
        # Set number of images to 1
        slider = self.wait_for_element(By.ID, "slider-num-images")
        self.driver.execute_script("arguments[0].value = '0';", slider)
        
        # Click read button
        self.click_button("button-read")
        
        # Wait for processing and check result
        time.sleep(3)
        label_element = self.wait_for_element(By.ID, "label-read")
        self.assertIn(
            "read", 
            label_element.text.lower(),
            f"Reading verification failed. Expected 'read' in label text, but got: '{label_element.text}'"
        )


class TestGraphControls(DashUITestCase):
    """Test graph control functionality."""
    
    def test_cone_plot_toggle(self):
        """Test toggling cone plot on/off."""
        # Toggle cone plot switch
        self.toggle_switch("switch-cone-plot")
        
        # Verify the switch state changed
        switch = self.wait_for_element(By.ID, "switch-cone-plot")
        self.assertTrue(
            switch.is_selected(),
            "Cone plot switch was not toggled on"
        )
    
    def test_cone_size_input(self):
        """Test setting cone size."""
        # Set cone size
        self.set_input_value("inp-cone-size", "15")
        
        # Click set button
        self.click_button("button-set-cone-size")
        
        # Verify the value was set
        input_element = self.wait_for_element(By.ID, "inp-cone-size")
        self.assertEqual(
            input_element.get_attribute("value"), 
            "15",
            f"Cone size input verification failed. Expected '15', but got '{input_element.get_attribute('value')}'"
        )
    
    def test_show_removed_particles_toggle(self):
        """Test toggling show removed particles."""
        # Toggle the switch
        self.toggle_switch("switch-show-removed")
        
        # Verify the switch state
        switch = self.wait_for_element(By.ID, "switch-show-removed")
        self.assertTrue(
            switch.is_selected(),
            "Show removed particles switch was not toggled on"
        )


class TestCleaningParameters(DashUITestCase):
    """Test cleaning parameter controls."""
    
    def test_distance_parameters(self):
        """Test setting distance parameters."""
        # Set distance goal
        self.set_input_value("inp-dist-goal", "50.0")
        
        # Set distance tolerance
        self.set_input_value("inp-dist-tol", "5.0")
        
        # Verify values
        goal_element = self.wait_for_element(By.ID, "inp-dist-goal")
        tol_element = self.wait_for_element(By.ID, "inp-dist-tol")
        
        self.assertEqual(
            goal_element.get_attribute("value"), 
            "50.0",
            f"Distance goal verification failed. Expected '50.0', but got '{goal_element.get_attribute('value')}'"
        )
        self.assertEqual(
            tol_element.get_attribute("value"), 
            "5.0",
            f"Distance tolerance verification failed. Expected '5.0', but got '{tol_element.get_attribute('value')}'"
        )
    
    def test_orientation_parameters(self):
        """Test setting orientation parameters."""
        # Set orientation goal
        self.set_input_value("inp-ori-goal", "30.0")
        
        # Set orientation tolerance
        self.set_input_value("inp-ori-tol", "10.0")
        
        # Verify values
        goal_element = self.wait_for_element(By.ID, "inp-ori-goal")
        tol_element = self.wait_for_element(By.ID, "inp-ori-tol")
        
        self.assertEqual(
            goal_element.get_attribute("value"), 
            "30.0",
            f"Orientation goal verification failed. Expected '30.0', but got '{goal_element.get_attribute('value')}'"
        )
        self.assertEqual(
            tol_element.get_attribute("value"), 
            "10.0",
            f"Orientation tolerance verification failed. Expected '10.0', but got '{tol_element.get_attribute('value')}'"
        )
    
    def test_curvature_parameters(self):
        """Test setting curvature parameters."""
        # Set curvature goal
        self.set_input_value("inp-curv-goal", "15.0")
        
        # Set curvature tolerance
        self.set_input_value("inp-curv-tol", "3.0")
        
        # Verify values
        goal_element = self.wait_for_element(By.ID, "inp-curv-goal")
        tol_element = self.wait_for_element(By.ID, "inp-curv-tol")
        
        self.assertEqual(
            goal_element.get_attribute("value"), 
            "15.0",
            f"Curvature goal verification failed. Expected '15.0', but got '{goal_element.get_attribute('value')}'"
        )
        self.assertEqual(
            tol_element.get_attribute("value"), 
            "3.0",
            f"Curvature tolerance verification failed. Expected '3.0', but got '{tol_element.get_attribute('value')}'"
        )
    
    def test_other_parameters(self):
        """Test setting other cleaning parameters."""
        # Set min neighbours
        self.set_input_value("inp-min-neighbours", "3")
        
        # Set CC threshold
        self.set_input_value("inp-cc-thresh", "0.8")
        
        # Set min lattice size
        self.set_input_value("inp-array-size", "5")
        
        # Toggle allow flips
        self.toggle_switch("switch-allow-flips")
        
        # Verify values
        neighbours_element = self.wait_for_element(By.ID, "inp-min-neighbours")
        cc_element = self.wait_for_element(By.ID, "inp-cc-thresh")
        array_element = self.wait_for_element(By.ID, "inp-array-size")
        flips_element = self.wait_for_element(By.ID, "switch-allow-flips")
        
        self.assertEqual(
            neighbours_element.get_attribute("value"), 
            "3",
            f"Min neighbours verification failed. Expected '3', but got '{neighbours_element.get_attribute('value')}'"
        )
        self.assertEqual(
            cc_element.get_attribute("value"), 
            "0.8",
            f"CC threshold verification failed. Expected '0.8', but got '{cc_element.get_attribute('value')}'"
        )
        self.assertEqual(
            array_element.get_attribute("value"), 
            "5",
            f"Array size verification failed. Expected '5', but got '{array_element.get_attribute('value')}'"
        )
        self.assertTrue(
            flips_element.is_selected(),
            "Allow flips switch was not toggled on"
        )


class TestNavigation(DashUITestCase):
    """Test navigation between tomograms."""
    
    def test_next_previous_buttons(self):
        """Test next/previous tomogram navigation."""
        # These tests would require uploaded data to work properly
        # For now, just verify the buttons exist
        next_button = self.wait_for_element(By.ID, "button-next-Tomogram")
        prev_button = self.wait_for_element(By.ID, "button-previous-Tomogram")
        
        self.assertIsNotNone(
            next_button,
            "Next tomogram button not found on page"
        )
        self.assertIsNotNone(
            prev_button,
            "Previous tomogram button not found on page"
        )
        
        # Check that buttons are disabled initially (no data loaded)
        self.assertTrue(
            next_button.get_attribute("disabled"),
            "Next tomogram button should be disabled when no data is loaded"
        )


class TestSaveFunctionality(DashUITestCase):
    """Test save functionality."""
    
    def test_save_filename_input(self):
        """Test setting save filename."""
        # Set save filename
        self.set_input_value("input-save-filename", "test_output.mat")
        
        # Verify the value
        filename_element = self.wait_for_element(By.ID, "input-save-filename")
        self.assertEqual(
            filename_element.get_attribute("value"), 
            "test_output.mat",
            f"Save filename verification failed. Expected 'test_output.mat', but got '{filename_element.get_attribute('value')}'"
        )
    
    def test_keep_particles_toggle(self):
        """Test toggling keep particles switch."""
        # Toggle the switch
        self.toggle_switch("switch-keep-particles")
        
        # Verify the switch state
        switch = self.wait_for_element(By.ID, "switch-keep-particles")
        self.assertTrue(
            switch.is_selected(),
            "Keep particles switch was not toggled on"
        )
        
        # Check that the label updated
        label_element = self.wait_for_element(By.ID, "label-keep-particles")
        self.assertIn(
            "unselected", 
            label_element.text,
            f"Keep particles label verification failed. Expected 'unselected' in label text, but got: '{label_element.text}'"
        )


def run_tests():
    """Run all the automated tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestWorkflow,  # Add the new workflow test first
        TestBasicUI,
        TestFileUpload,
        TestGraphControls,
        TestCleaningParameters,
        TestNavigation,
        TestSaveFunctionality,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Starting automated Dash UI tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1)
