# -*- coding: utf-8 -*-
"""
End-to-end tests using Playwright.

This test suite performs a complete workflow test:
1. Uploads test data
2. Sets cleaning parameters
3. Runs cleaning
4. Toggles settings
5. Downloads output

Requires running install_playwright.py (or just run "python -m playwright install")
Requires a running MagpiEM server on localhost:8050

Note that graph interactivity cannot be tested, as the whole graph renders as a
single canvas component, which is virtually impossible to automate interactions with
"""

import tempfile
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from ..test_utils import TestConfig, get_test_data_path, setup_test_logging


class E2ETest:
    """End-to-end test class."""

    def __init__(self):
        self.logger = setup_test_logging()
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix="e2e_test_"))

    def setup(self):
        """Set up the test environment."""
        self.logger.info("Setting up e2e test environment")
        self._setup_browser()

    def teardown(self):
        """Clean up the test environment."""
        self.logger.info("Cleaning up e2e test environment")

        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _setup_browser(self):
        """Set up Playwright browser."""
        self.logger.info("Setting up browser")

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(accept_downloads=True)
        self.page = self.context.new_page()

    def test_complete_workflow(self):
        """Test the complete workflow."""
        self.logger.info("Starting complete workflow test")

        try:
            self._navigate_to_app()
            self._upload_test_data()
            self._set_cleaning_parameters()
            self._run_cleaning()
            self._toggle_keep_particles()
            self._download_output()

            self.logger.info("Complete workflow test passed")

        except Exception as e:
            self.logger.error(f"Workflow test failed: {e}")
            raise

    def _navigate_to_app(self):
        """Navigate to the application."""
        self.logger.info("Navigating to application")

        self.page.goto("http://localhost:8050")
        self.page.wait_for_selector("h1:has-text('MagpiEM')", timeout=10000)

        title = self.page.title()
        assert "MagpiEM" in title, f"Expected 'MagpiEM' in title, got: {title}"

        self.logger.info("Successfully navigated to application")

    def _upload_test_data(self):
        """Upload the standard test data file."""
        self.logger.info("Uploading test data")

        test_data_path = get_test_data_path(TestConfig.TEST_DATA_STANDARD)

        self.page.wait_for_selector("#upload-data")

        with self.page.expect_file_chooser() as fc_info:
            self.page.click("#upload-data")
        file_chooser = fc_info.value
        file_chooser.set_files(str(test_data_path))

        self.page.wait_for_timeout(2000)

        # Click the dropdown to open it
        self.page.click("#dropdown-filetype")
        # Wait for options to appear and click .mat option
        self.page.wait_for_selector("text=.mat")
        self.page.click("text=.mat")

        self.page.click("#slider-num-images")
        self.page.keyboard.press("ArrowRight")
        self.page.keyboard.press("ArrowRight")

        self.page.click("#button-read")

        self.page.wait_for_timeout(5000)

        self.logger.info("Test data uploaded successfully")

    def _set_cleaning_parameters(self):
        """Set the standard cleaning parameters."""
        self.logger.info("Setting cleaning parameters")

        params = TestConfig.TEST_DASH_PARAMETERS

        self.page.fill("#inp-dist-goal", str(params["inp-dist-goal"]))
        self.page.fill("#inp-dist-tol", str(params["inp-dist-tol"]))

        self.page.fill("#inp-ori-goal", str(params["inp-ori-goal"]))
        self.page.fill("#inp-ori-tol", str(params["inp-ori-tol"]))

        self.page.fill("#inp-curv-goal", str(params["inp-curv-goal"]))
        self.page.fill("#inp-curv-tol", str(params["inp-curv-tol"]))

        self.page.fill("#inp-min-neighbours", str(params["inp-min-neighbours"]))
        self.page.fill("#inp-cc-thresh", str(params["inp-cc-thresh"]))
        self.page.fill("#inp-array-size", str(params["inp-array-size"]))

        if params["switch-allow-flips"]:
            self.page.click("#switch-allow-flips")

        self.logger.info("Cleaning parameters set successfully")

    def _run_cleaning(self):
        """Run the cleaning process."""
        self.logger.info("Running cleaning process")

        self.page.click("#button-full-clean")

        try:
            self.page.wait_for_function(
                "() => document.querySelector('#progress-processing').value >= 100",
                timeout=60000,
            )
        except Exception:
            self.page.wait_for_timeout(10000)

        self.logger.info("Cleaning process completed")

    def _toggle_keep_particles(self):
        """Toggle the 'Keep selected particles' switch."""
        self.logger.info("Toggling 'Keep selected particles' switch")

        self.page.wait_for_selector("#switch-keep-particles")
        # clicking is the only way the switch can be easily accessed - mysterious DAQ component
        self.page.click("#switch-keep-particles")

        self.logger.info("'Keep selected particles' switch toggled successfully")

    def _download_output(self):
        """Download the output .mat file."""
        self.logger.info("Downloading output file")

        self.page.fill("#input-save-filename", "e2e_test_output")

        with self.page.expect_download() as download_info:
            self.page.click("#button-save")

        download = download_info.value

        download_path = self.temp_dir / "e2e_test_output.mat"
        download.save_as(download_path)

        assert download_path.exists(), "Downloaded file does not exist"
        assert download_path.stat().st_size > 0, "Downloaded file is empty"

        self.logger.info(f"Output file downloaded successfully to: {download_path}")


def test_complete_workflow():
    """Test the complete workflow end-to-end."""
    test_instance = E2ETest()

    try:
        test_instance.setup()
        test_instance.test_complete_workflow()
    finally:
        test_instance.teardown()


if __name__ == "__main__":
    """Run the e2e test directly."""
    test_instance = E2ETest()
    try:
        test_instance.setup()
        test_instance.test_complete_workflow()
        print("E2E test completed successfully!")
    except Exception as e:
        print(f"E2E test failed: {e}")
        raise
    finally:
        test_instance.teardown()
