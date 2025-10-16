#!/usr/bin/env python3
"""
Install Playwright browsers for e2e tests.
"""

import subprocess
import sys


def main():
    """Install Playwright browsers."""
    print("Installing Playwright browsers...")

    try:
        subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)

        print("Playwright browsers installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to install Playwright browsers: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
