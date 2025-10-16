#!/usr/bin/env python3
"""
Start dash app for e2e testing.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from magpiem.dash.dash_ui import main as run_dash_app  # noqa: E402


def main():
    """Start the dash app for testing."""
    parser = argparse.ArgumentParser(description="Start dash app for testing")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run on (default: 8050)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting application for testing on {args.host}:{args.port}")

    try:
        # Start the dash app
        run_dash_app(open_browser=False, log_level=log_level)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
