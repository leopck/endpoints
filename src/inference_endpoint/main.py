#!/usr/bin/env python3
"""
Main entry point for the MLPerf Inference Endpoint Benchmarking System.

This module provides the main application logic and can be run directly
or imported as a module.
"""

import asyncio
import logging
import sys

from inference_endpoint.cli import main as cli_main
from inference_endpoint.utils.logging import setup_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting MLPerf Inference Endpoint Benchmarking System")

        # Run CLI
        await cli_main()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


def run() -> None:
    """Entry point for setuptools."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
