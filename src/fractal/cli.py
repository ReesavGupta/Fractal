#!/usr/bin/env python3
"""
Fractal CLI - Sync wrapper for the async main function
"""
import asyncio
from .agent.main import main as async_main

def main_sync():
    """Sync wrapper for the async main function"""
    asyncio.run(async_main())

# This is the entry point for the console script
main = main_sync

if __name__ == "__main__":
    main_sync()

