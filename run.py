#!/usr/bin/env python
"""
Run script for Traffic Light Control with Reinforcement Learning.
This is a simplified wrapper around the main module.
"""

import os
import sys
import argparse
import logging
import subprocess
import importlib.util

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("RunScript")

def check_dependencies():
    """Check if required dependencies are installed."""
    required = ["numpy", "gymnasium", "torch", "pygame"]
    missing = []
    
    for package in required:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.info("Please install them with:")
        logger.info(f"pip install {' '.join(missing)}")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Traffic Light Control with RL')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'test', 'visualize', 'benchmark', 'record', 'analyze'],
                      help='Mode to run (train, test, visualize, benchmark, record, analyze)')
    parser.add_argument('--install', action='store_true',
                      help='Install dependencies before running')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                      help='Path to model file for test/visualize mode')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Number of episodes (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--record-video', type=str, default=None,
                      help='Record a video of the environment (in record mode)')
    parser.add_argument('--video-duration', type=int, default=30,
                      help='Duration of recorded video in seconds')
    parser.add_argument('--traffic-pattern', type=str, default='uniform',
                      choices=['uniform', 'rush_hour', 'weekend', 'random'],
                      help='Traffic pattern to use for testing and visualization')
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install:
        if not install_dependencies():
            sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Use --install to install them.")
        sys.exit(1)
    
    # Import the main module
    try:
        from main import main as run_main
        
        # Pass arguments to main function
        sys.argv = [sys.argv[0]]  # Reset argv
        if args.mode:
            sys.argv.extend(['--mode', args.mode])
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.model:
            sys.argv.extend(['--model', args.model])
        if args.episodes:
            sys.argv.extend(['--episodes', str(args.episodes)])
        if args.visualize:
            sys.argv.append('--visualize')
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.seed:
            sys.argv.extend(['--seed', str(args.seed)])
        if args.debug:
            sys.argv.append('--debug')
        if args.record_video:
            sys.argv.extend(['--record-video', args.record_video])
        if args.video_duration:
            sys.argv.extend(['--video-duration', str(args.video_duration)])
        if args.traffic_pattern:
            sys.argv.extend(['--traffic-pattern', args.traffic_pattern])
        
        # Run the main function
        run_main()
    except ImportError:
        logger.error("Could not import main module. Make sure it exists.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()