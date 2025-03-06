"""
Command Line Interface
=====================
CLI entry point for the traffic_rl package.
"""

import sys
from traffic_rl.main import main as main_func

def main():
    """Entry point for the traffic_rl command."""
    # Remove the script name from the arguments
    sys.argv.pop(0)
    
    # If no arguments are provided, show help
    if len(sys.argv) == 0:
        sys.argv.append("--help")
    
    # Run the main function
    main_func()

if __name__ == "__main__":
    main()
