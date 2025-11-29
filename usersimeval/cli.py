#!/usr/bin/env python3
"""
Command-line interface for usersimeval.
"""

import argparse
import sys
import asyncio
from pathlib import Path

def viz_command(args):
    """Handle the viz subcommand."""
    # Import visualization_server from the same package
    from . import visualization_server

    # Use base_dir directly
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path.cwd()

    # Set up sys.argv to mimic the original visualization_server.py call
    original_argv = sys.argv.copy()
    sys.argv = ['visualization_server.py', '--base-dir', str(base_dir)]

    if args.port:
        sys.argv.extend(['--port', str(args.port)])

    try:
        # Call the main function from visualization_server
        visualization_server.main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def run_command(args):
    """Handle the run subcommand."""
    from . import model_grader

    # Set up sys.argv to mimic the original model_grader.py call
    original_argv = sys.argv.copy()
    sys.argv = ['model_grader.py']

    # Add required arguments
    sys.argv.extend(['--input_file', args.input_file])
    sys.argv.extend(['--output_dir', args.output_dir])

    if args.dimensions:
        sys.argv.extend(['--dimensions'] + args.dimensions)

    if args.num_tries_per_conversation:
        sys.argv.extend(['--num_tries_per_conversation', str(args.num_tries_per_conversation)])

    try:
        # Run the async main function from model_grader
        asyncio.run(model_grader.main())
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='UserSimEval tools')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # viz subcommand
    viz_parser = subparsers.add_parser('viz', help='Start visualization server')
    viz_parser.add_argument('--base-dir', '-d',
                           help='Base directory containing simulation results')
    viz_parser.add_argument('--port', '-p',
                           type=int,
                           default=8000,
                           help='Port to serve on (default: 8000)')

    # run subcommand
    run_parser = subparsers.add_parser('run', help='Run model grader on conversations')
    run_parser.add_argument('--input_file', required=True,
                           help='Input JSON file containing conversations')
    run_parser.add_argument('--output_dir', required=True,
                           help='Output directory for results')
    run_parser.add_argument('--dimensions', nargs='+',
                           help='Dimensions to evaluate')
    run_parser.add_argument('--num_tries_per_conversation', type=int,
                           help='Number of tries per conversation')

    args = parser.parse_args()

    if args.command == 'viz':
        viz_command(args)
    elif args.command == 'run':
        run_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()