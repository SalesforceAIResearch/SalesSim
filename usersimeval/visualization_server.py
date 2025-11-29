#!/usr/bin/env python3
"""
HTTP server for rollout viewer that can be pointed to any base directory.
Usage: python rollout_server.py [--base-dir <directory>] [--port <port>]
"""

import argparse
import http.server
import socketserver
import os
import sys
import shutil
import logging
import signal
import atexit
from pathlib import Path


class RolloutHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, base_directory=None, **kwargs):
        self.base_directory = base_directory or os.getcwd()
        super().__init__(*args, directory=self.base_directory, **kwargs)

    def do_GET(self):
        # Handle favicon.ico requests to avoid 404 errors
        if self.path == '/favicon.ico':
            self.send_response(204)  # No Content
            self.end_headers()
            return
        # Default to parent behavior
        super().do_GET()

    def end_headers(self):
        # Add CORS headers to allow local file access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()


def create_handler(base_directory):
    """Create a handler class with the specified base directory."""
    class CustomHandler(RolloutHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, base_directory=base_directory, **kwargs)
    return CustomHandler


def copy_rollout_html(base_dir):
    """Copy rollout_viewer.html to the base directory."""
    script_dir = Path(__file__).parent
    source_html = script_dir / 'rollout_viewer.html'
    target_html = base_dir / 'rollout_viewer.html'

    if source_html.exists():
        shutil.copy2(source_html, target_html)
        logging.info(f"Copied rollout_viewer.html to {target_html}")
    else:
        logging.warning(f"rollout_viewer.html not found in {script_dir}")


def cleanup_rollout_html(base_dir):
    """Remove rollout_viewer.html from the base directory."""
    target_html = base_dir / 'rollout_viewer.html'

    if target_html.exists():
        try:
            target_html.unlink()
            logging.info(f"Removed rollout_viewer.html from {base_dir}")
        except OSError as e:
            logging.warning(f"Failed to remove rollout_viewer.html: {e}")


def setup_cleanup_handlers(base_dir):
    """Set up cleanup handlers for when the server exits."""
    def cleanup():
        cleanup_rollout_html(base_dir)

    # Register cleanup function to run on normal exit
    atexit.register(cleanup)

    # Handle SIGINT (Ctrl+C) and SIGTERM
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    parser = argparse.ArgumentParser(description='HTTP server for rollout viewer')
    parser.add_argument('--base-dir', '-d',
                       default=os.getcwd(),
                       help='Base directory to serve files from (default: current directory)')
    parser.add_argument('--port', '-p',
                       type=int,
                       default=8000,
                       help='Port to serve on (default: 8000)')

    args = parser.parse_args()

    # Resolve the base directory path
    base_dir = Path(args.base_dir).resolve()

    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' does not exist")
        sys.exit(1)

    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a directory")
        sys.exit(1)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Copy rollout_viewer.html to base directory
    copy_rollout_html(base_dir)

    # Set up cleanup handlers
    setup_cleanup_handlers(base_dir)

    # Check for required files
    required_files = ['rollout_viewer.html', 'breakdown_scores.json']
    missing_files = []

    for file in required_files:
        if not (base_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        logging.warning(f"Missing required files in {base_dir}:")
        for file in missing_files:
            logging.warning(f"  - {file}")
        logging.warning("The rollout viewer may not work correctly.")

    # Check for human_readable_conversations directory
    conversations_dir = base_dir / 'human_readable_conversations'
    if not conversations_dir.exists():
        logging.warning(f"'human_readable_conversations' directory not found in {base_dir}")
        logging.warning("Conversation viewing will not work correctly.")

    # Create the handler with the specified base directory
    handler_class = create_handler(str(base_dir))

    # Start the server
    with socketserver.TCPServer(("", args.port), handler_class) as httpd:
        logging.info(f"Serving rollout viewer from: {base_dir}")
        logging.info(f"Server running at: http://localhost:{args.port}")
        logging.info(f"Open http://localhost:{args.port}/rollout_viewer.html in your browser")
        logging.info("Press Ctrl+C to stop the server")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logging.info("Server stopped.")


if __name__ == "__main__":
    main()