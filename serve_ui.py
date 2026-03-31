#!/usr/bin/env python3
"""
HTTP server to serve the test UI for ingesting and verifying tree images.
Discovers images from ./ingest-data and partitions by tree ID.
Acts as a reverse proxy for API calls to the FastAPI backend on port 8001.

Usage: python serve_ui.py
Then open: http://localhost:8080
Works via ngrok: ngrok http 8080
"""

import http.server
import socketserver
import webbrowser
import os
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

PORT = 8080
BACKEND_URL = "http://localhost:8001"
DIRECTORY = Path(__file__).parent
INGEST_DATA_DIR = DIRECTORY / "ingest-data"

# API paths to proxy to the FastAPI backend
PROXY_PREFIXES = ('/ingest', '/verify', '/health', '/debug', '/cleanup_gpu', '/graph')

# Milvus repository instance (lazy init)
_milvus_repo = None


def get_milvus_repo():
    """Get or create Milvus repository instance."""
    global _milvus_repo
    if _milvus_repo is None:
        try:
            from src.repository.milvusRepository import MilvusRepository

            # Create MilvusRepository with default config (uses AppConfig)
            _milvus_repo = MilvusRepository()

            # Test connection
            if not _milvus_repo._connected:
                print(f"[serve_ui.py] Warning: Milvus connection not established")
                _milvus_repo = None
                return None

            print(f"[serve_ui.py] Connected to Milvus successfully")
        except Exception as e:
            print(f"[serve_ui.py] Warning: Could not connect to Milvus: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _milvus_repo


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def _is_proxy_path(self, path: str) -> bool:
        """Check if this path should be proxied to the backend."""
        return any(path.startswith(prefix) for prefix in PROXY_PREFIXES)

    def _proxy_request(self, method: str):
        """Forward the request to the FastAPI backend and relay the response."""
        target_url = f"{BACKEND_URL}{self.path}"
        try:
            # Read request body if present
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            # Build the proxy request
            req = Request(target_url, data=body, method=method)

            # Forward relevant headers
            for header in ('Content-Type', 'Accept', 'Authorization'):
                value = self.headers.get(header)
                if value:
                    req.add_header(header, value)

            # Execute the request to backend
            with urlopen(req, timeout=300) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                # Forward response headers
                for key, val in resp.getheaders():
                    if key.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(resp_body)

        except HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(resp_body)
        except URLError as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_msg = json.dumps({
                "error": f"Backend unavailable: {e.reason}",
                "hint": "Make sure FastAPI is running: python main.py"
            })
            self.wfile.write(error_msg.encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Handle GET requests, including API endpoints and proxy."""
        parsed_path = urlparse(self.path)
        print(f"[serve_ui.py] GET {parsed_path.path}")  # Debug log

        if parsed_path.path == '/api/images':
            self.handle_images_api()
        elif parsed_path.path == '/api/tree-metadata':
            self.handle_tree_metadata_api()
        elif self._is_proxy_path(parsed_path.path):
            self._proxy_request('GET')
        else:
            # Serve static files (includes test_verify.html and ingest-data/)
            super().do_GET()

    def do_POST(self):
        """Handle POST requests by proxying to backend."""
        parsed_path = urlparse(self.path)

        if self._is_proxy_path(parsed_path.path):
            self._proxy_request('POST')
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def handle_images_api(self):
        """API endpoint to get images partitioned by tree ID."""
        try:
            images_by_tree = discover_images()
            response = {
                "success": True,
                "trees": images_by_tree,
                "total_trees": len(images_by_tree),
                "total_images": sum(len(imgs) for imgs in images_by_tree.values())
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

    def handle_tree_metadata_api(self):
        """API endpoint to get tree metadata from database."""
        try:
            milvus_repo = get_milvus_repo()
            if milvus_repo is None:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "Milvus not available"
                }).encode())
                return

            trees_metadata = milvus_repo.get_all_trees_metadata()

            response = {
                "success": True,
                "trees": trees_metadata,
                "total_trees": len(trees_metadata)
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            print(f"[serve_ui.py] Error getting tree metadata: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())


def extract_image_number(filename: str) -> int:
    """Extract numeric part from filename."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


def is_odd_number(filename: str) -> bool:
    """Check if the numeric part of filename is odd."""
    num = extract_image_number(filename)
    return num % 2 == 1


def get_trees_metadata() -> dict:
    """Get metadata for all trees from Milvus database.

    Returns:
        Dict mapping tree_id -> list of metadata dicts
    """
    try:
        milvus_repo = get_milvus_repo()
        if milvus_repo is None:
            return {}
        return milvus_repo.get_all_trees_metadata()
    except Exception as e:
        print(f"[serve_ui.py] Error getting trees metadata: {e}")
        return {}


def discover_images() -> dict:
    """Discover images from ingest-data and partition by tree ID.

    Also fetches metadata from Milvus database for each image.

    Returns:
        Dict mapping tree_id -> list of image info dicts with metadata
    """
    images_by_tree = {}

    if not INGEST_DATA_DIR.exists():
        return images_by_tree

    # Get metadata from Milvus
    trees_metadata = get_trees_metadata()

    # Check both ingest-image and verify-image subdirectories
    for subdir in ['ingest-image', 'verify-image']:
        subdir_path = INGEST_DATA_DIR / subdir
        if not subdir_path.exists():
            continue

        # Each folder in ingest-image or verify-image is a tree
        for tree_folder in subdir_path.iterdir():
            if not tree_folder.is_dir():
                continue

            tree_id = tree_folder.name

            # Get all jpg/png images
            images = sorted(tree_folder.glob("*.jpg")) + sorted(tree_folder.glob("*.png"))

            # Filter for odd-numbered images only
            images = [img for img in images if is_odd_number(img.name)]

            if images:
                # Get metadata for this tree
                tree_meta_list = trees_metadata.get(tree_id, [])

                # Build a lookup from image_id to metadata
                metadata_lookup = {}
                for meta in tree_meta_list:
                    img_id = meta.get("image_id", "")
                    if img_id:
                        metadata_lookup[img_id] = meta

                # Store image info with paths, names, and metadata
                images_by_tree[tree_id] = []
                for img in images:
                    img_name = img.name
                    # Try to match by image name (without extension)
                    base_name = Path(img_name).stem

                    # Look for metadata - try exact match first, then partial match
                    metadata = metadata_lookup.get(img_name) or metadata_lookup.get(base_name)

                    # Try partial match if no exact match
                    if not metadata:
                        for mid, mdata in metadata_lookup.items():
                            if base_name in mid or mid in base_name:
                                metadata = mdata
                                break

                    img_info = {
                        "path": f"{subdir}/{tree_id}/{img.name}",
                        "name": img.name
                    }

                    # Add metadata if available
                    if metadata:
                        img_info["latitude"] = metadata.get("latitude")
                        img_info["longitude"] = metadata.get("longitude")
                        img_info["hor_angle"] = metadata.get("hor_angle")
                        img_info["ver_angle"] = metadata.get("ver_angle")
                        img_info["pitch"] = metadata.get("pitch")
                        img_info["captured_at"] = metadata.get("captured_at")

                    images_by_tree[tree_id].append(img_info)

    return images_by_tree


def start_server():
    os.chdir(DIRECTORY)

    # Allow port reuse to avoid "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🌐 Server started at http://localhost:{PORT}")
        print(f"📄 Open test_verify.html in your browser")
        print(f"🔗 Direct link: http://localhost:{PORT}/test_verify.html")
        print(f"\n🔀 Reverse proxy: API calls → {BACKEND_URL}")
        print(f"   Proxied paths: {', '.join(PROXY_PREFIXES)}")
        print(f"\n📁 Image discovery:")
        images_by_tree = discover_images()
        print(f"   Found {len(images_by_tree)} trees with {sum(len(imgs) for imgs in images_by_tree.values())} images")
        for tree_id, imgs in images_by_tree.items():
            print(f"   - {tree_id}: {len(imgs)} images")
        print(f"\n⚠️  Make sure the FastAPI server is running at {BACKEND_URL}")
        print(f"   Start it with: python main.py")
        print(f"\n🌍 For ngrok: ngrok http {PORT}")
        print(f"   All API calls will be proxied automatically!")
        print(f"\nPress Ctrl+C to stop the server")

        # Open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}/test_verify.html')
        except:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")

if __name__ == "__main__":
    start_server()

