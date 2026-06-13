#!/usr/bin/env python3
"""Simple HTTP server with proxy support for work hosts."""
import http.server
import socketserver
import os

PORT = 12000
DIRECTORY = "/workspace/project/TRADE"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/dashboard.html'
        return super().do_GET()

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"Serving at http://0.0.0.0:{PORT}")
    print(f"Access dashboard at http://0.0.0.0:{PORT}/dashboard.html")
    httpd.serve_forever()