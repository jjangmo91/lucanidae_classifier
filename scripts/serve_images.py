"""CORS 허용 이미지 서버 — python scripts/serve_images.py"""
import http.server
import socketserver

PORT = 8082
DIRECTORY = "data/final"


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # 로그 억제


with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    print(f"이미지 서버 실행 중: http://localhost:{PORT}")
    httpd.serve_forever()
