import argparse
import http.server
import socketserver
import sys


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Virtual Browser")

    parser.add_argument('--ip', type=str, default='0.0.0.0',
                        help='IP of HTTP video server.')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port number of HTTP video server.')
    return parser.parse_args()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    args = parse_args()
    ip = args.ip
    port = args.port
    server = ThreadedHTTPServer(
        (ip, port), http.server.SimpleHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
