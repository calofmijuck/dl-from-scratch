
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import numpy as np
from simple_conv_net import SimpleConvNet

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def _html(self, message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.
        """
        content = f"<html><body><h1>{message}</h1></body></html>"
        return content.encode("utf8")  # NOTE: must return a bytes object!

    def do_GET(self):
        self._set_headers()
        self.wfile.write(self._html("hi!"))

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        # Doesn't do anything with posted data
        content_length = int(self.headers["Content-Length"])
        data_string = self.rfile.read(content_length)

        data = json.loads(data_string)
        ans = int(predict(data['data']))

        result = json.dumps({"ans": ans})

        self._set_headers()
        self.wfile.write(bytes(result, 'utf-8'))


def run(server_class=HTTPServer, handler_class=S, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()

def init_model():
    global network
    network = SimpleConvNet()
    network.load_params("params.pkl")

def predict(x):
    x = np.array(x).reshape(1, 1, 28, 28)
    return np.argmax(network.predict(x))


if __name__ == "__main__":
    init_model()
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
