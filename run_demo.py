"""Bootstrap a real NVIDIA run, then serve the loopback-only review UI."""
import argparse
from pathlib import Path
import secrets
import subprocess
import sys

from review import Review, make_server

ROOT = Path(__file__).resolve().parent


def bootstrap():
    name = 'runs/bootstrap-' + secrets.token_hex(8)
    subprocess.run([sys.executable, str(ROOT / 'nvidia_ner_pipeline.py'),
                    '--images', str(ROOT / 'inputs/sample.png'),
                    '--outputs', str(ROOT / name)], cwd=ROOT, check=True)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18196)
    parser.add_argument('--run', help='Reuse a verified runs/name for inputs/sample.png')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    print('Preparing local demo; first NVIDIA inference can take several minutes.', flush=True)
    name = args.run or bootstrap()
    with make_server(Review(ROOT, name, 'inputs/sample.png'), args.port) as server:
        print(f'Review: http://127.0.0.1:{server.server_port} (unapproved); run={name}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()