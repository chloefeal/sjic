#!/usr/bin/env python3
"""
Generate a signed SJIC license file (trial or official).

Usage:
  python scripts/generate_license.py \\
    --machine-code <hash> \\
    --edition trial \\
    --max-cameras 4 \\
    --out license.json

  python scripts/generate_license.py \\
    --machine-code <hash> \\
    --edition official \\
    --max-cameras 32 \\
    --expires-at 2027-12-31 \\
    --customer-id acme \\
    --out license-official.json

Signing key must match backend config:
  license.signing_key in config.yaml  OR  env LICENSE_SIGNING_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from backend/ or repo root
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.services.license_service import LicenseService  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Generate SJIC license.json')
    parser.add_argument('--machine-code', required=True, help='Target machine_code from /api/license/status')
    parser.add_argument('--edition', choices=['trial', 'official'], required=True)
    parser.add_argument('--max-cameras', type=int, required=True, help='Max video streams; <=0 means unlimited')
    parser.add_argument('--expires-at', default=None, help='Required for official (YYYY-MM-DD)')
    parser.add_argument('--customer-id', default='')
    parser.add_argument('--algorithms', default='', help='Comma-separated algorithm types; empty = all')
    parser.add_argument('--out', default='license.json')
    parser.add_argument('--signing-key', default=None, help='Override LICENSE_SIGNING_KEY / config')
    args = parser.parse_args()

    if args.signing_key:
        os.environ['LICENSE_SIGNING_KEY'] = args.signing_key

    if args.edition == 'official' and not args.expires_at:
        parser.error('--expires-at is required for official edition')

    algos = [a.strip() for a in args.algorithms.split(',') if a.strip()]

    svc = LicenseService()
    data = svc.build_license(
        edition=args.edition,
        machine_code=args.machine_code,
        max_cameras=args.max_cameras,
        expires_at=args.expires_at,
        allowed_algorithms=algos,
        customer_id=args.customer_id,
    )

    out_path = os.path.abspath(args.out)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'Wrote {out_path}')
    print(f"  edition={data['edition']} max_cameras={data['max_cameras']}")
    print(f"  expires_at={data['expires_at']}")
    print(f"  machine_code={data['machine_code']}")


if __name__ == '__main__':
    main()
