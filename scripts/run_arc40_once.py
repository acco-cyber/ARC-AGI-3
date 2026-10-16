"""Request one explicitly provisioned GPU smoke run; never submit blindly.

Credentials are read by the Kaggle SDK from KAGGLE_API_TOKEN or its normal
credential storage. This script never stores them. No accelerator fallback.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    meta = json.loads((ROOT/'launch/arc40/kernel-metadata.json').read_text())
    notebook = ROOT/'launch/arc40'/meta['code_file']
    receipt = {
        'requested_at_utc': datetime.now(timezone.utc).isoformat(),
        'kernel': meta['id'], 'mode': 'smoke', 'gpu': 'NvidiaRtxPro6000',
        'timeout_seconds': 1800, 'submission_attempted': False,
        'official_score': None,
        'notebook_sha256': hashlib.sha256(notebook.read_bytes()).hexdigest(),
        'account': api.get_config_value('username'),
    }
    status = 0
    try:
        receipt['accelerator_quota'] = json.loads(str(api.quota_view()))
    except Exception as exc:
        receipt['quota_read_error'] = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
    try:
        response = api.kernels_push(str(ROOT/'launch/arc40'), timeout='1800', acc='NvidiaRtxPro6000')
        receipt['response'] = json.loads(str(response))
        receipt['request_returned'] = True
    except Exception as exc:
        receipt['request_returned'] = False
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        status = 2
    Path(args.receipt).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    return status

if __name__ == '__main__':
    raise SystemExit(main())
