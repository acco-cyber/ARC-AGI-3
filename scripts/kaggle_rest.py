"""Small client for Kaggle's public www.kaggle.com/api/v1 REST routes.

Routes and request fields follow the installed official kaggle/kagglesdk client.
Authentication comes only from KAGGLE_API_TOKEN. No official competition
submission operation is exposed here. Mutating notebook pushes are explicit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request

BASE = "https://www.kaggle.com"


def request(path, *, query=None, payload=None, timeout=60):
    if not path.startswith('/api/v1/'):
        raise ValueError('Only documented Kaggle v1 REST paths are accepted')
    token = os.environ['KAGGLE_API_TOKEN']
    url = BASE + path
    if query:
        url += '?' + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + token,
                 'Content-Type': 'application/json', 'User-Agent': 'kaggle-api/v1.8.3'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def push_payload(directory, *, timeout_seconds):
    directory = Path(directory)
    metadata = json.loads((directory/'kernel-metadata.json').read_text())
    if metadata.get('machine_shape') != 'NvidiaRtxPro6000':
        raise ValueError('Exact validated accelerator is required')
    if not metadata.get('enable_gpu') or metadata.get('enable_internet'):
        raise ValueError('GPU must be enabled and notebook internet disabled')
    notebook_path = directory/metadata['code_file']
    notebook = copy.deepcopy(json.loads(notebook_path.read_text()))
    for cell in notebook['cells']:
        cell['source'] = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    payload = {
        'id': metadata.get('id_no'), 'slug': metadata['id'],
        'newTitle': metadata['title'], 'text': json.dumps(notebook),
        'language': metadata['language'], 'kernelType': metadata['kernel_type'],
        'isPrivate': metadata['is_private'], 'enableGpu': True,
        'enableTpu': False, 'enableInternet': False,
        'datasetDataSources': metadata.get('dataset_sources', []),
        'competitionDataSources': metadata.get('competition_sources', []),
        'kernelDataSources': metadata.get('kernel_sources', []),
        'modelDataSources': metadata.get('model_sources', []),
        'categoryIds': metadata.get('keywords', []),
        'dockerImagePinningType': metadata['docker_image_pinning_type'],
        'dockerImage': metadata['docker_image'],
        'sessionTimeoutSeconds': int(timeout_seconds),
        'machineShape': 'NvidiaRtxPro6000',
    }
    return payload, {'kernel': metadata['id'],
                     'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                     'timeout_seconds': timeout_seconds,
                     'submission_attempted': False}


def save_output(kernel, directory, pattern=r'(?:arc.*\.json|benchmark\.json|serve\.log|.*probe.*)'):
    owner, slug = kernel.split('/', 1)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    token = None
    saved = []
    while True:
        result = request('/api/v1/kernels/output', query={
            'userName': owner, 'kernelSlug': slug, 'pageSize': 100, 'pageToken': token})
        if result.get('log'):
            (directory/'kernel.log').write_text(result['log'])
        for item in result.get('files', []):
            name = item['fileName']
            if not re.search(pattern, name):
                continue
            target = (directory/name).resolve()
            if not target.is_relative_to(directory):
                raise ValueError('Unsafe output filename')
            target.parent.mkdir(parents=True, exist_ok=True)
            # These are normal download URLs returned by Kaggle. Do not send
            # account credentials to the storage host or persist signed URLs.
            with urllib.request.urlopen(item['url'], timeout=60) as response:
                data = response.read(32 * 1024 * 1024 + 1)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError('Compact diagnostic artifact exceeds32MiB')
            target.write_bytes(data)
            saved.append({'file': name, 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()})
        token = result.get('nextPageToken')
        if not token:
            break
    receipt = {'kernel': kernel, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
               'files': saved}
    (directory/'download-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='operation', required=True)
    status = sub.add_parser('status')
    status.add_argument('kernel')
    output = sub.add_parser('output')
    output.add_argument('kernel')
    output.add_argument('directory')
    push = sub.add_parser('push')
    push.add_argument('directory')
    push.add_argument('--timeout', type=int, required=True)
    push.add_argument('--receipt', required=True)
    args = parser.parse_args()
    if args.operation == 'status':
        owner, slug = args.kernel.split('/', 1)
        print(json.dumps(request('/api/v1/kernels/status', query={
            'userName': owner, 'kernelSlug': slug}), indent=2))
    elif args.operation == 'output':
        print(json.dumps(save_output(args.kernel, args.directory), indent=2))
    else:
        payload, receipt = push_payload(args.directory, timeout_seconds=args.timeout)
        receipt['requested_at_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['response'] = request('/api/v1/kernels/push', payload=payload, timeout=90)
        # New notebooks may receive a title-derived slug. Monitor the returned
        # canonical URL rather than assuming the requested slug was retained.
        returned_url = receipt['response'].get('url')
        if returned_url:
            canonical_path = urllib.parse.urlsplit(returned_url).path
            if canonical_path.startswith('/code/'):
                receipt['requested_kernel'] = receipt['kernel']
                receipt['kernel'] = canonical_path.removeprefix('/code/')
        Path(args.receipt).write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
