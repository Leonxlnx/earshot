"""Generate Fish Audio voice previews through Monid; see MONID.md."""
import argparse
import json
import os
import time

import requests


class VoiceDesignError(Exception):
    """Fish Audio rejected the voice brief."""


def voice_design(instruction, *, timeout=60):
    key = os.environ.get('MONID_API_KEY')
    if not key:
        raise ValueError('set MONID_API_KEY to run the voice-design example')
    headers = {'Authorization': f'Bearer {key}'}
    deadline = time.monotonic() + timeout
    response = requests.post(
        'https://api.monid.ai/v1/run',
        headers={**headers, 'Content-Type': 'application/json; charset=utf-8'},
        data=json.dumps({'provider': 'fishaudio', 'endpoint': '/voice-design',
                         'input': {'body': {'instruction': instruction},
                                   'queryParams': {}, 'pathParams': {}}},
                        ensure_ascii=False).encode('utf-8'),
        timeout=timeout,
    )
    response.raise_for_status()
    run = response.json()
    if response.status_code == 202:
        run_id = run['runId']
        while run.get('status') not in ('COMPLETED', 'FAILED', 'TIMED_OUT', 'STOPPED', 'BLOCKED'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Monid voice design timed out')
            time.sleep(min(2, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Monid voice design timed out')
            response = requests.get(f'https://api.monid.ai/v1/runs/{run_id}',
                                    headers=headers, timeout=min(30, remaining))
            response.raise_for_status()
            run = response.json()
    if run.get('status') not in (None, 'COMPLETED'):
        raise RuntimeError(f"Monid voice design: {run['status']}")
    provider = run.get('providerResponse') or {}
    if provider.get('httpStatus', 200) >= 400:
        raise VoiceDesignError(f"Fish Audio HTTP {provider['httpStatus']}: {provider.get('error')}")
    return run['output']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instruction', help='voice description (gender, accent, tone, pacing)')
    args = parser.parse_args()
    try:
        print(json.dumps(voice_design(args.instruction), ensure_ascii=False, indent=2))
    except (ValueError, requests.RequestException, VoiceDesignError, RuntimeError, TimeoutError) as e:
        parser.exit(1, f'voice design: error: {e}\n')


if __name__ == '__main__':
    main()
