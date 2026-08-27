"""Launch-day load against the running service, over HTTP.

loadsim.py measures the pipeline; this measures the thing visitors actually
touch: the FastAPI service under launchd, through its own queue, locks and
throttle. Each virtual visitor gets its own address, so the per-address rate
limit behaves as it would in the wild rather than throttling the whole test.

    python3 loadsim_http.py --users 12 --window 300 --evals 10000

Reports what a visitor feels (queue wait, total latency) and what the machine
feels (memory over baseline, sampled from vm_stat throughout).
"""

import argparse
import json
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8801'
CLIP = '/tmp/real30.mp3'   # a real Deezer preview, which is what the app fetches


# ----------------------------------------------------------------- sampling

def used_mb():
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        v = v.strip().rstrip('.')
        if v.isdigit():
            vals[k.strip()] = int(v)
    pages = (vals.get('Pages active', 0) + vals.get('Pages wired down', 0)
             + vals.get('Pages occupied by compressor', 0))
    return pages * 16384 / 1e6


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.baseline = used_mb()
        self.peak = self.baseline

    def run(self):
        while not self.stop_flag.is_set():
            self.peak = max(self.peak, used_mb())
            time.sleep(0.25)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)
        return self.peak - self.baseline


# --------------------------------------------------------------- http calls

def _request(path, ip, data=None, headers=None, timeout=900):
    req = urllib.request.Request(BASE + path, data=data, method='POST' if data else 'GET')
    # Only against the local service: Cloudflare rejects a request that tries
    # to set its own client-address header, and in front of the tunnel it sets
    # the real one anyway.
    if BASE.startswith('http://127.'):
        req.add_header('cf-connecting-ip', ip)
    # Cloudflare's bot protection 403s the default urllib agent, so against the
    # public hostname the harness has to look like a browser.
    req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/141.0 Safari/537.36')
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _multipart(fields, files):
    """Minimal multipart body, to keep this script dependency-free."""
    boundary = '----instrumentalloadsim'
    out = b''
    for k, v in fields.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                f'{v}\r\n').encode()
    for k, (name, blob) in files.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; '
                f'filename="{name}"\r\nContent-Type: audio/wav\r\n\r\n').encode()
        out += blob + b'\r\n'
    out += f'--{boundary}--\r\n'.encode()
    return out, f'multipart/form-data; boundary={boundary}'


def visit(idx, args, clip, results):
    """One visitor: search a song, then separate and match a clip."""
    ip = f'203.0.113.{idx + 1}'
    t_arrive = time.time()
    stage = {}
    try:
        t = time.time()
        _request('/api/search?q=daft+punk', ip, timeout=60)
        stage['search'] = time.time() - t

        t = time.time()
        body, ctype = _multipart({'stem_count': '4'}, {'file': ('clip.mp3', clip)})
        sep = _request('/api/separate', ip, data=body, headers={'Content-Type': ctype})
        stage['separate'] = time.time() - t

        t = time.time()
        # The endpoint takes the file field even when it reads a stem instead,
        # so the clip rides along and stem_name selects what actually gets matched.
        body, ctype = _multipart(
            {'stem_job_id': sep['job_id'], 'stem_name': 'other',
             'n_evals': str(args.evals)},
            {'file': ('clip.mp3', clip)})
        job = _request('/api/match-sequence', ip, data=body,
                       headers={'Content-Type': ctype})
        job_id = job['job_id']
        result = None
        while result is None:
            time.sleep(2)
            r = _request(f'/api/job-result/{job_id}', ip, timeout=60)
            if r.get('type') == 'complete':
                result = r
        stage['match'] = time.time() - t
        stage['loss'] = result.get('loss')
    except urllib.error.HTTPError as e:
        stage['http_error'] = f'{e.code} {e.read()[:120].decode(errors="replace")}'
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        stage['error'] = repr(e)[:160]

    stage['total'] = time.time() - t_arrive
    results.put((idx, stage))
    print(f'  visitor {idx:2d}  total {stage["total"]:6.1f}s  '
          + (f'match {stage.get("match", 0):5.1f}s  sep {stage.get("separate", 0):4.1f}s  '
             f'loss {stage.get("loss")}' if 'match' in stage
             else f'FAILED {stage.get("http_error") or stage.get("error")}'),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', type=int, default=12)
    ap.add_argument('--window', type=float, default=300)
    ap.add_argument('--evals', type=int, default=10000)
    args = ap.parse_args()

    clip = open(CLIP, 'rb').read()
    try:
        _request('/api/health', '127.0.0.1', timeout=10)
    except Exception as e:                      # noqa: BLE001
        sys.exit(f'service not answering on {BASE}: {e}')

    # Launch day: half the visitors in the first fifth of the window.
    rng = random.Random(5)
    burst = args.users // 2
    arrivals = sorted([rng.uniform(0, args.window * 0.2) for _ in range(burst)]
                      + [rng.uniform(args.window * 0.2, args.window)
                         for _ in range(args.users - burst)])
    print(f'{args.users} visitors over {args.window:.0f}s against {BASE}, '
          f'{args.evals} evals each')
    print('arrivals at: ' + ', '.join(f'{a:.0f}s' for a in arrivals))

    sampler = Sampler()
    sampler.start()
    print(f'baseline memory in use: {sampler.baseline / 1000:.2f} GB')

    results, threads = queue.Queue(), []
    t0 = time.time()
    for i, at in enumerate(arrivals):
        delay = at - (time.time() - t0)
        if delay > 0:
            time.sleep(delay)
        t = threading.Thread(target=visit, args=(i, args, clip, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    wall = time.time() - t0
    over = sampler.stop()
    rows = [r[1] for r in sorted(results.queue)]
    ok = [r for r in rows if 'match' in r]
    print(f'\nwall {wall:.0f}s, {len(ok)}/{len(rows)} completed')
    if ok:
        tot = sorted(r['total'] for r in ok)
        print(f'total latency   median {statistics.median(tot):6.1f}s   worst {tot[-1]:6.1f}s')
        print(f'separation      median '
              f'{statistics.median([r["separate"] for r in ok]):6.1f}s')
        print(f'match           median '
              f'{statistics.median([r["match"] for r in ok]):6.1f}s')
    for r in rows:
        if 'match' not in r:
            print('  failure:', r.get('http_error') or r.get('error'))
    print(f'memory over baseline, peak: {over:.0f} MB')


if __name__ == '__main__':
    main()
