#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify every source cited in data/*.json is live and contains its expected marker string.

Each entry names a URL plus an `expected_token` that must appear in the fetched body.
This provides an automated check that the cited page is reachable and retains the expected
marker text. It does not replace human verification of complex technical conclusions.

Network:
  By default probes 127.0.0.1:19001. If active, uses the proxy; otherwise falls back to
  a direct connection without environment proxy overrides.
  Override via CLOAK_PROXY:
    CLOAK_PROXY=off / direct / no -> force direct
    CLOAK_PROXY=http://... -> custom proxy

Verdicts:
  OK          page live and expected marker found
  STALE       page live but expected marker missing (requires human review)
  HTTP_ERROR  non-2xx response (404, 403, etc.)
  UNREACHABLE transport failure after retries
"""
import glob
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request

CLOAK_PROXY = "http://127.0.0.1:19001"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _proxy_port_open(host="127.0.0.1", port=19001, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def opener():
    """Return a urllib opener."""
    override = os.environ.get("CLOAK_PROXY", "").strip()
    if override.lower() in ("off", "direct", "no"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if override:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": override, "https": override})
        )
    if _proxy_port_open("127.0.0.1", 19001):
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": CLOAK_PROXY, "https": CLOAK_PROXY})
        )
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(op, url, expected_token, attempts=3):
    """Fetch a URL; return (http_status, verdict)."""
    token = re.sub(r"\s+", " ", expected_token.strip()).lower()
    if not token:
        return 0, "INVALID_TOKEN"

    last_err = "ERR: not attempted"
    for i in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
        try:
            with op.open(req, timeout=40) as r:
                text = r.read().decode("utf-8", "replace")
                clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).lower()
                if token in clean:
                    return r.status, "OK"
                return r.status, "STALE"
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUSES or i == attempts - 1:
                return e.code, "HTTP_ERROR"
            last_err = f"HTTP {e.code}"
            time.sleep(2 * (i + 1))
        except Exception as e:
            last_err = f"ERR: {e}"
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    return 0, f"UNREACHABLE ({last_err})"


def collect():
    """Pull verification_sources out of every data file, tagged by filename."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        for s in d.get("verification_sources", []):
            s = dict(s)
            s["_from"] = os.path.basename(path)
            out.append(s)
    return out


def main():
    op = opener()
    sources = collect()
    if not sources:
        # An empty list is a pass that proves nothing — fail instead.
        sys.exit("No verification_sources found. The gate must be able to fail.")

    failed = []
    for s in sources:
        status, res = check(op, s["url"], s["expected_token"])
        marker = "✅" if res == "OK" else "🔴"
        print(f"{marker} [{res:<11}] {status:<3} | {s['_from']:<18} | {s['url']}")
        if res != "OK":
            failed.append(f"{s['url']}  ({res}, {status})")

    if failed:
        print(f"\n🔴 {len(failed)}/{len(sources)} sources failed:")
        for f in failed:
            print(f"   - {f}")
        sys.exit(1)
    print(f"\nAll {len(sources)} sources verified.")


if __name__ == "__main__":
    main()
