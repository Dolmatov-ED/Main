"""
download_hltv_demos.py — Скачивает до 50 про-матчей с HLTV.
Демки сохраняются в текущую папку.

Usage:
    pip install cloudscraper
    python download_hltv_demos.py
"""

import os, sys, re, time
from pathlib import Path

try:
    import cloudscraper
except ImportError:
    print("Install cloudscraper: pip install cloudscraper")
    sys.exit(1)

MAX_DEMOS = 50
OUT_DIR = Path.cwd()

scraper = cloudscraper.create_scraper()
scraper.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.5",
})


def _get_match_ids(limit=100):
    r = scraper.get("https://www.hltv.org/results", timeout=60)
    ids = re.findall(r'/matches/(\d+)/', r.text)
    return list(dict.fromkeys(ids))[:limit]


def _get_demo_url(match_id):
    for attempt in range(3):
        try:
            r = scraper.get(
                f"https://www.hltv.org/matches/{match_id}/-",
                timeout=60
            )
            m = re.search(r'href="(/download/demo/\d+)"', r.text)
            if m:
                return "https://www.hltv.org" + m.group(1)
            m = re.search(r'href="(https://www\.hltv\.org/download/demo/\d+)"', r.text)
            if m:
                return m.group(1)
            return ""
        except Exception as e:
            if attempt < 2:
                print(f"retry {attempt + 1}...", end=" ", flush=True)
                time.sleep(3)
            else:
                print(f"failed: {e}")
                return ""


def download_demos(limit=MAX_DEMOS):
    match_ids = _get_match_ids(limit * 2)
    print(f"  Found {len(match_ids)} matches")

    count = 0
    for mid in match_ids:
        if count >= limit:
            break
        dest = OUT_DIR / f"hltv_{mid}.dem"
        if dest.exists() or dest.with_suffix(".dem.gz").exists():
            continue

        url = _get_demo_url(mid)
        if not url:
            continue

        try:
            print(f"  [{count + 1}/{limit}] Match #{mid}...", end=" ", flush=True)
            r = scraper.get(url, stream=True, timeout=120, allow_redirects=True)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}")
                continue

            ct = r.headers.get("Content-Type", "")
            ext = ".dem" if "gzip" not in ct and "zip" not in ct else ".dem.gz"
            dest = dest.with_suffix(ext)

            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

            kb = dest.stat().st_size // 1024
            if kb > 10:
                print(f"{kb} KB ok")
                count += 1
            else:
                print("too small, skip")
                dest.unlink(missing_ok=True)
            time.sleep(1.5)
        except Exception as e:
            print(f"error: {e}")

    print(f"\n  Downloaded {count} demos to {OUT_DIR}")


if __name__ == "__main__":
    print("=" * 50)
    print("  HLTV demo downloader")
    print("=" * 50)
    download_demos()
