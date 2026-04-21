"""Download NYC Yellow Taxi Trip Record Parquet files into ``data/``.

The Parquet files are too large to keep inside the git repository, so
they live as assets of the tagged ``v1.0-data`` release on GitHub.
This script streams each missing file into ``data/`` so the assessment
framework can run end-to-end.

Usage::

    python scripts/download_data.py                 # fetch everything
    python scripts/download_data.py --year 2024     # only one year
    python scripts/download_data.py --month 2024-06 # a single month

The script is resumable: existing files of the right size are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# Release tag and repo that hosts the Parquet assets.
GITHUB_REPO = "junaidbabar314-droid/nyc-taxi-assessment"
RELEASE_TAG = "v1.0-data"

# All yellow-taxi months we ship.
MONTHS: list[str] = [f"2019-{m:02d}" for m in range(1, 13)] + \
                    [f"2024-{m:02d}" for m in range(1, 13)]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _asset_url(filename: str) -> str:
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download/"
        f"{RELEASE_TAG}/{filename}"
    )


def _stream_download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a basic progress indicator."""
    req = Request(url, headers={"User-Agent": "nyc-taxi-assessment-fetch/1.0"})
    try:
        with urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MiB
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as out:
                while True:
                    buf = response.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = 100 * downloaded / total
                        sys.stdout.write(
                            f"\r  {dest.name}: {downloaded / 1_048_576:6.1f} / "
                            f"{total / 1_048_576:6.1f} MB ({pct:5.1f}%)"
                        )
                        sys.stdout.flush()
            tmp.replace(dest)
            sys.stdout.write("\n")
    except (HTTPError, URLError) as exc:
        if dest.with_suffix(dest.suffix + ".part").exists():
            dest.with_suffix(dest.suffix + ".part").unlink()
        raise SystemExit(f"Download failed for {url}: {exc}") from exc


def _need_download(path: Path) -> bool:
    """Skip files that already exist with a non-zero size."""
    return not (path.exists() and path.stat().st_size > 0)


def _filter_months(year: int | None, month: str | None) -> list[str]:
    if month:
        return [month]
    if year:
        return [m for m in MONTHS if m.startswith(str(year))]
    return MONTHS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[2019, 2024],
                        help="Download only one year's files.")
    parser.add_argument("--month", type=str,
                        help="Download a single month in YYYY-MM format, "
                             "e.g. 2024-06.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    wanted = _filter_months(args.year, args.month)
    print(f"Target directory: {DATA_DIR}")
    print(f"Source release:   {GITHUB_REPO}@{RELEASE_TAG}")
    print(f"Fetching {len(wanted)} file(s).\n")

    for ym in wanted:
        filename = f"yellow_tripdata_{ym}.parquet"
        dest = DATA_DIR / filename
        if not _need_download(dest):
            print(f"  {filename}: already present, skipping")
            continue
        print(f"  {filename}: downloading ...")
        _stream_download(_asset_url(filename), dest)

    print("\nDone.  Run the assessments with:")
    print("    python run_all.py --year 2024 --month 6 --sample 0.01")


if __name__ == "__main__":
    main()
