"""Download the official Azure LMM inference trace and sample a window.

Fetches AzureLMMInferenceTrace_multimodal.csv.gz from the Azure public dataset
(CC-BY), caches it at SOURCE_TRACE, renames columns to the lowercase the
pipeline expects, and writes a sampled window to traces/<TRACE>.csv.

    pixi run python utils/fetch_trace.py                 # first hour -> traces/<TRACE>.csv
    pixi run python utils/fetch_trace.py --start-hour 5 --hours 1
"""

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from lmm_serving import config as cfg

URL = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLMMInferenceTrace_multimodal.csv.gz"
COLUMNS = {
    "NumImages": "num_images",
    "ContextTokens": "n_context_tokens",
    "GeneratedTokens": "n_generated_tokens",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def download_canonical() -> Path:
    dst = Path(cfg.SOURCE_TRACE)
    if dst.exists():
        print(f"cached: {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    gz = dst.with_suffix(dst.suffix + ".gz")
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, gz)
    with gzip.open(gz, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    gz.unlink()
    print(f"wrote {dst}")
    return dst


def sample(src: Path, start_hour: float, hours: float) -> pd.DataFrame:
    df = pd.read_csv(src).rename(columns=COLUMNS)
    ts = pd.to_datetime(df["TIMESTAMP"])
    t0 = ts.min()
    lo = t0 + pd.Timedelta(hours=start_hour)
    hi = lo + pd.Timedelta(hours=hours)
    return df[(ts >= lo) & (ts < hi)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-hour", type=float, default=0.0)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "traces" / f"{cfg.TRACE}.csv"))
    a = ap.parse_args()

    window = sample(download_canonical(), a.start_hour, a.hours)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    window.to_csv(out, index=False)
    img = (window["num_images"] > 0).mean()
    print(f"wrote {out}: {len(window)} requests, {img:.0%} with images")


if __name__ == "__main__":
    main()
