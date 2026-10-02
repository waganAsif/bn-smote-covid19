"""Download the training and test files from the CovidPred GitHub repository
(Zoabi et al., 2021) into data/."""

import sys
import urllib.request

from src import config


def download(name: str):
    target = config.DATA_DIR / f"{name}.zip"
    if target.exists() or (config.DATA_DIR / name).exists():
        print(f"{name}: already present, skipped")
        return
    url = config.COVIDPRED_RAW_URL.format(name=name)
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:  # network errors, proxies, etc.
        target.unlink(missing_ok=True)
        sys.exit(f"Download failed ({exc}). Download the file manually from "
                 f"https://github.com/nshomron/covidpred/tree/master/data and place it in data/.")
    print(f"  saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    config.DATA_DIR.mkdir(exist_ok=True)
    for file_name in (config.TRAIN_FILE, config.TEST_FILE):
        download(file_name)
