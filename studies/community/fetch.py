"""Fetch pinned public safetensors to a local directory; never an audit runtime step."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("sources.json"))
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads(args.sources.read_text())
    for source in manifest["sources"]:
        folder = root / source["role"]
        folder.mkdir(parents=True, exist_ok=True)
        for item in source["files"]:
            path = folder / item["file"]
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
                print(f"{source['role']}/{item['file']}: cached hash verified")
                continue
            url = (
                f"https://huggingface.co/{source['repository']}/resolve/"
                f"{source['revision']}/{item['file']}"
            )
            temporary = path.with_suffix(path.suffix + ".partial")
            digest = hashlib.sha256()
            count = 0
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    temporary.open("wb") as stream,
                ):
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        digest.update(block)
                        count += len(block)
                if count != item["bytes"] or digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Size/hash mismatch: {item['file']}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"{source['role']}/{item['file']}: {count} original bytes verified")


if __name__ == "__main__":
    main()
