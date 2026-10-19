"""Import a completed parallel profile into the same fixed master cohort.

Stop the master coordinator before invoking this command; its current worker
may continue on an earlier profile. Completed source chunks are never edited.
The master revalidates every imported chunk through its ordinary resume path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from run_chunks import HERE, sha, validate_chunk, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master", type=Path)
    parser.add_argument("worker", type=Path)
    parser.add_argument("prompts", type=Path)
    args = parser.parse_args()
    master = json.loads((args.master / "contract.json").read_text())
    finished = json.loads((args.worker / "completed.json").read_text())
    worker = finished["contract"]
    assert master["mode"] == worker["mode"] == "generate"
    assert master["is_complete_cohort"] and not worker["is_complete_cohort"]
    assert len(worker["profiles"]) == 1
    profile = worker["profiles"][0]
    assert profile in master["profiles"]
    assert worker["target_records"] == worker["available_records"]
    for key in master.keys() - {"profiles", "is_complete_cohort"}:
        assert worker[key] == master[key], key
    assert all(sha(HERE / name) == value for name, value in master["source_hashes"].items())
    assert sha(args.prompts) == master["prompts_sha256"]
    prompts_list = json.loads(args.prompts.read_text())
    prompts = {row["id"]: row for row in prompts_list}
    ids = [row["id"] for row in prompts_list]
    ranges = [
        (start, min(start + worker["chunk_size"], len(ids)))
        for start in range(0, len(ids), worker["chunk_size"])
    ]
    assert [(c["start"], c["stop"]) for c in finished["chunks"]] == ranges
    progress = json.loads((args.master / "progress.json").read_text())
    last_profile = progress["chunks"][-1]["profile"]
    assert master["profiles"].index(profile) > master["profiles"].index(last_profile), (
        "Import only a future profile, with the master coordinator stopped"
    )
    imports = []
    for chunk in finished["chunks"]:
        name, start, stop = chunk["directory"], chunk["start"], chunk["stop"]
        source, destination = args.worker / name, args.master / name
        item = validate_chunk(
            source, "generate", profile, start, stop, ids[start:stop], prompts, master
        )
        assert {"directory": name, **item} == chunk
        assert item["weight_states"] == finished["weight_states"]
        assert json.loads((source / "completed.json").read_text()) == item
        if destination.exists():
            assert (
                validate_chunk(
                    destination, "generate", profile, start, stop, ids[start:stop], prompts, master
                )
                == item
            ), "Refuse to overwrite differing master work"
            assert json.loads((destination / "completed.json").read_text()) == item
            imports.append({"directory": name, **item})
            continue
        temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=args.master))
        try:
            shutil.copytree(source, temporary, dirs_exist_ok=True)
            assert (
                validate_chunk(
                    temporary, "generate", profile, start, stop, ids[start:stop], prompts, master
                )
                == item
            )
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        imports.append({"directory": name, **item})
    receipt = args.master / f"imported-{profile}.json"
    result = {
        "worker_directory": args.worker.name,
        "worker_completed_sha256": sha(args.worker / "completed.json"),
        "worker_contract_sha256": sha(args.worker / "contract.json"),
        "script_sha256": sha(Path(__file__)),
        "chunks": imports,
    }
    if receipt.exists():
        assert json.loads(receipt.read_text()) == result
    else:
        write_json(receipt, result)
    print(json.dumps({"imported_profile": profile, "completed_chunks": len(imports)}))


if __name__ == "__main__":
    main()
