"""Compute one later native chunk and import it into a running profile coordinator.

The fixed inference worker and protocol stay unchanged. A disjoint output keeps
an explicit child exit record. Import briefly stops only the verified owning
coordinator; its current inference worker may continue in a different directory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from run_chunks import HERE, sha, validate_chunk, write_json


def owned_coordinator(pid: int, parent: Path, profile: str) -> str:
    proc = Path(f"/proc/{pid}")
    command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    assert str(parent.resolve()) in command and "generate" in command
    assert (proc / "cwd").resolve() == HERE.parents[1]
    assert (proc / "exe").resolve() == Path(sys.executable).resolve()
    assert any(
        argument.endswith("/run_chunks.py")
        and (HERE.parents[1] / argument).resolve() == (HERE / "run_chunks.py").resolve()
        for argument in command
    )
    contract = json.loads((parent / "contract.json").read_text())
    selected = [command[i + 1] for i, value in enumerate(command) if value == "--profile"]
    if selected:
        assert selected == contract["profiles"] == [profile]
    else:
        protocol = json.loads((HERE / "protocol.json").read_text())
        assert contract["profiles"] == protocol["profiles"] and contract["is_complete_cohort"]
        assert profile in contract["profiles"]
    state = next(
        line for line in (proc / "status").read_text().splitlines() if line.startswith("State:")
    ).split()[1]
    assert state not in {"Z", "X"}, "Coordinator has exited"
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "bfcl", "prompts", "parent", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--coordinator-pid", type=int, required=True)
    args = parser.parse_args()
    contract_path = args.parent / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract_hash, script_hash = sha(contract_path), sha(Path(__file__))
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert contract["mode"] == "generate" and args.profile in contract["profiles"]
    assert contract["profiles"] == [args.profile] or (
        contract["profiles"] == protocol["profiles"] and contract["is_complete_cohort"]
    )
    assert contract["target_records"] == contract["available_records"]
    size, start = contract["chunk_size"], args.start
    assert start > 0 and start % size == 0
    stop = min(start + size, contract["available_records"])
    assert stop > start
    for root in (args.parent, args.models, args.bfcl, HERE):
        assert not args.output.resolve().is_relative_to(root.resolve())
    prompts_list = json.loads(args.prompts.read_text())
    prompts = {row["id"]: row for row in prompts_list}
    ids = [row["id"] for row in prompts_list]
    assert len(prompts) == len(ids) == contract["available_records"]

    def unchanged() -> None:
        assert sha(contract_path) == contract_hash
        assert sha(args.prompts) == contract["prompts_sha256"]
        assert sha(args.parent / "grader.json") == contract["grader_sha256"]
        assert all(sha(HERE / name) == value for name, value in contract["source_hashes"].items())
        assert sha(Path(__file__)) == script_hash

    def frontier() -> dict:
        progress = json.loads((args.parent / "progress.json").read_text())
        last = progress["chunks"][-1]
        assert last["profile"] == args.profile
        assert start >= last["stop"] + size, "Leave the active native chunk disjoint"
        return last

    unchanged()
    last = frontier()
    assert owned_coordinator(args.coordinator_pid, args.parent, args.profile) != "T"
    name = f"{args.profile}-{start:04d}-{stop:04d}"
    destination = args.parent / name
    assert not destination.exists()
    args.output.mkdir(parents=True)
    command = [
        sys.executable,
        str(HERE / "replay.py"),
        str(args.models),
        str(args.bfcl),
        str(args.output),
        "--chunk-start",
        str(start),
        "--chunk-stop",
        str(stop),
        "--profile",
        args.profile,
    ]
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3000,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="2", HF_HUB_OFFLINE="1"),
        )
    execution = {
        "worker_returncode": process.returncode,
        "python": platform.python_version(),
        "script_sha256": script_hash,
        "parent_contract_sha256": contract_hash,
        "source_hashes": contract["source_hashes"],
        "profile": args.profile,
        "start": start,
        "stop": stop,
    }
    write_json(args.output / "execution.json", execution)
    assert process.returncode == 0, "Preserve failed work without importing it"
    unchanged()
    item = validate_chunk(
        args.output, "generate", args.profile, start, stop, ids[start:stop], prompts, contract
    )
    assert item["weight_states"] == last["weight_states"]
    write_json(args.output / "completed.json", item)
    frontier()
    assert owned_coordinator(args.coordinator_pid, args.parent, args.profile) != "T"
    stopped = False
    try:
        os.kill(args.coordinator_pid, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            if owned_coordinator(args.coordinator_pid, args.parent, args.profile) == "T":
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("Coordinator did not stop for the import")
        unchanged()
        current = frontier()
        assert not destination.exists(), "Never replace competing canonical work"
        temporary = Path(tempfile.mkdtemp(prefix=".future-", dir=args.parent))
        try:
            shutil.copytree(args.output, temporary, dirs_exist_ok=True)
            assert (
                validate_chunk(
                    temporary,
                    "generate",
                    args.profile,
                    start,
                    stop,
                    ids[start:stop],
                    prompts,
                    contract,
                )
                == item
            )
            assert json.loads((temporary / "completed.json").read_text()) == item
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        write_json(
            args.parent / f"assisted-{name}.json",
            {
                **execution,
                "frontier_stop_at_import": current["stop"],
                "coordinator_stopped_for_import": True,
                "directory": name,
                "validated_chunk": item,
            },
        )
    finally:
        if stopped:
            owned_coordinator(args.coordinator_pid, args.parent, args.profile)
            os.kill(args.coordinator_pid, signal.SIGCONT)
    print(json.dumps({"imported_native_chunk": name, "records": item["records"]}), flush=True)


if __name__ == "__main__":
    main()
