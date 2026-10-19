"""Run and resume fixed-order inference chunks, validating every completed receipt.

Linux/POSIX research runner. The original protocol, prompts and scoring stay
fixed; an incomplete worker directory is retained and never accepted as a result.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VERSIONS = ("torch", "peft", "transformers", "tokenizers", "numpy", "safetensors")
SOURCES = (
    "replay.py",
    "teacher_forced.py",
    "bfcl.py",
    "verify_grader.py",
    "protocol.json",
    "teacher-forced-protocol.json",
    "run_chunks.py",
    "../dora/public-sources.json",
    "../dora/results/public.json",
)


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_chunk(
    folder: Path,
    mode: str,
    profile: str,
    start: int,
    stop: int,
    expected_ids: list[str],
    prompts: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = folder / ("receipt.json" if mode == "generate" else "teacher-forced.json")
    receipt = json.loads(receipt_path.read_text())
    selected = receipt["selection"]
    assert selected == {
        "start": start,
        "stop": stop,
        "available_records": contract["available_records"],
        "selected_records": stop - start,
        "is_complete_cohort": start == 0 and stop == contract["available_records"],
        "is_limited_pilot": False,
    }
    assert receipt["versions"] == contract["versions"]
    rows_path = folder / (f"{profile}.jsonl" if mode == "generate" else "questions.jsonl")
    rows = read_rows(rows_path)
    assert [row["id"] for row in rows] == expected_ids
    for row in rows:
        for key in ("question_hash", "prompt_hash"):
            assert row[key] == prompts[row["id"]][key]
    if mode == "generate":
        assert receipt["source_prompts_sha256"] == contract["prompts_sha256"]
        assert receipt["script_sha256"] == contract["source_hashes"]["replay.py"]
        assert receipt["grader_wrapper_sha256"] == contract["source_hashes"]["bfcl.py"]
        assert receipt["protocol_sha256"] == contract["source_hashes"]["protocol.json"]
        assert len(receipt["profiles"]) == 1
        state = receipt["profiles"][0]
        assert state["profile"] == profile and state["records"] == stop - start
        assert state["model_state_sha256"] == state["post_generation_state_sha256"]
        assert state["parameter_devices"] == ["cpu"]
        assert state["output_sha256"] == sha(rows_path)
        assert all(row["profile"] == profile for row in rows)
        weights = {profile: state["model_state_sha256"]}
    else:
        assert receipt["prompt_receipt_sha256"] == contract["prompts_sha256"]
        assert receipt["complete_protocol_records"] == contract["available_records"]
        assert receipt["questions"] == stop - start
        assert receipt["question_output_sha256"] == sha(rows_path)
        assert receipt["source_hashes"] == {
            name: contract["source_hashes"][name] for name in receipt["source_hashes"]
        }
        assert set(receipt["weight_identities"]) == set(contract["profiles"])
        weights = {}
        for name, state in receipt["weight_identities"].items():
            assert state["before_state_sha256"] == state["after_state_sha256"]
            assert state["parameter_devices"] == ["cpu"]
            weights[name] = state["before_state_sha256"]
    return {
        "profile": profile,
        "start": start,
        "stop": stop,
        "records": len(rows),
        "receipt_file": receipt_path.name,
        "receipt_sha256": sha(receipt_path),
        "records_file": rows_path.name,
        "records_sha256": sha(rows_path),
        "weight_states": weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "teacher"])
    parser.add_argument("models", type=Path)
    parser.add_argument("bfcl", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--pilot-records", type=int)
    args = parser.parse_args()
    if args.chunk_size <= 0 or (args.pilot_records is not None and args.pilot_records <= 0):
        parser.error("Chunk size and optional pilot count must be positive")
    if args.mode == "teacher" and args.profile:
        parser.error("Teacher-forced chunks always measure all six fixed profiles")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(args: argparse.Namespace) -> None:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2", HF_HUB_OFFLINE="1")
    protocol = json.loads((HERE / "protocol.json").read_text())
    source_hashes = {name: sha(HERE / name) for name in SOURCES}
    prompts_list = json.loads(args.prompts.read_text())
    prompts = {row["id"]: row for row in prompts_list}
    assert len(prompts) == len(prompts_list)
    preflight_path = args.output / "grader.json"
    if not preflight_path.exists():
        with (args.output / "grader.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "verify_grader.py"),
                    str(args.bfcl),
                    str(args.output / "grader-work"),
                    str(preflight_path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    preflight = json.loads(preflight_path.read_text())
    assert preflight["bfcl_revision"] == protocol["bfcl_revision"]
    assert set(row["id"] for row in preflight["results"]) == set(prompts)
    eligible = {row["id"] for row in preflight["results"] if row["score"] == 1}
    all_ids = [
        row["id"] for row in prompts_list if args.mode == "generate" or row["id"] in eligible
    ]
    profiles = args.profile or protocol["profiles"]
    assert set(profiles) <= set(protocol["profiles"]) and len(profiles) == len(set(profiles))
    target_count = min(len(all_ids), args.pilot_records or len(all_ids))
    contract = {
        "mode": args.mode,
        "source_hashes": source_hashes,
        "prompts_sha256": sha(args.prompts),
        "grader_sha256": sha(preflight_path),
        "available_records": len(all_ids),
        "target_records": target_count,
        "is_complete_cohort": target_count == len(all_ids) and not args.profile,
        "profiles": profiles,
        "chunk_size": args.chunk_size,
        "versions": {name: importlib.metadata.version(name) for name in VERSIONS},
    }
    contract_path = args.output / "contract.json"
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract, "Existing run contract differs"
    else:
        write_json(contract_path, contract)
    completed, weight_states = [], {}
    jobs = [
        (profile, start, min(start + args.chunk_size, target_count))
        for profile in (profiles if args.mode == "generate" else ["all-profiles"])
        for start in range(0, target_count, args.chunk_size)
    ]
    for profile, start, stop in jobs:
        assert all(sha(HERE / name) == value for name, value in source_hashes.items()), (
            "Source changed"
        )
        folder = args.output / f"{profile}-{start:04d}-{stop:04d}"
        marker = folder / "completed.json"
        if not marker.exists():
            if folder.exists():
                attempt = 1
                while folder.with_name(folder.name + f".interrupted-{attempt}").exists():
                    attempt += 1
                folder.rename(folder.with_name(folder.name + f".interrupted-{attempt}"))
            folder.mkdir()
            command = [
                sys.executable,
                str(HERE / ("replay.py" if args.mode == "generate" else "teacher_forced.py")),
                str(args.models),
                str(args.bfcl),
            ]
            if args.mode == "teacher":
                command.append(str(args.prompts))
            command.extend([str(folder), "--chunk-start", str(start), "--chunk-stop", str(stop)])
            if args.mode == "generate":
                command.extend(["--profile", profile])
                if start == 0:
                    command.append("--cache-control")
            print(
                json.dumps(
                    {
                        "starting": folder.name,
                        "completed_chunks": len(completed),
                        "total_chunks": len(jobs),
                    }
                ),
                flush=True,
            )
            with (folder / "worker.log").open("w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3000, check=True
                )
            item = validate_chunk(
                folder, args.mode, profile, start, stop, all_ids[start:stop], prompts, contract
            )
            write_json(marker, item)
        item = validate_chunk(
            folder, args.mode, profile, start, stop, all_ids[start:stop], prompts, contract
        )
        assert item == json.loads(marker.read_text())
        for name, state in item["weight_states"].items():
            if name in weight_states:
                assert weight_states[name] == state, "Model state differs between chunks"
            weight_states[name] = state
        completed.append({"directory": folder.name, **item})
        write_json(
            args.output / "progress.json",
            {"completed_chunks": len(completed), "total_chunks": len(jobs), "chunks": completed},
        )
        print(
            json.dumps(
                {
                    "completed": folder.name,
                    "completed_chunks": len(completed),
                    "total_chunks": len(jobs),
                }
            ),
            flush=True,
        )
    assert all(sha(HERE / name) == value for name, value in source_hashes.items())
    write_json(
        args.output / "completed.json",
        {"contract": contract, "chunks": completed, "weight_states": weight_states},
    )


if __name__ == "__main__":
    main()
