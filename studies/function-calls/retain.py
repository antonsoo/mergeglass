"""Retain complete native records and their original receipt chains, without weights or logs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import analyze
from run_chunks import sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generation", "teacher", "prompts", "summary", "independent_check", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--bfcl", type=Path, required=True, help="Pinned benchmark source checkout")
    parser.add_argument("--generation-check", type=Path, required=True)
    args = parser.parse_args()
    for root in (args.generation, args.teacher):
        assert not args.output.resolve().is_relative_to(root.resolve())
    prompts = json.loads(args.prompts.read_text())
    gen, _ = analyze.complete(args.generation, "generate", prompts)
    teacher, _ = analyze.complete(args.teacher, "teacher", prompts)
    assert gen["weight_states"] == teacher["weight_states"]
    report = json.loads(args.summary.read_text())
    checked = json.loads(args.independent_check.read_text())
    assert checked["serialized_summary_sha256"] == sha(args.summary)
    assert checked["independent_sql_queries_agree"]
    assert checked["category_and_context_budget_checked"] is True
    assert checked["generation_rows"] == 3948 and checked["teacher_questions"] == 656
    assert report["generation_receipt_sha256"] == sha(args.generation / "completed.json")
    assert report["teacher_receipt_sha256"] == sha(args.teacher / "completed.json")
    assert report["prompts_sha256"] == sha(args.prompts)
    regraded = json.loads(args.generation_check.read_text())
    assert regraded["script_sha256"] == sha(Path(__file__).with_name("check_generations.py"))
    assert regraded["generation_receipt_sha256"] == report["generation_receipt_sha256"]
    assert regraded["prompts_sha256"] == report["prompts_sha256"]
    assert regraded["grader_wrapper_sha256"] == gen["contract"]["source_hashes"]["bfcl.py"]
    assert regraded["model_runtimes_imported"] == []
    assert set(regraded["profiles"]) == set(gen["contract"]["profiles"])
    for profile in regraded["profiles"].values():
        assert profile["records_decoded_and_regraded"] == 658
        assert all(
            profile[key] is True
            for key in (
                "all_decoded_texts_match",
                "all_original_grade_dicts_match",
                "all_stops_and_token_bounds_match",
            )
        )
    blobs, originals = {}, {}

    def add(path: Path, name: str) -> None:
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert name not in blobs
        raw = path.read_bytes()
        assert b"/tmp/claude-1000/" not in raw and b"/home/antonsoloviev/" not in raw, path
        blobs[name] = raw
        originals[name] = path

    add(args.prompts, "prompts.json")
    add(args.summary, "summary.json")
    add(args.independent_check, "independent-check.json")
    add(args.generation_check, "generation-check.json")
    benchmark = args.bfcl / "berkeley-function-call-leaderboard"
    grader = json.loads((args.generation / "grader.json").read_text())
    for entry in grader["data_files"]:
        path = benchmark / entry["file"]
        assert sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]
        add(path, f"benchmark/{entry['file']}")
    for name in ("BFCL-LICENSE.txt", "NOTICE.md"):
        add(Path(__file__).parent / name, f"benchmark/{name}")
    for label, root, receipt in (
        ("generation", args.generation, gen),
        ("teacher", args.teacher, teacher),
    ):
        for name in ("contract.json", "completed.json", "progress.json", "grader.json"):
            add(root / name, f"{label}/{name}")
        for chunk in receipt["chunks"]:
            name = chunk["directory"]
            assert name == f"{chunk['profile']}-{chunk['start']:04d}-{chunk['stop']:04d}"
            for file in (chunk["receipt_file"], chunk["records_file"], "completed.json"):
                add(root / name / file, f"{label}/{name}/{file}")
            execution_path = root / name / "execution.json"
            if execution_path.exists():
                execution = json.loads(execution_path.read_text())
                assert execution["worker_returncode"] == 0
                assert execution["source_hashes"] == receipt["contract"]["source_hashes"]
                add(execution_path, f"{label}/{name}/execution.json")
    master_chunks = {chunk["directory"]: chunk for chunk in gen["chunks"]}

    def add_assisted(root: Path, prefix: str) -> None:
        for assisted in sorted(root.glob("assisted-*.json")):
            proof = json.loads(assisted.read_text())
            assert proof["worker_returncode"] == 0 and proof["coordinator_stopped_for_import"]
            assert proof["source_hashes"] == gen["contract"]["source_hashes"]
            assert master_chunks[proof["directory"]] == {
                "directory": proof["directory"],
                **proof["validated_chunk"],
            }
            add(assisted, f"{prefix}/{assisted.name}")

    add_assisted(args.generation, "generation")
    for path in sorted(args.generation.glob("imported-*.json")):
        imported = json.loads(path.read_text())
        directory = imported["worker_directory"]
        assert Path(directory).name == directory
        worker = args.generation.parent / directory
        assert sha(worker / "completed.json") == imported["worker_completed_sha256"]
        assert sha(worker / "contract.json") == imported["worker_contract_sha256"]
        finished = json.loads((worker / "completed.json").read_text())
        assert finished["chunks"] == imported["chunks"]
        assert all(master_chunks[chunk["directory"]] == chunk for chunk in finished["chunks"])
        add(path, f"generation/{path.name}")
        for name in ("contract.json", "completed.json", "grader.json"):
            add(worker / name, f"origins/{directory}/{name}")
        add_assisted(worker, f"origins/{directory}")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "native-cohorts.tar.gz"
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0, compresslevel=9) as stream:
            with tarfile.open(fileobj=stream, mode="w|") as tar:
                for name, data in sorted(blobs.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    tar.addfile(info, io.BytesIO(data))
    manifest = {
        name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in sorted(blobs.items())
    }
    with tarfile.open(archive, "r:gz") as tar:
        restored = {
            item.name: {
                "bytes": item.size,
                "sha256": hashlib.sha256(tar.extractfile(item).read()).hexdigest(),
            }
            for item in tar.getmembers()
            if item.isfile()
        }
    assert restored == manifest
    assert all(sha(originals[name]) == item["sha256"] for name, item in manifest.items())
    manifest_path = args.output / "native-cohorts-files.json.gz"
    manifest_path.write_bytes(
        gzip.compress((json.dumps(manifest, indent=2) + "\n").encode(), mtime=0)
    )
    (args.output / "archive.json").write_text(
        json.dumps(
            {
                "scope": "Complete canonical native outputs and original receipts; "
                "no model weights, "
                "worker logs or incomplete attempts. Parallel origins retain their metadata; "
                "their identical record bytes are present once in the canonical generation tree.",
                "script_sha256": sha(Path(__file__)),
                "files": len(manifest),
                "generation_questions_per_profile": 658,
                "generation_profiles": 6,
                "teacher_questions": 656,
                "all_member_bytes_match_originals": True,
                "archive": {
                    "file": archive.name,
                    "bytes": archive.stat().st_size,
                    "sha256": sha(archive),
                },
                "manifest": {"file": manifest_path.name, "sha256": sha(manifest_path)},
                "summary_sha256": sha(args.summary),
                "independent_check_sha256": sha(args.independent_check),
                "generation_check_sha256": sha(args.generation_check),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps({"retained_files": len(manifest), "archive_bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
