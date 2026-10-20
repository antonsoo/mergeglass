"""Decode and regrade complete native generations without loading model weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import analyze
from bfcl import BFCL, canonical, digest
from run_chunks import HERE, sha
from tokenizers import Tokenizer

DEPENDENCIES = (
    "tokenizers",
    "tree-sitter",
    "tree-sitter-java",
    "tree-sitter-javascript",
    "python-dotenv",
    "tenacity",
    "filelock",
)


def check_profile(benchmark: BFCL, tokenizer: Tokenizer, config: dict, rows: list[dict]) -> dict:
    source_rows = {row["id"]: row for row in benchmark.rows}
    assert [row["id"] for row in rows] == [row["id"] for row in benchmark.rows]
    eos = tokenizer.token_to_id(config["eos_token"])
    assert eos is not None
    cap = json.loads((HERE / "protocol.json").read_text())["generation"]["max_new_tokens"]
    grades, stops, accepted, ungraded = Counter(), Counter(), [], []
    for row in rows:
        source = source_rows[row["id"]]
        assert row["question_hash"] == digest(canonical([source["question"], source["function"]]))
        assert row["category"] == source["category"]
        assert row["function_name"] == source["function"][0]["name"]
        ids = row["generated_ids"]
        assert 0 < len(ids) <= cap and all(type(value) is int and value >= 0 for value in ids)
        assert all(tokenizer.id_to_token(value) is not None for value in ids)
        assert eos not in ids[:-1], "Generation must stop at its first EOS"
        text = tokenizer.decode(ids, skip_special_tokens=True)
        assert text == row["output"], row["id"]
        expected_stop = "eos" if ids[-1] == eos else "token_limit"
        assert row["stop"] == expected_stop
        assert expected_stop == "eos" or len(ids) == cap
        grade = benchmark.grade(source, text)
        assert canonical(grade) == canonical(row["grade"]), row["id"]
        grades[grade["status"]] += 1
        stops[expected_stop] += 1
        if grade["score"] == 1:
            accepted.append(row["id"])
        elif grade["score"] is None:
            ungraded.append({"id": row["id"], "status": grade["status"]})
    return {
        "records_decoded_and_regraded": len(rows),
        "all_decoded_texts_match": True,
        "all_original_grade_dicts_match": True,
        "all_stops_and_token_bounds_match": True,
        "statuses": dict(grades),
        "stops": dict(stops),
        "accepted_ids": accepted,
        "ungraded": ungraded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generation", "prompts", "bfcl", "tokenizer", "work", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    for root in (args.generation, args.bfcl, args.tokenizer):
        assert not args.output.resolve().is_relative_to(root.resolve())
        assert not args.work.resolve().is_relative_to(root.resolve())
    assert args.output.resolve() != args.prompts.resolve()
    script_hash, prompt_hash = sha(Path(__file__)), sha(args.prompts)
    prompts = json.loads(args.prompts.read_text())
    finished, generations = analyze.complete(args.generation, "generate", prompts)
    assert finished["contract"]["prompts_sha256"] == prompt_hash
    complete_hash = sha(args.generation / "completed.json")
    protocol = json.loads((HERE / "protocol.json").read_text())
    tokenizer_inputs = {}
    for item in protocol["tokenizer_files"]:
        path = args.tokenizer / item["file"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        tokenizer_inputs[item["file"]] = item["sha256"]
    config = json.loads((args.tokenizer / "tokenizer_config.json").read_text())
    assert config["clean_up_tokenization_spaces"] is False
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    benchmark = BFCL(args.bfcl, args.work, protocol)
    preflight = json.loads((args.generation / "grader.json").read_text())
    assert benchmark.data_files == preflight["data_files"]
    profiles = {
        profile: check_profile(benchmark, tokenizer, config, rows)
        for profile, rows in generations.items()
    }
    modules = benchmark.loaded_sources()
    assert modules == preflight["sources"]
    for module in modules:
        path = benchmark.root / module["file"]
        git_bytes = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.bfcl),
                "show",
                f"{protocol['bfcl_revision']}:berkeley-function-call-leaderboard/{module['file']}",
            ]
        )
        assert path.read_bytes() == git_bytes and digest(git_bytes) == module["sha256"]
    for chunk in finished["chunks"]:
        receipt = json.loads((args.generation / chunk["directory"] / "receipt.json").read_text())
        assert receipt["benchmark_modules"] == modules
        assert receipt["benchmark_files"] == benchmark.data_files
    model_imports = sorted({"torch", "transformers", "peft"}.intersection(sys.modules))
    assert not model_imports
    assert sha(args.generation / "completed.json") == complete_hash
    assert sha(args.prompts) == prompt_hash and sha(Path(__file__)) == script_hash
    result = {
        "scope": "Complete fixed generation cohort; fresh pinned upstream source and tokenizer; "
        "no model inference or generated function execution",
        "script_sha256": script_hash,
        "grader_wrapper_sha256": sha(HERE / "bfcl.py"),
        "generation_receipt_sha256": complete_hash,
        "prompts_sha256": prompt_hash,
        "bfcl_revision": protocol["bfcl_revision"],
        "tokenizer_files": tokenizer_inputs,
        "loaded_sources": modules,
        "data_files": benchmark.data_files,
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES},
        "model_runtimes_imported": model_imports,
        "profiles": profiles,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"profiles": len(profiles), "records": sum(len(r) for r in generations.values())}
        )
    )


if __name__ == "__main__":
    main()
