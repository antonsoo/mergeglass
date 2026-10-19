"""Summarize complete fixed cohorts; refuse missing chunks and mismatched identities."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from run_chunks import HERE, read_rows, sha, validate_chunk


def complete(root: Path, mode: str, prompts_list: list[dict]) -> tuple[dict, dict]:
    finished = json.loads((root / "completed.json").read_text())
    contract = finished["contract"]
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert contract["mode"] == mode and contract["is_complete_cohort"]
    assert contract["profiles"] == protocol["profiles"]
    assert contract["target_records"] == contract["available_records"]
    assert all(sha(HERE / name) == value for name, value in contract["source_hashes"].items())
    grader = root / "grader.json"
    assert sha(grader) == contract["grader_sha256"]
    preflight = json.loads(grader.read_text())
    eligible = {r["id"] for r in preflight["results"] if r["score"] == 1}
    prompts = {row["id"]: row for row in prompts_list}
    ids = [r["id"] for r in prompts_list if mode == "generate" or r["id"] in eligible]
    assert len(ids) == contract["available_records"]
    profiles = protocol["profiles"] if mode == "generate" else ["all-profiles"]
    expected = [
        (profile, start, min(start + contract["chunk_size"], len(ids)))
        for profile in profiles
        for start in range(0, len(ids), contract["chunk_size"])
    ]
    assert [(c["profile"], c["start"], c["stop"]) for c in finished["chunks"]] == expected
    data = {p: [] for p in profiles}
    weights = {}
    for chunk in finished["chunks"]:
        profile, start, stop = chunk["profile"], chunk["start"], chunk["stop"]
        folder = root / chunk["directory"]
        item = validate_chunk(
            folder, mode, profile, start, stop, ids[start:stop], prompts, contract
        )
        assert {"directory": chunk["directory"], **item} == chunk
        assert json.loads((folder / "completed.json").read_text()) == item
        for name, value in item["weight_states"].items():
            assert weights.get(name, value) == value
            weights[name] = value
        data[profile].extend(read_rows(folder / item["records_file"]))
    assert weights == finished["weight_states"]
    for rows in data.values():
        assert [row["id"] for row in rows] == ids
    return finished, data


def generation(rows: list[dict]) -> dict:
    grades = [row["grade"]["score"] for row in rows]
    assert all(value in (0, 1, None) for value in grades)
    known = sum(value is not None for value in grades)
    return {
        "questions": len(rows),
        "accepted": grades.count(1),
        "rejected": grades.count(0),
        "ungraded": grades.count(None),
        "accepted_fraction_of_graded": grades.count(1) / known if known else None,
        "statuses": dict(Counter(row["grade"]["status"] for row in rows)),
        "stops": dict(Counter(row["stop"] for row in rows)),
        "generated_tokens": sum(len(row["generated_ids"]) for row in rows),
    }


def paired_generation(reference: list[dict], candidate: list[dict]) -> dict:
    assert [r["id"] for r in reference] == [r["id"] for r in candidate]
    outcomes = Counter()
    identical = 0
    for a, b in zip(reference, candidate, strict=True):
        x, y = a["grade"]["score"], b["grade"]["score"]
        key = "ungraded" if x is None or y is None else f"reference_{x}_candidate_{y}"
        outcomes[key] += 1
        identical += a["generated_ids"] == b["generated_ids"]
    return {
        "questions": len(reference),
        "identical_token_sequences": identical,
        "different_token_sequences": len(reference) - identical,
        "paired_outcomes": dict(outcomes),
    }


def teacher(rows: list[dict], profiles: list[str]) -> dict:
    result = {}
    for profile in profiles:
        values = [row["profiles"][profile] for row in rows]
        tokens = sum(v["target_tokens"] for v in values)
        result[profile] = {
            "questions": len(rows),
            "target_tokens": tokens,
            "mean_question_target_nll_nats": math.fsum(v["mean_target_nll_nats"] for v in values)
            / len(rows),
            "token_weighted_target_nll_nats": math.fsum(v["target_nll_sum_nats"] for v in values)
            / tokens,
            "teacher_forced_target_argmax_matches": sum(
                v["teacher_forced_target_argmax_matches"] for v in values
            ),
        }
    return result


def probability_pairs(rows: list[dict], profiles: list[str]) -> dict:
    result = {}
    for reference in ("unmerged-f32", "unmerged-bf16"):
        result[reference] = {}
        for profile in profiles:
            if profile == reference:
                continue
            drift = [row["drift"][reference][profile] for row in rows]
            positions = sum(v["positions"] for v in drift)
            differences = [
                row["profiles"][profile]["mean_target_nll_nats"]
                - row["profiles"][reference]["mean_target_nll_nats"]
                for row in rows
            ]
            result[reference][profile] = {
                "questions": len(rows),
                "positions": positions,
                "mean_question_kl_nats": math.fsum(v["mean_kl_nats"] for v in drift) / len(rows),
                "position_weighted_kl_nats": math.fsum(
                    v["mean_kl_nats"] * v["positions"] for v in drift
                )
                / positions,
                "mean_question_total_variation": math.fsum(v["mean_total_variation"] for v in drift)
                / len(rows),
                "position_weighted_total_variation": math.fsum(
                    v["mean_total_variation"] * v["positions"] for v in drift
                )
                / positions,
                "max_position_total_variation": max(v["max_total_variation"] for v in drift),
                "argmax_disagreements": sum(v["argmax_disagreements"] for v in drift),
                "mean_question_target_nll_difference_nats": math.fsum(differences) / len(rows),
                "min_question_target_nll_difference_nats": min(differences),
                "max_question_target_nll_difference_nats": max(differences),
                "lower_selected_target_nll_questions": sum(v < 0 for v in differences),
                "higher_selected_target_nll_questions": sum(v > 0 for v in differences),
                "equal_selected_target_nll_questions": sum(v == 0 for v in differences),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analysis_hash = sha(Path(__file__))
    for root in (args.generation, args.teacher):
        assert not args.output.resolve().is_relative_to(root.resolve()), "Preserve input evidence"
    assert args.output.resolve() != args.prompts.resolve()
    prompt_hash = sha(args.prompts)
    prompts = json.loads(args.prompts.read_text())
    gen_receipt, generations = complete(args.generation, "generate", prompts)
    teacher_receipt, teacher_data = complete(args.teacher, "teacher", prompts)
    assert gen_receipt["weight_states"] == teacher_receipt["weight_states"]
    for key in ("source_hashes", "versions", "profiles", "prompts_sha256"):
        assert gen_receipt["contract"][key] == teacher_receipt["contract"][key], key
    assert gen_receipt["contract"]["prompts_sha256"] == prompt_hash
    profiles = gen_receipt["contract"]["profiles"]
    rows = teacher_data["all-profiles"]
    categories = sorted({row["category"] for row in rows})
    protocol = json.loads((HERE / "protocol.json").read_text())
    token_cap = protocol["generation"]["max_new_tokens"]
    result = {
        "scope": (
            "complete fixed local cohorts; unknown training overlap and training prompt; "
            "not a full BFCL submission or a held-out quality estimate"
        ),
        "teacher_target_definition": (
            "one selected valid call per question, including EOS; "
            "other acceptable answers are not scored"
        ),
        "context_and_generation_budget": {
            "max_prompt_tokens": max(len(row["input_ids"]) for row in prompts),
            "max_selected_target_tokens_including_eos": max(len(row["target_ids"]) for row in rows),
            "max_prompt_plus_selected_target_tokens": max(
                row["input_tokens"] + len(row["target_ids"]) for row in rows
            ),
            "generation_token_cap": token_cap,
            "selected_references_exceeding_generation_budget": [
                row["id"] for row in rows if len(row["target_ids"]) > token_cap
            ],
            "limit": (
                "selected reference lengths do not establish the shortest acceptable output; "
                "capped generations remain scored as actually observed"
            ),
        },
        "analysis_sha256": analysis_hash,
        "generation_receipt_sha256": sha(args.generation / "completed.json"),
        "teacher_receipt_sha256": sha(args.teacher / "completed.json"),
        "prompts_sha256": prompt_hash,
        "versions": gen_receipt["contract"]["versions"],
        "weight_states": gen_receipt["weight_states"],
        "generation": {p: generation(generations[p]) for p in profiles},
        "generation_by_category": {
            c: {p: generation([r for r in generations[p] if r["category"] == c]) for p in profiles}
            for c in categories
        },
        "paired_generation": {
            reference: {
                p: paired_generation(generations[reference], generations[p])
                for p in profiles
                if p != reference
            }
            for reference in ("unmerged-f32", "unmerged-bf16")
        },
        "teacher_forced": teacher(rows, profiles),
        "teacher_forced_by_category": {
            c: teacher([r for r in rows if r["category"] == c], profiles) for c in categories
        },
        "probability_pairs": probability_pairs(rows, profiles),
    }
    assert sha(args.prompts) == prompt_hash
    assert sha(Path(__file__)) == analysis_hash
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
