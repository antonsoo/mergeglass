"""Independently check aggregate counts and floating measurements with SQLite JSON queries."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import analyze
from run_chunks import sha


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), (actual, expected)


def check_categories(db: sqlite3.Connection, profiles: list[str], report: dict) -> None:
    categories = {
        row[0]
        for table in ("generation", "teacher")
        for row in db.execute(f"SELECT DISTINCT json_extract(payload,'$.category') FROM {table}")
    }
    assert set(report["generation_by_category"]) == categories
    assert set(report["teacher_forced_by_category"]) == categories
    for category in categories:
        generated = report["generation_by_category"][category]
        targets = report["teacher_forced_by_category"][category]
        assert set(generated) == set(targets) == set(profiles)
        for profile in profiles:
            observed = generated[profile]
            n, accepted, rejected, known, tokens = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(CASE WHEN json_extract(payload,'$.grade.score')=1 "
                "THEN 1 ELSE 0 END),0),COALESCE(SUM(CASE WHEN "
                "json_extract(payload,'$.grade.score')=0 THEN 1 ELSE 0 END),0),"
                "COUNT(json_extract(payload,'$.grade.score')),COALESCE(SUM("
                "json_array_length(json_extract(payload,'$.generated_ids'))),0) "
                "FROM generation WHERE profile=? AND json_extract(payload,'$.category')=?",
                (profile, category),
            ).fetchone()
            assert (
                observed["questions"],
                observed["accepted"],
                observed["rejected"],
                observed["ungraded"],
                observed["generated_tokens"],
            ) == (n, accepted, rejected, n - known, tokens)
            if known:
                close(observed["accepted_fraction_of_graded"], accepted / known)
            else:
                assert observed["accepted_fraction_of_graded"] is None
            for column, path in (("statuses", "$.grade.status"), ("stops", "$.stop")):
                expected = dict(
                    db.execute(
                        "SELECT json_extract(payload,?),COUNT(*) FROM generation WHERE "
                        "profile=? AND json_extract(payload,'$.category')=? "
                        "GROUP BY json_extract(payload,?)",
                        (path, profile, category, path),
                    )
                )
                assert observed[column] == expected
            prefix = f'$.profiles."{profile}".'
            n, mean, weighted, tokens, matches, target_ids = db.execute(
                "SELECT COUNT(*),AVG(json_extract(payload,?)),SUM(json_extract(payload,?))/"
                "SUM(json_extract(payload,?)),SUM(json_extract(payload,?)),"
                "SUM(json_extract(payload,?)),SUM(json_array_length("
                "json_extract(payload,'$.target_ids'))) "
                "FROM teacher WHERE json_extract(payload,'$.category')=?",
                tuple(
                    prefix + field
                    for field in (
                        "mean_target_nll_nats",
                        "target_nll_sum_nats",
                        "target_tokens",
                        "target_tokens",
                        "teacher_forced_target_argmax_matches",
                    )
                )
                + (category,),
            ).fetchone()
            observed = targets[profile]
            assert n > 0 and observed["questions"] == n
            close(observed["mean_question_target_nll_nats"], mean)
            close(observed["token_weighted_target_nll_nats"], weighted)
            assert observed["target_tokens"] == tokens == target_ids
            assert observed["teacher_forced_target_argmax_matches"] == matches


def check_budget(db: sqlite3.Connection, prompts: list[dict], report: dict) -> None:
    db.execute("CREATE TABLE prompts(id TEXT PRIMARY KEY, payload TEXT)")
    db.executemany(
        "INSERT INTO prompts VALUES (?,?)", [(row["id"], json.dumps(row)) for row in prompts]
    )
    for table in ("generation", "teacher"):
        rows, joined, mismatched = db.execute(
            "SELECT (SELECT COUNT(*) FROM " + table + "),COUNT(*),COALESCE(SUM("
            "json_extract(a.payload,'$.input_tokens') IS NOT "
            "json_array_length(json_extract(p.payload,'$.input_ids'))),0) "
            "FROM " + table + " a JOIN prompts p ON json_extract(a.payload,'$.id')=p.id"
        ).fetchone()
        assert rows == joined and mismatched == 0
    observed = report["context_and_generation_budget"]
    longest = db.execute(
        "SELECT MAX(json_array_length(json_extract(payload,'$.input_ids'))) FROM prompts"
    ).fetchone()[0]
    assert observed["max_prompt_tokens"] == longest
    target, combined = db.execute(
        "SELECT MAX(json_array_length(json_extract(payload,'$.target_ids'))),"
        "MAX(json_extract(payload,'$.input_tokens')+"
        "json_array_length(json_extract(payload,'$.target_ids'))) FROM teacher"
    ).fetchone()
    assert observed["max_selected_target_tokens_including_eos"] == target
    assert observed["max_prompt_plus_selected_target_tokens"] == combined
    cap = json.loads((Path(__file__).parent / "protocol.json").read_text())["generation"][
        "max_new_tokens"
    ]
    assert observed["generation_token_cap"] == cap
    over_budget = [
        row[0]
        for row in db.execute(
            "SELECT json_extract(payload,'$.id') FROM teacher "
            "WHERE json_array_length(json_extract(payload,'$.target_ids'))>? ORDER BY rowid",
            (cap,),
        )
    ]
    assert observed["selected_references_exceeding_generation_budget"] == over_budget


def check(
    generations: dict,
    rows: list[dict],
    profiles: list[str],
    report: dict | None = None,
    *,
    prompts: list[dict] | None = None,
) -> dict:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE generation(profile TEXT, id TEXT, payload TEXT)")
    db.execute("CREATE TABLE teacher(payload TEXT)")
    db.executemany(
        "INSERT INTO generation VALUES (?,?,?)",
        [(p, r["id"], json.dumps(r, allow_nan=False)) for p, rs in generations.items() for r in rs],
    )
    db.executemany("INSERT INTO teacher VALUES (?)", [(json.dumps(r),) for r in rows])
    for profile in profiles:
        observed = (
            report["generation"][profile]
            if report is not None
            else analyze.generation(generations[profile])
        )
        n, accepted, rejected, known, tokens = db.execute(
            "SELECT COUNT(*),SUM(CASE WHEN json_extract(payload,'$.grade.score')=1 "
            "THEN 1 ELSE 0 END),"
            "SUM(CASE WHEN json_extract(payload,'$.grade.score')=0 THEN 1 ELSE 0 END),"
            "COUNT(json_extract(payload,'$.grade.score')),"
            "SUM(json_array_length(json_extract(payload,'$.generated_ids'))) "
            "FROM generation WHERE profile=?",
            (profile,),
        ).fetchone()
        assert (
            observed["questions"],
            observed["accepted"],
            observed["rejected"],
            observed["ungraded"],
            observed["generated_tokens"],
        ) == (n, accepted, rejected, n - known, tokens)
        if known:
            close(observed["accepted_fraction_of_graded"], accepted / known)
        else:
            assert observed["accepted_fraction_of_graded"] is None
        for column, path in [("statuses", "$.grade.status"), ("stops", "$.stop")]:
            expected = dict(
                db.execute(
                    "SELECT json_extract(payload,?),COUNT(*) FROM generation WHERE profile=? "
                    "GROUP BY json_extract(payload,?)",
                    (path, profile, path),
                )
            )
            assert observed[column] == expected
    target_stats = (
        report["teacher_forced"] if report is not None else analyze.teacher(rows, profiles)
    )
    for profile in profiles:
        prefix = f'$.profiles."{profile}".'
        mean, weighted, tokens, matches = db.execute(
            "SELECT AVG(json_extract(payload,?)),SUM(json_extract(payload,?))/"
            "SUM(json_extract(payload,?)),SUM(json_extract(payload,?)),"
            "SUM(json_extract(payload,?)) FROM teacher",
            tuple(
                prefix + field
                for field in [
                    "mean_target_nll_nats",
                    "target_nll_sum_nats",
                    "target_tokens",
                    "target_tokens",
                    "teacher_forced_target_argmax_matches",
                ]
            ),
        ).fetchone()
        observed = target_stats[profile]
        assert observed["questions"] == len(rows)
        close(observed["mean_question_target_nll_nats"], mean)
        close(observed["token_weighted_target_nll_nats"], weighted)
        assert observed["target_tokens"] == tokens
        assert observed["teacher_forced_target_argmax_matches"] == matches
        assert (
            tokens
            == db.execute(
                "SELECT SUM(json_array_length(json_extract(payload,'$.target_ids'))) FROM teacher"
            ).fetchone()[0]
        )
    comparisons = (
        report["probability_pairs"]
        if report is not None
        else analyze.probability_pairs(rows, profiles)
    )
    for reference in ("unmerged-f32", "unmerged-bf16"):
        for profile in profiles:
            if profile == reference:
                continue
            prefix = f'$.drift."{reference}"."{profile}".'
            observed = comparisons[reference][profile]
            assert observed["questions"] == len(rows)
            for field in ("kl_nats", "total_variation"):
                mean, weighted = db.execute(
                    "SELECT AVG(json_extract(payload,?)),SUM(json_extract(payload,?)*"
                    "json_extract(payload,?))/SUM(json_extract(payload,?)) FROM teacher",
                    (
                        prefix + "mean_" + field,
                        prefix + "mean_" + field,
                        prefix + "positions",
                        prefix + "positions",
                    ),
                ).fetchone()
                close(observed["mean_question_" + field], mean)
                close(observed["position_weighted_" + field], weighted)
            positions, disagreements, largest = db.execute(
                "SELECT SUM(json_extract(payload,?)),SUM(json_extract(payload,?)),"
                "MAX(json_extract(payload,?)) FROM teacher",
                (
                    prefix + "positions",
                    prefix + "argmax_disagreements",
                    prefix + "max_total_variation",
                ),
            ).fetchone()
            assert observed["positions"] == positions
            assert observed["argmax_disagreements"] == disagreements
            close(observed["max_position_total_variation"], largest)
            difference = db.execute(
                "WITH differences AS (SELECT json_extract(payload,?)-"
                "json_extract(payload,?) AS d FROM teacher) "
                "SELECT AVG(d),MIN(d),MAX(d),SUM(d<0),SUM(d>0),SUM(d=0) FROM differences",
                (
                    f'$.profiles."{profile}".mean_target_nll_nats',
                    f'$.profiles."{reference}".mean_target_nll_nats',
                ),
            ).fetchone()
            for field, value in zip(
                (
                    "mean_question_target_nll_difference_nats",
                    "min_question_target_nll_difference_nats",
                    "max_question_target_nll_difference_nats",
                ),
                difference[:3],
                strict=True,
            ):
                close(observed[field], value)
            assert (
                observed["lower_selected_target_nll_questions"],
                observed["higher_selected_target_nll_questions"],
                observed["equal_selected_target_nll_questions"],
            ) == difference[3:]
            paired = (
                report["paired_generation"][reference][profile]
                if report is not None
                else analyze.paired_generation(generations[reference], generations[profile])
            )
            expected = dict(
                db.execute(
                    "SELECT CASE WHEN json_extract(a.payload,'$.grade.score') IS NULL OR "
                    "json_extract(b.payload,'$.grade.score') IS NULL THEN 'ungraded' ELSE "
                    "'reference_'||json_extract(a.payload,'$.grade.score')||'_candidate_'||"
                    "json_extract(b.payload,'$.grade.score') END,COUNT(*) FROM generation a "
                    "JOIN generation b ON a.id=b.id WHERE a.profile=? AND b.profile=? GROUP BY 1",
                    (reference, profile),
                )
            )
            assert paired["paired_outcomes"] == expected
            identical = db.execute(
                "SELECT COUNT(*) FROM generation a JOIN generation b ON a.id=b.id "
                "WHERE a.profile=? AND b.profile=? AND json_extract(a.payload,'$.generated_ids')="
                "json_extract(b.payload,'$.generated_ids')",
                (reference, profile),
            ).fetchone()[0]
            assert paired["identical_token_sequences"] == identical
            assert paired["questions"] == len(generations[reference])
            assert paired["different_token_sequences"] == paired["questions"] - identical
    if report is not None:
        assert prompts is not None
        check_categories(db, profiles, report)
        check_budget(db, prompts, report)
    db.close()
    return {
        "sqlite": sqlite3.sqlite_version,
        "generation_profiles": len(profiles),
        "generation_rows": sum(len(r) for r in generations.values()),
        "teacher_questions": len(rows),
        "independent_sql_queries_agree": True,
        "category_and_context_budget_checked": report is not None,
        "float_tolerance": {"relative": 1e-12, "absolute": 1e-15},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--summary", type=Path, help="Check the actual serialized analysis artifact"
    )
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    gen_receipt, generations = analyze.complete(args.generation, "generate", prompts)
    teacher_receipt, teacher_data = analyze.complete(args.teacher, "teacher", prompts)
    assert gen_receipt["weight_states"] == teacher_receipt["weight_states"]
    for key in ("source_hashes", "versions", "profiles", "prompts_sha256"):
        assert gen_receipt["contract"][key] == teacher_receipt["contract"][key]
    assert sha(args.prompts) == gen_receipt["contract"]["prompts_sha256"]
    for root in (args.generation, args.teacher):
        assert not args.output.resolve().is_relative_to(root.resolve()), "Preserve input evidence"
    assert args.output.resolve() != args.prompts.resolve()
    report = None
    summary_hash = None
    if args.summary:
        assert args.summary.resolve() != args.output.resolve()
        summary_hash = sha(args.summary)
        report = json.loads(args.summary.read_text())
        assert report["analysis_sha256"] == sha(Path(analyze.__file__))
        assert report["generation_receipt_sha256"] == sha(args.generation / "completed.json")
        assert report["teacher_receipt_sha256"] == sha(args.teacher / "completed.json")
        assert report["prompts_sha256"] == sha(args.prompts)
        assert report["weight_states"] == gen_receipt["weight_states"]
        assert report["versions"] == gen_receipt["contract"]["versions"]
    result = check(
        generations,
        teacher_data["all-profiles"],
        gen_receipt["contract"]["profiles"],
        report,
        prompts=prompts,
    )
    result.update(
        {
            "script_sha256": sha(Path(__file__)),
            "analysis_sha256": sha(Path(analyze.__file__)),
            "complete_parent_receipts": {
                "generation": sha(args.generation / "completed.json"),
                "teacher": sha(args.teacher / "completed.json"),
                "prompts": sha(args.prompts),
            },
        }
    )
    if args.summary:
        assert sha(args.summary) == summary_hash
        result["serialized_summary_sha256"] = summary_hash
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
