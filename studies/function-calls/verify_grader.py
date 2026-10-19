"""Check the study's upstream grading path against pinned answer alternatives."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bfcl import BFCL, digest


def literal(value, schema):
    if schema.get("type") == "dict" and isinstance(value, dict):
        properties = schema.get("properties", {})
        return {
            key: literal(next(x for x in options if x != ""), properties.get(key, {}))
            for key, options in value.items()
            if "" not in options
        }
    if schema.get("type") in ["array", "tuple"] and isinstance(value, list):
        return [literal(item, schema.get("items", {})) for item in value]
    if schema.get("type") == "float" and isinstance(value, int):
        return float(value)
    return value


def main() -> None:
    checkout, work, destination = map(Path, sys.argv[1:])
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    benchmark = BFCL(checkout, work, protocol)
    results = []
    for row in benchmark.rows:
        answer = benchmark.answers[row["id"]][0]
        assert len(answer) == 1
        name, options = next(iter(answer.items()))
        required = row["function"][0]["parameters"]["required"]
        values = {}
        for param, alternatives in options.items():
            if "" in alternatives and param not in required:
                continue
            value = next((x for x in alternatives if x != ""), [])
            values[param] = literal(value, row["function"][0]["parameters"]["properties"][param])
        text = "[" + name + "(" + ", ".join(f"{k}={v!r}" for k, v in values.items()) + ")]"
        grade = benchmark.grade(row, text)
        direct = None
        if row["id"] in benchmark.incomplete_references:
            decoded = benchmark.utils.default_decode_ast_prompting(text)
            direct = benchmark.checker.ast_checker(
                row["function"],
                decoded,
                benchmark.answers[row["id"]],
                language=benchmark.enums.Language.PYTHON,
                test_category=row["category"],
                model_name="mergeglass-smollm2",
            )
            assert not direct["valid"]
        results.append(
            {
                "id": row["id"],
                "score": grade["score"],
                "status": grade["status"],
                "constructed_call_sha256": digest(text.encode()),
                "upstream_empty_array_control": direct,
            }
        )
    # Known wrong name and wrong count must remain failures.
    first = benchmark.rows[0]
    negative = [
        benchmark.grade(first, "[other_function(base=10, height=5)]"),
        benchmark.grade(first, "[]"),
        benchmark.grade(first, "not a call"),
    ]
    assert all(row["score"] == 0 for row in negative)
    marker = work / "must-not-exist"
    expression = (
        f"[calculate_triangle_area(base=__import__('pathlib').Path({str(marker)!r})"
        ".write_text('ran') + 1, height=5)]"
    )
    blocked = benchmark.grade(first, expression)
    assert blocked["score"] is None and blocked["status"] == "unevaluated_expression"
    assert not marker.exists()
    result = {
        "bfcl_revision": protocol["bfcl_revision"],
        "positive_controls": len(results),
        "passed": sum(r["score"] == 1 for r in results),
        "incomplete_references": benchmark.incomplete_references,
        "negative_controls": negative,
        "unevaluated_expression_control": blocked,
        "generated_call_executed": False,
        "results": results,
        "sources": benchmark.loaded_sources(),
        "data_files": benchmark.data_files,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "positive_controls": len(results),
                "passed": result["passed"],
                "failed": [r for r in results if r["score"] != 1],
            }
        )
    )
    assert result["passed"] + len(benchmark.incomplete_references) == len(results)


if __name__ == "__main__":
    main()
