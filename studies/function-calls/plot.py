"""Plot measured merge effects only after complete, independently checked cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("independent_check", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary_hash, check_hash = sha(args.summary), sha(args.independent_check)
    report = json.loads(args.summary.read_text())
    checked = json.loads(args.independent_check.read_text())
    assert checked["serialized_summary_sha256"] == summary_hash
    assert checked["independent_sql_queries_agree"]
    assert checked["generation_profiles"] == 6 and checked["generation_rows"] == 3948
    assert checked["teacher_questions"] == 656
    assert checked["analysis_sha256"] == report["analysis_sha256"]
    for name, field in (
        ("generation", "generation_receipt_sha256"),
        ("teacher", "teacher_receipt_sha256"),
        ("prompts", "prompts_sha256"),
    ):
        assert checked["complete_parent_receipts"][name] == report[field]
    assert len(report["generation"]) == len(report["teacher_forced"]) == 6
    assert all(value["questions"] == 658 for value in report["generation"].values())
    assert all(value["questions"] == 656 for value in report["teacher_forced"].values())
    comparisons = [
        ("unmerged-f32", "merged-f32", "FP32 native merge\nvs FP32 unmerged", "#2b6cb0"),
        ("unmerged-bf16", "merged-bf16", "BF16 native merge\nvs BF16 unmerged", "#ac6800"),
        ("unmerged-f32", "as-lora-f32", "FP32 control: omit DoRA m\nvs FP32 unmerged", "#9d4440"),
    ]
    positions = report["teacher_forced"]["unmerged-f32"]["target_tokens"]
    kl, argmax, sequences, labels, colors = [], [], [], [], []
    for reference, candidate, label, color in comparisons:
        drift = report["probability_pairs"][reference][candidate]
        paired = report["paired_generation"][reference][candidate]
        assert drift["positions"] == positions
        assert paired["questions"] == 658
        assert 0 <= drift["argmax_disagreements"] <= positions
        assert 0 <= paired["different_token_sequences"] <= 658
        value = drift["position_weighted_kl_nats"]
        assert math.isfinite(value) and value >= 0
        kl.append(value)
        argmax.append(drift["argmax_disagreements"])
        sequences.append(paired["different_token_sequences"])
        labels.append(label)
        colors.append(color)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("merge-effects.svg", "merge-effects.png", "figure.json"):
        assert (args.output / name).resolve() not in {
            args.summary.resolve(),
            args.independent_check.resolve(),
        }
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {"font.size": 9, "svg.fonttype": "path", "svg.hashsalt": "mergeglass-function-calls-v1"}
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.76, bottom=0.28, wspace=0.30)
    y = [0, 1, 2]
    values = [kl, [n / positions for n in argmax], [n / 658 for n in sequences]]
    captions = [
        [f"{v:.3g}" for v in kl],
        [f"{n:,}/{positions:,}" for n in argmax],
        [f"{n}/658" for n in sequences],
    ]
    titles = [
        "Next-token probability drift",
        "Changed argmax at fixed prefixes",
        "Different greedy token sequences",
    ]
    for axis, data, texts, title in zip(axes, values, captions, titles, strict=True):
        axis.barh(y, data, color=colors, height=0.48)
        axis.set_title(title, fontsize=9.5, pad=14)
        axis.set_yticks(y, labels)
        axis.tick_params(axis="y", length=0, pad=12)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        for row, text in zip(y, texts, strict=True):
            axis.text(
                0.98,
                row,
                text,
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2},
            )
    if all(value > 0 for value in kl):
        axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean KL(reference || candidate), nats")
    for axis, data in zip(axes[1:], values[1:], strict=True):
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.set_xlim(0, min(1, max(0.01, max(data) * 1.25)))
    axes[1].set_xlabel("Selected-target prediction positions")
    axes[2].set_xlabel("All 658 questions")
    axes[0].invert_yaxis()
    fig.suptitle("Native DoRA merge effects at identical inputs", fontsize=14, y=0.97)
    fig.text(
        0.24,
        0.85,
        "Local CPU runtime; one pinned base and adapter; fixed BFCL prompt",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.02,
        0.11,
        f"Probability panels: 656 selected valid calls, {positions:,} prefixes including EOS. "
        "KL is position-weighted.\n"
        "Greedy panel: 658 recorded outputs, capped at 160 new tokens. "
        "Token differences are not differences in task accuracy.\n"
        "Training prompt and training overlap are unknown; "
        "this is not a held-out quality estimate.",
        fontsize=8.2,
        linespacing=1.6,
        color="#444444",
    )
    fig.savefig(args.output / "merge-effects.svg", metadata={"Date": None, "Creator": "Mergeglass"})
    fig.savefig(args.output / "merge-effects.png", dpi=170, metadata={"Software": "Mergeglass"})
    plt.close(fig)
    assert sha(args.summary) == summary_hash and sha(args.independent_check) == check_hash
    (args.output / "figure.json").write_text(
        json.dumps(
            {
                "summary_sha256": summary_hash,
                "independent_check_sha256": check_hash,
                "script_sha256": sha(Path(__file__)),
                "matplotlib": matplotlib.__version__,
                "python": platform.python_version(),
                "comparison_references": comparisons,
                "files": {
                    name: sha(args.output / name)
                    for name in ("merge-effects.svg", "merge-effects.png")
                },
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
