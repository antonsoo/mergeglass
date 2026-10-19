"""Use pinned BFCL prompts/checkers without importing provider clients or executing calls."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class BFCL:
    def __init__(self, checkout: Path, work: Path, protocol: dict[str, Any]):
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        assert revision == protocol["bfcl_revision"]
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "diff",
                "--exit-code",
                revision,
                "--",
                "berkeley-function-call-leaderboard",
            ],
            check=True,
            capture_output=True,
        )
        self.root = checkout / "berkeley-function-call-leaderboard"
        os.environ["BFCL_PROJECT_ROOT"] = str(work)
        sys.path.insert(0, str(self.root))
        # The AST checker reads only underscore_to_dot from this registry.
        # Upstream's full registry imports every API provider. Supply the one
        # explicit local model entry instead; no provider client is instantiated.
        registry = ModuleType("bfcl_eval.constants.model_config")
        registry.MODEL_CONFIG_MAPPING = {  # type: ignore[attr-defined]
            "mergeglass-smollm2": SimpleNamespace(underscore_to_dot=False)
        }
        sys.modules[registry.__name__] = registry
        self.utils = importlib.import_module("bfcl_eval.model_handler.utils")
        self.checker = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker")
        self.enums = importlib.import_module("bfcl_eval.constants.enums")
        self.rows = []
        self.answers = {}
        self.data_files = []
        for category in protocol["categories"]:
            filename = f"BFCL_v4_{category}.json"
            for relative in ["data/" + filename, "data/possible_answer/" + filename]:
                path = self.root / "bfcl_eval" / relative
                data = path.read_bytes()
                self.data_files.append(
                    {"file": "bfcl_eval/" + relative, "bytes": len(data), "sha256": digest(data)}
                )
                rows = [json.loads(line) for line in data.splitlines()]
                if "possible_answer" in relative:
                    for row in rows:
                        assert row["id"] not in self.answers
                        self.answers[row["id"]] = row["ground_truth"]
                else:
                    for row in rows:
                        assert len(row["question"]) == 1 and len(row["function"]) == 1
                        row["category"] = category
                    self.rows.extend(rows)
        assert len({row["id"] for row in self.rows}) == len(self.rows)
        assert {row["id"] for row in self.rows} == self.answers.keys()
        self.incomplete_references = {}
        for row in self.rows:
            function = row["function"][0]
            answer = self.answers[row["id"]][0][function["name"]]
            empty = [name for name in function["parameters"]["required"] if not answer.get(name)]
            if empty:
                self.incomplete_references[row["id"]] = empty

    def messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        return self.utils.system_prompt_pre_processing_chat_model(
            copy.deepcopy(row["question"][0]), copy.deepcopy(row["function"]), row["id"]
        )

    def grade(self, row: dict[str, Any], text: str) -> dict[str, Any]:
        if row["id"] in self.incomplete_references:
            return {
                "score": None,
                "status": "incomplete_reference",
                "parameters_without_alternatives": self.incomplete_references[row["id"]],
            }
        normalized = text.strip("`\n ")
        if not normalized.startswith("["):
            normalized = "[" + normalized
        if not normalized.endswith("]"):
            normalized += "]"
        try:
            tree = ast.parse(normalized, mode="eval")
        except (SyntaxError, ValueError, RecursionError):
            return {"score": 0, "status": "decode_error"}
        # Upstream resolves these two expression types with eval(). They are
        # outside this offline study. Keep the outcome unknown, not a failure.
        if any(isinstance(node, (ast.BinOp, ast.Lambda)) for node in ast.walk(tree)):
            return {"score": None, "status": "unevaluated_expression"}
        try:
            decoded = self.utils.default_decode_ast_prompting(
                text, language=self.enums.ReturnFormat.PYTHON, has_tool_call_tag=False
            )
        except (Exception, RecursionError) as exc:
            return {"score": 0, "status": "decode_error", "error_class": type(exc).__name__}
        try:
            result = self.checker.ast_checker(
                copy.deepcopy(row["function"]),
                decoded,
                copy.deepcopy(self.answers[row["id"]]),
                language=self.enums.Language.PYTHON,
                test_category=row["category"],
                model_name="mergeglass-smollm2",
            )
        except Exception as exc:
            return {"score": None, "status": "checker_error", "error_class": type(exc).__name__}
        return {
            "score": int(result["valid"]),
            "status": "valid" if result["valid"] else result.get("error_type", "ast_mismatch"),
            "decoded": decoded,
        }

    def loaded_sources(self) -> list[dict[str, Any]]:
        files = []
        for name, module in sorted(sys.modules.items()):
            if not name.startswith("bfcl_eval") or not getattr(module, "__file__", None):
                continue
            path = Path(module.__file__).resolve()
            if path.suffix == ".py" and path.is_relative_to(self.root):
                files.append(
                    {
                        "module": name,
                        "file": str(path.relative_to(self.root)),
                        "sha256": digest(path.read_bytes()),
                    }
                )
        return files
