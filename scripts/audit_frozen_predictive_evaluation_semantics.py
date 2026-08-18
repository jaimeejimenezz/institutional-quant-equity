"""Inspect the exact evaluation semantics used by the frozen final ensemble."""

from __future__ import annotations

import ast
import inspect

import quant_equity.models.model_evaluation as evaluation
from quant_equity.config import PROJECT_ROOT

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_final_alpha_signal.py"


def _source_segment_for_assignment(
    source: str,
    tree: ast.AST,
    target_name: str,
) -> str:
    """Return the assignment statement for a named variable."""
    for node in ast.walk(tree):
        targets = []

        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        for target in targets:
            names = {child.id for child in ast.walk(target) if isinstance(child, ast.Name)}

            if target_name in names:
                segment = ast.get_source_segment(
                    source,
                    node,
                )

                if segment:
                    return segment

    return "<assignment not found>"


def main() -> None:
    """Print only the source needed to reproduce the frozen evaluation."""
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Final alpha script not found: {SCRIPT_PATH}")

    script_source = SCRIPT_PATH.read_text(encoding="utf-8")

    tree = ast.parse(
        script_source,
        filename=str(SCRIPT_PATH),
    )

    print("Institutional Quant Equity Research Platform")
    print("Frozen predictive evaluation semantics")
    print("================================================")

    print()
    print("evaluate_model_predictions source")
    print("---------------------------------")
    print(inspect.getsource(evaluation.evaluate_model_predictions).rstrip())

    print()
    print("_spearman_ic source")
    print("-------------------")
    print(inspect.getsource(evaluation._spearman_ic).rstrip())

    for variable in (
        "candidate_predictions",
        "ablation_predictions",
    ):
        print()
        print(f"{variable} assignment")
        print("-" * (len(variable) + len(" assignment")))
        print(
            _source_segment_for_assignment(
                script_source,
                tree,
                variable,
            )
        )

    print()
    print("Relevant final-alpha script window")
    print("----------------------------------")

    lines = script_source.splitlines()
    start = 180
    end = min(
        len(lines),
        265,
    )

    for index in range(
        start - 1,
        end,
    ):
        print(f"{index + 1:04d}: {lines[index]}")


if __name__ == "__main__":
    main()
