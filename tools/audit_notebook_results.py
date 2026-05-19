#!/usr/bin/env python3
"""Audit executed notebook outputs for basic convergence sanity.

This is intentionally conservative. It does not prove that every model is
scientifically optimal; it catches stale stamps, missing outputs, error outputs,
and obviously bad final metrics such as all-zero accuracy, very high RMSE, or
translation examples with only zero BLEU scores.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


FRAMEWORKS = ("pytorch", "tensorflow", "jax", "mxnet")

TRIPLE_ACC_RE = re.compile(
    r"loss\s+([0-9.eE+-]+),\s+train acc\s+([0-9.eE+-]+),\s+test acc\s+([0-9.eE+-]+)"
)
VALID_ACC_RE = re.compile(
    r"(?:loss|train loss)\s+([0-9.eE+-]+),\s+train acc\s+([0-9.eE+-]+),\s+valid acc\s+([0-9.eE+-]+)"
)
TEST_ACC_RE = re.compile(r"test acc:?\s+([0-9.eE+-]+)")
VALID_ACC_ONLY_RE = re.compile(r"valid acc\s+([0-9.eE+-]+)")
RMSE_RE = re.compile(r"test RMSE\s+([0-9.eE+-]+)")
PERPLEXITY_RE = re.compile(r"perplexity\s+([0-9.eE+-]+)")
BLEU_RE = re.compile(r"bleu,([0-9.eE+-]+)")
OPT_LOSS_RE = re.compile(r"^loss:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
GAN_RE = re.compile(r"loss_D\s+([0-9.eE+-]+),\s+loss_G\s+([0-9.eE+-]+)")


def as_float(value: str) -> float | None:
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def output_to_text(output: dict) -> str:
    chunks: list[str] = []
    if output.get("output_type") == "error":
        chunks.append(output.get("ename", ""))
        chunks.append(output.get("evalue", ""))
        chunks.extend(output.get("traceback", []))
    if "text" in output:
        text = output["text"]
        chunks.append("".join(text) if isinstance(text, list) else str(text))
    data = output.get("data", {})
    for mime in ("text/plain", "text/html"):
        if mime in data:
            text = data[mime]
            chunks.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(chunks)


def cell_output_text(cell: dict) -> tuple[str, int]:
    chunks: list[str] = []
    error_outputs = 0
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            error_outputs += 1
        chunks.append(output_to_text(output))
    return "\n".join(chunks), error_outputs


def collect_output_text(nb: dict) -> tuple[str, int]:
    chunks: list[str] = []
    error_outputs = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text, errors = cell_output_text(cell)
        chunks.append(text)
        error_outputs += errors
    return "\n".join(chunks), error_outputs


def notebook_has_training(nb: dict) -> bool:
    needles = (
        "trainer.fit(",
        "d2l.train",
        "train_ch",
        "fit_epoch",
        "train_ranking",
        "train_recsys",
    )
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(needle in source for needle in needles):
            return True
    return False


def metrics_from_text(text: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)

    for loss, train_acc, test_acc in TRIPLE_ACC_RE.findall(text):
        metrics["loss"].append(as_float(loss))
        metrics["train_acc"].append(as_float(train_acc))
        metrics["test_acc"].append(as_float(test_acc))
    for loss, train_acc, valid_acc in VALID_ACC_RE.findall(text):
        metrics["loss"].append(as_float(loss))
        metrics["train_acc"].append(as_float(train_acc))
        metrics["valid_acc"].append(as_float(valid_acc))
    for value in TEST_ACC_RE.findall(text):
        metrics["test_acc"].append(as_float(value))
    for value in VALID_ACC_ONLY_RE.findall(text):
        metrics["valid_acc"].append(as_float(value))
    for value in RMSE_RE.findall(text):
        metrics["rmse"].append(as_float(value))
    for value in PERPLEXITY_RE.findall(text):
        metrics["perplexity"].append(as_float(value))
    for value in BLEU_RE.findall(text):
        metrics["bleu"].append(as_float(value))
    for value in OPT_LOSS_RE.findall(text):
        metrics["optim_loss"].append(as_float(value))
    for loss_d, loss_g in GAN_RE.findall(text):
        metrics["loss_d"].append(as_float(loss_d))
        metrics["loss_g"].append(as_float(loss_g))

    return {k: [v for v in vals if v is not None] for k, vals in metrics.items()}


def add_fine_tuning_phase_metrics(nb: dict, metrics: dict[str, list[float]]) -> None:
    metrics = defaultdict(list, metrics)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        text, _ = cell_output_text(cell)
        if not text:
            continue
        if "train_fine_tuning(finetune_net" in source or (
            "finetune_vars" in source and "train_fine_tuning" in source
        ):
            prefix = "fine_tune"
        elif "scratch_net" in source and "train_fine_tuning" in source:
            prefix = "scratch"
        else:
            continue
        phase_metrics = metrics_from_text(text)
        for key, values in phase_metrics.items():
            if values:
                metrics[f"{prefix}_{key}"].append(values[-1])
    return metrics


def metric_summary(metrics: dict[str, list[float]]) -> str:
    if metrics.get("fine_tune_test_acc") or metrics.get("fine_tune_valid_acc"):
        parts = []
        for prefix, label in (("fine_tune", "fine_tune"), ("scratch", "scratch")):
            subparts = []
            for key in ("loss", "train_acc", "test_acc", "valid_acc"):
                values = metrics.get(f"{prefix}_{key}", [])
                if values:
                    subparts.append(f"{key}={values[-1]:.3g}")
            if subparts:
                parts.append(f"{label}: " + ", ".join(subparts))
        return "; ".join(parts)

    parts = []
    for key in (
        "loss",
        "train_acc",
        "test_acc",
        "valid_acc",
        "rmse",
        "perplexity",
        "bleu",
        "optim_loss",
        "loss_d",
        "loss_g",
    ):
        values = metrics.get(key, [])
        if values:
            parts.append(f"{key}={values[-1]:.3g}")
    return ", ".join(parts)


def classify(rel: str, nb: dict, stamp: Path, output_text: str, error_outputs: int):
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    executed_cells = sum(c.get("execution_count") is not None for c in code_cells)
    output_cells = sum(bool(c.get("outputs")) for c in code_cells)
    stamp_exists = stamp.exists()
    stamp_current = stamp_exists and stamp.stat().st_mtime >= stamp.with_suffix(".ipynb").stat().st_mtime
    has_training = notebook_has_training(nb)
    metrics = metrics_from_text(output_text)
    if rel == "chapter_computer-vision/fine-tuning.ipynb":
        metrics = add_fine_tuning_phase_metrics(nb, metrics)

    issues: list[tuple[str, str]] = []
    if error_outputs:
        issues.append(("fail", f"{error_outputs} error output(s)"))
    if not stamp_exists:
        issues.append(("unverified", "missing .executed stamp; notebook has no passing execution record"))
    if stamp_exists and not stamp_current:
        issues.append(("unverified", "stale .executed stamp; notebook was regenerated after passing"))
    if stamp_exists and executed_cells == 0:
        issues.append(("unverified", "stamp exists but current notebook has no executed cells"))
    losses = metrics.get("fine_tune_loss", metrics.get("loss", []))
    train_accs = metrics.get("fine_tune_train_acc", metrics.get("train_acc", []))
    test_accs = metrics.get("fine_tune_test_acc", metrics.get("test_acc", []))
    valid_accs = metrics.get("fine_tune_valid_acc", metrics.get("valid_acc", []))
    rmses = metrics.get("rmse", [])
    bleus = metrics.get("bleu", [])
    perplexities = metrics.get("perplexity", [])
    optim_losses = metrics.get("optim_loss", [])

    if losses and train_accs and test_accs:
        if losses[-1] == 0 and train_accs[-1] == 0 and test_accs[-1] == 0:
            issues.append(("fail", "all-zero loss/train/test accuracy metric"))
        if train_accs[-1] >= 0.7 and test_accs[-1] < 0.5:
            issues.append(("warn", "test accuracy is low relative to training accuracy"))
    if losses and train_accs and valid_accs and train_accs[-1] >= 0.7 and valid_accs[-1] < 0.35:
        issues.append(("warn", "validation accuracy is low relative to training accuracy"))
    if rel == "chapter_computer-vision/fine-tuning.ipynb":
        ft_accs = metrics.get("fine_tune_test_acc", [])
        scratch_accs = metrics.get("scratch_test_acc", [])
        if ft_accs and scratch_accs and ft_accs[-1] + 0.05 < scratch_accs[-1]:
            issues.append(("warn", "fine-tuned model underperforms scratch baseline"))
        if ft_accs and ft_accs[-1] < 0.7:
            issues.append(("warn", f"fine-tuned test accuracy {ft_accs[-1]:.3f} is low"))
    if rmses:
        threshold = 2.0 if "mf.ipynb" in rel else 1.5
        if rmses[-1] > threshold:
            issues.append(("fail", f"test RMSE {rmses[-1]:.3f} exceeds {threshold:.1f}"))
    if bleus and max(bleus) == 0:
        issues.append(("warn", "all reported BLEU scores are zero"))
    if perplexities and perplexities[-1] > 100:
        issues.append(("warn", f"perplexity {perplexities[-1]:.1f} is high"))
    if optim_losses and optim_losses[-1] > 0.5:
        issues.append(("warn", f"optimizer demo final loss {optim_losses[-1]:.3f} is high"))

    return {
        "executed_cells": executed_cells,
        "output_cells": output_cells,
        "has_training": has_training,
        "metrics": metrics,
        "summary": metric_summary(metrics),
        "issues": issues,
        "stamp_exists": stamp_exists,
        "stamp_current": stamp_current,
    }


def audit(root: Path):
    rows = []
    for fw in FRAMEWORKS:
        for nb_path in sorted((root / fw).glob("chapter_*/*.ipynb")):
            rel = nb_path.relative_to(root / fw).as_posix()
            with nb_path.open(encoding="utf-8") as fh:
                nb = json.load(fh)
            output_text, error_outputs = collect_output_text(nb)
            stamp = nb_path.with_suffix(".executed")
            rows.append({"framework": fw, "notebook": rel, **classify(rel, nb, stamp, output_text, error_outputs)})
    return rows


def combined_issue(row: dict) -> tuple[str, str] | None:
    issues = row["issues"]
    if not issues:
        return None
    severity_rank = {"unverified": 0, "warn": 1, "fail": 2}
    severity = max((severity for severity, _ in issues), key=lambda x: severity_rank[x])
    message = "; ".join(issue for _, issue in issues)
    return severity, message


def write_markdown(rows, path: Path):
    counts = Counter((r["framework"], "ipynb") for r in rows)
    stamp_counts = Counter((r["framework"], "stamps") for r in rows if r["stamp_exists"])
    current_counts = Counter((r["framework"], "current") for r in rows if r["stamp_current"])
    output_counts = Counter((r["framework"], "outputs") for r in rows if r["output_cells"] > 0)
    issue_rows = [(r, combined_issue(r)) for r in rows if r["issues"]]
    issue_counts = Counter(issue[0] for _, issue in issue_rows if issue)

    lines = [
        "# Notebook Result Audit",
        "",
        "This report audits current `_notebooks` outputs and `.executed` stamps.",
        "",
        "## Coverage",
        "",
        "| Framework | Notebooks | Stamps | Current Stamps | Notebooks With Outputs |",
        "|-----------|-----------|--------|----------------|------------------------|",
    ]
    for fw in FRAMEWORKS:
        lines.append(
            f"| {fw} | {counts[(fw, 'ipynb')]} | {stamp_counts[(fw, 'stamps')]} | "
            f"{current_counts[(fw, 'current')]} | {output_counts[(fw, 'outputs')]} |"
        )

    lines += [
        "",
        "## Result",
        "",
        "| Severity | Count | Meaning |",
        "|----------|-------|---------|",
        f"| fail | {issue_counts['fail']} | current outputs contain explicit bad/error metrics |",
        f"| warn | {issue_counts['warn']} | current outputs look suspicious and need review |",
        f"| unverified | {issue_counts['unverified']} | existing stamp cannot prove current notebook output quality |",
    ]

    lines += [
        "",
        "## Issues",
        "",
        "| Severity | Framework | Notebook | Metrics | Issue |",
        "|----------|-----------|----------|---------|-------|",
    ]
    if not issue_rows:
        lines.append("| ok | all | all |  | no issues detected |")
    for r, issue in issue_rows:
        if issue is None:
            continue
        severity, message = issue
        lines.append(
            f"| {severity} | {r['framework']} | `{r['notebook']}` | "
            f"{r['summary'] or ''} | {message} |"
        )

    lines += [
        "",
        "## Passing Metrics Sample",
        "",
        "| Framework | Notebook | Metrics |",
        "|-----------|----------|---------|",
    ]
    shown = 0
    for r in rows:
        if r["issues"] or not r["summary"]:
            continue
        lines.append(f"| {r['framework']} | `{r['notebook']}` | {r['summary']} |")
        shown += 1
        if shown >= 80:
            break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks-dir", type=Path, default=Path("_notebooks"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = audit(args.notebooks_dir)
    if args.out:
        write_markdown(rows, args.out)
        print(f"wrote {args.out}")
        return

    for r in rows:
        issue = combined_issue(r)
        if issue is None:
            continue
        severity, message = issue
        print(f"{severity}\t{r['framework']}\t{r['notebook']}\t{r['summary']}\t{message}")


if __name__ == "__main__":
    main()
