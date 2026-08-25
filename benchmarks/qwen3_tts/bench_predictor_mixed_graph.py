#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark mixed sampled/argmax Qwen3-TTS Predictor CUDA Graph dispatch.

This is an isolated Predictor benchmark. It reuses the deterministic miniature
Predictor fixture from the CUDA Graph tests and does not represent end-to-end
TTS serving throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


def interleaved_modes(rounds: int) -> list[tuple[str, str]]:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    return [
        ("eager", "graphed") if index % 2 == 0 else ("graphed", "eager")
        for index in range(rounds)
    ]


def _nearest_rank(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize empty samples")
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def summarize(samples: Iterable[float]) -> dict[str, float | int]:
    values = sorted(float(value) for value in samples)
    if not values:
        raise ValueError("cannot summarize empty samples")
    return {
        "count": len(values),
        "mean_us": statistics.fmean(values),
        "p10_us": _nearest_rank(values, 0.10),
        "p50_us": _nearest_rank(values, 0.50),
        "p90_us": _nearest_rank(values, 0.90),
    }


def build_report(
    *,
    metadata: dict[str, Any],
    config: dict[str, Any],
    eager_samples: Iterable[float],
    graphed_samples: Iterable[float],
    eager_aa_samples: Iterable[float],
    graph_hits: int,
    graph_attempts: int,
    correctness: dict[str, bool],
) -> dict[str, Any]:
    if graph_attempts < 1 or not 0 <= graph_hits <= graph_attempts:
        raise ValueError("graph hit counters are inconsistent")
    eager_values = [float(value) for value in eager_samples]
    graphed_values = [float(value) for value in graphed_samples]
    eager_aa_values = [float(value) for value in eager_aa_samples]
    arms = {
        "eager": {
            "samples_us": eager_values,
            "summary": summarize(eager_values),
        },
        "graphed": {
            "samples_us": graphed_values,
            "summary": summarize(graphed_values),
        },
        "eager_aa": {
            "samples_us": eager_aa_values,
            "summary": summarize(eager_aa_values),
        },
    }
    eager_p50 = arms["eager"]["summary"]["p50_us"]
    graphed_p50 = arms["graphed"]["summary"]["p50_us"]
    aa_p50 = arms["eager_aa"]["summary"]["p50_us"]
    return {
        "schema_version": 1,
        "metadata": metadata,
        "config": config,
        "correctness": correctness,
        "graph_hits": graph_hits,
        "graph_attempts": graph_attempts,
        "graph_hit_rate": graph_hits / graph_attempts,
        "arms": arms,
        "speedup": eager_p50 / graphed_p50,
        "aa_noise_fraction": abs(aa_p50 - eager_p50) / eager_p50,
    }


def write_reports(report: dict[str, Any], prefix: Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("arm", "sample", "latency_us"))
        writer.writeheader()
        for arm, values in report["arms"].items():
            for index, latency in enumerate(values["samples_us"]):
                writer.writerow(
                    {"arm": arm, "sample": index, "latency_us": latency}
                )
    return json_path, csv_path


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _time_cuda_calls(torch: Any, function: Callable[[], Any], count: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _parse_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample rows must be comma-separated integers") from exc
    if not rows or rows[0] < 0:
        raise argparse.ArgumentTypeError("sample rows must be non-negative and non-empty")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-rows", type=_parse_rows, default=(0, 2))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if min(args.batch_size, args.warmup, args.rounds, args.iterations) < 1:
        raise ValueError("batch size and iteration arguments must be positive")
    if args.sample_rows[-1] >= args.batch_size:
        raise ValueError("sample row exceeds batch size")
    if len(args.sample_rows) == args.batch_size:
        raise ValueError("benchmark requires a mixed sampled/argmax batch")

    import torch

    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tests.unit_test.qwen3_tts.test_predictor_cuda_graph import (
        _build_talker,
        _request,
        _step_inputs,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    talker = _build_talker(device)
    sampled_rows = set(args.sample_rows)
    requests = [
        _request(
            dosample=row in sampled_rows,
            temperature=0.8 + row * 0.01,
            top_p=0.9 if row in sampled_rows else 0.0,
            top_k=5 if row in sampled_rows else 0,
            sub_seed=1000 + row,
            semantic_seed=2000 + row,
        )
        for row in range(args.batch_size)
    ]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(args.batch_size, device)

    def eager():
        return talker._code_predictor_forward_incremental(
            layer0, hidden, semantic_positions=positions
        )

    def graphed():
        return talker.code_predictor_forward(
            layer0, hidden, semantic_positions=positions
        )

    for _ in range(args.warmup):
        eager()
        graphed()
    torch.cuda.synchronize()

    eager_codes, eager_embeddings = (value.clone() for value in eager())
    graph_codes, graph_embeddings = (value.clone() for value in graphed())
    torch.cuda.synchronize()
    correctness = {
        "codes_exact": bool(torch.equal(eager_codes, graph_codes)),
        "embeddings_exact": bool(torch.equal(eager_embeddings, graph_embeddings)),
    }

    samples: dict[str, list[float]] = {"eager": [], "graphed": []}
    graph_hits = 0
    graph_attempts = 0
    functions = {"eager": eager, "graphed": graphed}
    for first, second in interleaved_modes(args.rounds):
        for arm in (first, second):
            samples[arm].extend(
                _time_cuda_calls(torch, functions[arm], args.iterations)
            )
            if arm == "graphed":
                graph_attempts += args.iterations
                if talker._predictor_graphs:
                    graph_hits += args.iterations
    eager_aa_samples = _time_cuda_calls(torch, eager, args.iterations * 2)

    device_index = torch.cuda.current_device()
    driver = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(device_index),
        "capability": list(torch.cuda.get_device_capability(device_index)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": driver.splitlines()[0],
        "git_sha": _command_output(["git", "rev-parse", "HEAD"]),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device_index),
    }
    config = {
        "batch_size": args.batch_size,
        "sample_rows": list(args.sample_rows),
        "warmup": args.warmup,
        "rounds": args.rounds,
        "iterations": args.iterations,
    }
    report = build_report(
        metadata=metadata,
        config=config,
        eager_samples=samples["eager"],
        graphed_samples=samples["graphed"],
        eager_aa_samples=eager_aa_samples,
        graph_hits=graph_hits,
        graph_attempts=graph_attempts,
        correctness=correctness,
    )
    json_path, csv_path = write_reports(report, args.output_prefix)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    if not all(correctness.values()) or report["graph_hit_rate"] != 1.0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
