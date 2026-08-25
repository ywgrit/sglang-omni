# SPDX-License-Identifier: Apache-2.0
"""Import-light contracts for the mixed Predictor CUDA Graph benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


BENCHMARK = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "qwen3_tts"
    / "bench_predictor_mixed_graph.py"
)


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("bench_predictor_mixed_graph", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_interleaved_modes_crosses_order():
    module = _load_benchmark()
    assert module.interleaved_modes(4) == [
        ("eager", "graphed"),
        ("graphed", "eager"),
        ("eager", "graphed"),
        ("graphed", "eager"),
    ]


def test_report_records_hit_rate_noise_and_speedup():
    module = _load_benchmark()
    report = module.build_report(
        metadata={"gpu": "test"},
        config={"batch_size": 4},
        eager_samples=[10.0, 12.0, 11.0],
        graphed_samples=[5.0, 6.0, 5.5],
        eager_aa_samples=[10.5, 11.0, 11.5],
        graph_hits=3,
        graph_attempts=4,
        correctness={"codes_exact": True, "embeddings_exact": True},
    )

    assert report["graph_hit_rate"] == 0.75
    assert report["speedup"] == 2.0
    assert report["aa_noise_fraction"] == 0.0
    assert report["arms"]["eager"]["summary"]["p50_us"] == 11.0
    assert report["arms"]["graphed"]["summary"]["p90_us"] == 6.0


def test_write_reports_preserves_raw_samples(tmp_path: Path):
    module = _load_benchmark()
    report = module.build_report(
        metadata={},
        config={},
        eager_samples=[9.0, 10.0],
        graphed_samples=[4.0, 5.0],
        eager_aa_samples=[9.5, 10.5],
        graph_hits=2,
        graph_attempts=2,
        correctness={"codes_exact": True, "embeddings_exact": True},
    )
    json_path, csv_path = module.write_reports(report, tmp_path / "result")

    assert json.loads(json_path.read_text(encoding="utf-8"))["arms"]["eager"][
        "samples_us"
    ] == [9.0, 10.0]
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "arm,sample,latency_us"
    assert len(rows) == 7
