# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contract tests for Predictor CUDA Graph dispatch."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def _load_method(name: str, namespace: dict[str, object] | None = None):
    """Compile one Talker method without importing the full CUDA runtime."""
    source = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni"
        / "models"
        / "qwen3_tts"
        / "sglang_model.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    execution_namespace = dict(namespace or {})
    exec(compile(module, str(source), "exec"), execution_namespace)
    return execution_namespace[name]


def test_mixed_sampled_argmax_rows_have_bounded_graph_signature():
    signature = _load_method("_predictor_graph_signature")
    talker = SimpleNamespace(
        _sub_has_sampled_rows=True,
        _sub_sample_count=2,
        _sub_sampled_max_top_k=8,
        _sub_sampled_has_top_p=True,
        _sub_sampled_has_unbounded_top_k=False,
    )
    positions = SimpleNamespace(is_cuda=True, ndim=1, shape=(4,))

    assert signature(talker, 4, positions) == ("mixed", 8, True, False)

    talker._sub_sample_count = 1
    sparse_signature = signature(talker, 4, positions)
    talker._sub_sample_count = 3
    dense_signature = signature(talker, 4, positions)
    assert sparse_signature == dense_signature


def test_graph_capture_mixed_sampling_uses_fixed_full_batch_shapes():
    sample = _load_method("_sample_subtalker_token", {"torch": torch})
    calls: list[tuple[tuple[int, ...], torch.Tensor, torch.Tensor, bool]] = []

    def _sample_seeded(
        logits,
        layer_idx,
        *,
        row_indices,
        semantic_positions,
        graph_safe_params=False,
    ):
        del layer_idx
        calls.append(
            (
                tuple(logits.shape),
                row_indices.clone(),
                semantic_positions.clone(),
                graph_safe_params,
            )
        )
        return row_indices + 10

    def _select_positions(positions, batch_size, device):
        del batch_size, device
        return positions.index_select(0, talker._sub_sample_row_indices_tensor[:2])

    talker = SimpleNamespace(
        _sub_batch_size=4,
        _sub_has_sampled_rows=True,
        _sub_sample_rows=[0, 2],
        _sub_sample_count=2,
        _sub_sample_row_indices_tensor=torch.tensor([0, 2, 0, 0]),
        _sub_all_row_indices_tensor=torch.arange(4),
        _sub_sample_mask_tensor=torch.tensor([True, False, True, False]),
        _sub_fixed_shape_mixed_sampling=True,
        _sample_subtalker_token_seeded=_sample_seeded,
        _select_semantic_positions=_select_positions,
    )
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0],
            [0.0, 1.0, 9.0],
            [9.0, 1.0, 0.0],
            [0.0, 9.0, 1.0],
        ]
    )
    positions = torch.tensor([20, 21, 22, 23])

    tokens = sample(talker, logits, 0, semantic_positions=positions)

    assert torch.equal(tokens, torch.tensor([10, 2, 12, 1]))
    assert len(calls) == 1
    assert calls[0][0] == (4, 3)
    assert torch.equal(calls[0][1], torch.arange(4))
    assert torch.equal(calls[0][2], positions)
    assert calls[0][3] is True


def test_prepare_mixed_batch_stages_safe_params_for_argmax_rows():
    prepare = _load_method(
        "prepare_decode_buffers",
        {
            "Any": Any,
            "torch": torch,
            "_quantize_predictor_top_k": lambda max_top_k, vocab_size: (
                8 if max_top_k <= 8 < vocab_size else None
            ),
        },
    )
    talker = SimpleNamespace(
        config=SimpleNamespace(code_predictor_config=SimpleNamespace(vocab_size=16)),
        _sub_temperature_tensor=torch.empty(4),
        _sub_top_p_tensor=torch.empty(4),
        _sub_top_k_tensor=torch.empty(4, dtype=torch.long),
        _semantic_sampling_seed_tensor=torch.empty(4, dtype=torch.long),
        _sub_sampling_seed_tensor=torch.empty(4, dtype=torch.long),
        _sub_graph_temperature_tensor=torch.empty(4),
        _sub_graph_top_p_tensor=torch.empty(4),
        _sub_graph_top_k_tensor=torch.empty(4, dtype=torch.long),
        _sub_graph_sampling_seed_tensor=torch.empty(4, dtype=torch.long),
        _sub_sample_row_indices_tensor=torch.empty(4, dtype=torch.long),
        _sub_sample_mask_tensor=torch.empty(4, dtype=torch.bool),
        _decode_prep_rids=None,
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                semantic_sampling_seed=10,
                subtalker_dosample=True,
                subtalker_temperature=0.7,
                subtalker_top_p=0.8,
                subtalker_top_k=5,
                subtalker_sampling_seed=100,
            )
        ),
        SimpleNamespace(
            data=SimpleNamespace(
                semantic_sampling_seed=11,
                subtalker_dosample=False,
                subtalker_temperature=0.0,
                subtalker_top_p=0.0,
                subtalker_top_k=0,
                subtalker_sampling_seed=101,
            )
        ),
    ]

    prepare(talker, requests)

    assert torch.equal(talker._sub_sample_mask_tensor, torch.tensor([1, 0, 0, 0]))
    assert talker._sub_temperature_tensor[1].item() == 0.0
    assert talker._sub_top_p_tensor[1].item() == 0.0
    assert talker._sub_top_k_tensor[1].item() == 0
    assert talker._sub_sampling_seed_tensor[1].item() == 101
    assert talker._sub_graph_temperature_tensor[1].item() == 1.0
    assert talker._sub_graph_top_p_tensor[1].item() == 1.0
    assert talker._sub_graph_top_k_tensor[1].item() == 8
    assert talker._sub_graph_sampling_seed_tensor[1].item() == 0
    assert torch.equal(talker._sub_graph_temperature_tensor[2:], torch.ones(2))
    assert torch.equal(talker._sub_graph_top_p_tensor[2:], torch.ones(2))
    assert torch.equal(talker._sub_graph_top_k_tensor[2:], torch.full((2,), 8))
    assert torch.equal(talker._sub_graph_sampling_seed_tensor[2:], torch.zeros(2))


def test_mixed_capture_state_uses_fixed_shape_without_losing_live_rows():
    capture_state = contextmanager(
        _load_method("_predictor_graph_capture_state", {"torch": torch})
    )
    talker = SimpleNamespace(
        _sub_batch_size=3,
        _sub_sample_count=1,
        _sub_sample_rows=[1],
        _sub_has_sampled_rows=True,
        _sub_fixed_shape_mixed_sampling=False,
        _sub_sampled_has_top_p=True,
        _sub_sampled_max_top_k=8,
        _sub_sampled_has_unbounded_top_k=False,
        _sub_sample_row_indices_tensor=torch.tensor([1, 0, 0, 0]),
        _decode_prep_rids=[("request", 1)],
    )

    with capture_state(talker, 4, ("mixed", 8, True, False)):
        assert talker._sub_batch_size == 4
        assert talker._sub_sample_count == 1
        assert talker._sub_sample_rows == [1]
        assert talker._sub_has_sampled_rows is True
        assert talker._sub_fixed_shape_mixed_sampling is True

    assert talker._sub_batch_size == 3
    assert talker._sub_sample_count == 1
    assert talker._sub_sample_rows == [1]
    assert talker._sub_fixed_shape_mixed_sampling is False
    assert talker._decode_prep_rids is None
