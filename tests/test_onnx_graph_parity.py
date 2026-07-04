from __future__ import annotations

import os
from pathlib import Path

import pytest

from chatterbox.onnx_export.cli import export
from chatterbox.onnx_export.config import ExportConfig
from chatterbox.onnx_export.graphs import ALL_GRAPHS
from chatterbox.onnx_export.validation.runner import \
    run_validation_for_precisions


@pytest.mark.parametrize("graph_name", [spec.name for spec in ALL_GRAPHS])
def test_graph_registered(graph_name):
    assert graph_name


@pytest.mark.skipif(
    not os.getenv("CHATTERBOX_TEST_CHECKPOINT_DIR"),
    reason="Set CHATTERBOX_TEST_CHECKPOINT_DIR to run ONNX parity tests.",
)
def test_all_onnx_graphs_export_and_validate(tmp_path):
    checkpoint_dir = Path(os.environ["CHATTERBOX_TEST_CHECKPOINT_DIR"])
    config = ExportConfig(
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "artifacts",
        precision="fp32",
        validate=False,
        device=os.getenv("CHATTERBOX_TEST_DEVICE", "cpu"),
    )
    export(config)
    report = run_validation_for_precisions(
        artifact_dir=config.output_dir,
        checkpoint_dir=checkpoint_dir,
        precisions=("fp32",),
        device=config.device,
    )
    assert set(report["fp32"]["graphs"]) == {spec.name for spec in ALL_GRAPHS}
