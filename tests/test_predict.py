"""Tests for run persistence, resumption, and NMS post-processing."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box


@pytest.fixture(scope="module")
def predict_module():
    """Import the prediction module for either supported package layout."""
    for module_name in ("src.inference.predict", "inference.predict"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            package_root = module_name.split(".", maxsplit=1)[0]
            if error.name != package_root:
                raise

    pytest.fail("Could not import the prediction module")


@pytest.fixture
def empty_predictions() -> gpd.GeoDataFrame:
    """Return an empty raw-prediction GeoDataFrame with the expected schema."""
    return gpd.GeoDataFrame(
        {
            "confidence": [],
            "crop_id": [],
            "tile": [],
        },
        geometry=gpd.GeoSeries([], crs="EPSG:2154"),
        crs="EPSG:2154",
    )


def test_predict_and_save_tile_writes_final_empty_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predict_module,
    empty_predictions: gpd.GeoDataFrame,
) -> None:
    """An empty tile result is still saved as a completed raw output."""
    raw_predictions_dir = tmp_path / "raw_predictions"
    raw_predictions_dir.mkdir()

    monkeypatch.setattr(
        predict_module,
        "predict_tile_obb",
        lambda **_: (empty_predictions, 1),
    )

    predict_module.predict_and_save_tile(
        tile=tmp_path / "tile_a.jp2",
        raw_preds_dir=raw_predictions_dir,
        path_save_jpg=tmp_path / "jpg",
        model=object(),
        window_size=2048,
        window_overlap=0.2,
        predict_args={},
        verbose=False,
    )

    output_path = raw_predictions_dir / "tile_a.parquet"
    assert output_path.is_file()
    assert not (raw_predictions_dir / "tile_a.parquet.part").exists()
    assert gpd.read_parquet(output_path).empty


def test_run_raw_predictions_processes_only_direct_jp2_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predict_module,
) -> None:
    """The initial run ignores JP2 files located in nested directories."""
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    (input_dir / "direct.jp2").touch()
    nested_dir = input_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested.jp2").touch()

    processed_tiles = []
    monkeypatch.setattr(
        predict_module,
        "predict_and_save_tile",
        lambda **kwargs: processed_tiles.append(kwargs["tile"].stem),
    )

    model = SimpleNamespace(model_name=str(tmp_path / "model.pt"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    predict_module.run_raw_predictions(
        input_path=input_dir,
        run_dir=run_dir,
        model=model,
        window_size=2048,
        window_overlap=0.2,
        path_save_jpg=tmp_path / "jpg",
    )

    assert processed_tiles == ["direct"]
    metadata = json.loads((run_dir / "metadata_prediction.json").read_text())
    assert metadata["input_path"] == str(input_dir.resolve())
    assert (run_dir / "raw_predictions").is_dir()


def test_resume_retries_partial_and_missing_tiles_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predict_module,
    empty_predictions: gpd.GeoDataFrame,
) -> None:
    """A finalized Parquet is skipped while partial and missing tiles rerun."""
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    for tile_name in ("done", "partial", "missing"):
        (input_dir / f"{tile_name}.jp2").touch()

    run_dir = tmp_path / "run"
    raw_predictions_dir = run_dir / "raw_predictions"
    raw_predictions_dir.mkdir(parents=True)
    empty_predictions.to_parquet(raw_predictions_dir / "done.parquet")
    (raw_predictions_dir / "partial.parquet.part").touch()

    metadata = {
        "input_path": str(input_dir),
        "path_save_jpg": str(tmp_path / "jpg"),
        "window_size": 2048,
        "window_overlap": 0.2,
        "predict_args": {"conf": 0.25},
        "verbose": False,
        "model_path": str(tmp_path / "model.pt"),
    }
    (run_dir / "metadata_prediction.json").write_text(json.dumps(metadata))

    processed_tiles = []
    monkeypatch.setattr(predict_module, "YOLO", lambda _: object())
    monkeypatch.setattr(
        predict_module,
        "predict_and_save_tile",
        lambda **kwargs: processed_tiles.append(kwargs["tile"].stem),
    )

    assert predict_module.resume_run(run_dir) == 2
    assert processed_tiles == ["missing", "partial"]


def test_resume_returns_none_when_metadata_is_invalid(
    tmp_path: Path,
    predict_module,
) -> None:
    """Invalid metadata cancels the resume without attempting inference."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata_prediction.json").write_text("not valid JSON")

    assert predict_module.resume_run(run_dir) is None


def test_resume_recreates_missing_raw_predictions_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predict_module,
) -> None:
    """A run interrupted before directory creation can still be resumed."""
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    (input_dir / "tile_a.jp2").touch()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = {
        "input_path": str(input_dir),
        "path_save_jpg": str(tmp_path / "jpg"),
        "window_size": 2048,
        "window_overlap": 0.2,
        "predict_args": {"conf": 0.25},
        "verbose": False,
        "model_path": str(tmp_path / "model.pt"),
    }
    (run_dir / "metadata_prediction.json").write_text(json.dumps(metadata))

    monkeypatch.setattr(predict_module, "YOLO", lambda _: object())
    monkeypatch.setattr(predict_module, "predict_and_save_tile", lambda **_: None)

    assert predict_module.resume_run(run_dir) == 1
    assert (run_dir / "raw_predictions").is_dir()


def test_save_results_after_nms_returns_none_for_empty_predictions(
    tmp_path: Path,
    predict_module,
    empty_predictions: gpd.GeoDataFrame,
) -> None:
    """Post-processing skips NMS when all completed tiles are empty."""
    raw_predictions_dir = tmp_path / "raw_predictions"
    raw_predictions_dir.mkdir()
    empty_predictions.to_parquet(raw_predictions_dir / "tile_a.parquet")

    assert predict_module.save_results_after_nms(tmp_path, 0.1) is None


def test_save_results_after_nms_writes_final_geopackage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predict_module,
) -> None:
    """NMS can be rerun from raw outputs without loading a YOLO model."""
    raw_predictions_dir = tmp_path / "raw_predictions"
    raw_predictions_dir.mkdir()
    raw_predictions = gpd.GeoDataFrame(
        {
            "confidence": [0.9],
            "crop_id": ["tile_a_0"],
            "tile": ["tile_a.jp2"],
        },
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:2154",
    )
    raw_predictions.to_parquet(raw_predictions_dir / "tile_a.parquet")

    monkeypatch.setattr(
        predict_module,
        "spatial_nms",
        lambda gdf, _: gdf.copy(),
    )

    results_path = predict_module.save_results_after_nms(tmp_path, 0.1)

    assert results_path == tmp_path / "final" / "predictions_obb.gpkg"
    saved_predictions = gpd.read_file(results_path)
    assert {"confidence", "crop_id", "tile", "geometry"} <= set(saved_predictions.columns)
    metadata = json.loads((tmp_path / "metadata_nms.json").read_text())
    assert metadata["nms_threshold"] == 0.1
    assert metadata["gpkg"] == str(results_path.resolve())


def test_save_results_after_nms_requires_completed_outputs(
    tmp_path: Path,
    predict_module,
) -> None:
    """Post-processing fails clearly when no completed tile output exists."""
    with pytest.raises(ValueError, match="No completed raw prediction files found"):
        predict_module.save_results_after_nms(tmp_path, 0.1)
