from __future__ import annotations

from pathlib import Path

from xai_ensemble.core.paths import (
    FULL_MATRIX_ASSET_OVERLAY_ROOT_ENV,
    FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV,
    FULL_MATRIX_LOGICAL_ASSET_ROOT_ENV,
    FULL_MATRIX_LOGICAL_CACHE_ROOT_ENV,
    resolve_full_matrix_runtime_path,
    resolve_pilot_definition_dir,
)


def test_absolute_pilot_directory_is_authoritative(tmp_path) -> None:
    value = tmp_path / "absolute-pilots"
    assert resolve_pilot_definition_dir(value) == value.resolve()


def test_existing_worker_cwd_directory_wins(tmp_path, monkeypatch) -> None:
    expected = tmp_path / "configs" / "pilots"
    expected.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    assert resolve_pilot_definition_dir("configs/pilots") == expected.resolve()


def test_project_root_can_supply_repository_code_directory(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    expected = project_root / "code" / "configs" / "pilots"
    expected.mkdir(parents=True)
    worker = tmp_path / "worker"
    worker.mkdir()
    monkeypatch.chdir(worker)

    assert (
        resolve_pilot_definition_dir("configs/pilots", project_root=project_root)
        == expected.resolve()
    )


def test_package_source_directory_is_the_canonical_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = Path(__file__).resolve().parents[1] / "configs" / "pilots"

    assert resolve_pilot_definition_dir("configs/pilots") == expected.resolve()


def test_full_matrix_overlay_maps_registered_local_assets(tmp_path, monkeypatch) -> None:
    logical_assets = tmp_path / "logical-assets"
    overlay_assets = tmp_path / "overlay-assets"
    logical_cache = tmp_path / "logical-cache"
    overlay_cache = tmp_path / "overlay-cache"
    monkeypatch.setenv(FULL_MATRIX_LOGICAL_ASSET_ROOT_ENV, str(logical_assets))
    monkeypatch.setenv(FULL_MATRIX_ASSET_OVERLAY_ROOT_ENV, str(overlay_assets))
    monkeypatch.setenv(FULL_MATRIX_LOGICAL_CACHE_ROOT_ENV, str(logical_cache))
    monkeypatch.setenv(FULL_MATRIX_CACHE_OVERLAY_ROOT_ENV, str(overlay_cache))

    target_manifest = logical_assets / "datasets" / "octmnist" / "manifest.json"
    target_cache = logical_cache / "octmnist" / "octmnist_224.npz"
    unrelated_manifest = logical_assets / "datasets" / "food101" / "manifest.json"

    assert (
        resolve_full_matrix_runtime_path(target_manifest)
        == (overlay_assets / "datasets" / "octmnist" / "manifest.json").resolve()
    )
    assert (
        resolve_full_matrix_runtime_path(target_cache)
        == (overlay_cache / "octmnist" / "octmnist_224.npz").resolve()
    )
    assert (
        resolve_full_matrix_runtime_path(unrelated_manifest)
        == (overlay_assets / "datasets" / "food101" / "manifest.json").resolve()
    )
