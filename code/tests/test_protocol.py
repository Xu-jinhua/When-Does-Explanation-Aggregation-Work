from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xai_ensemble.core.protocol import ProtocolError, load_protocol


def _base() -> dict:
    return {
        "run": {"id": "test"},
        "datasets": {},
        "models": {},
        "explainers": {},
        "evaluation": {"patch_size": 16, "top_k": 20, "fill": "dataset_mean"},
        "targets": {"ind": "reference_prediction"},
        "ranking": {"base": 1},
        "storage": {"provider": "local", "private": True},
    }


def test_protocol_digest_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(_base()), encoding="utf-8")
    first = load_protocol(path)
    second = load_protocol(path)
    assert first.digest == second.digest
    assert first.run_id == "test"


def test_protocol_rejects_wrong_primary_patch_size(tmp_path: Path) -> None:
    value = _base()
    value["evaluation"]["patch_size"] = 14
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ProtocolError, match="patch_size=16"):
        load_protocol(path)


def test_protocol_rejects_unknown_storage_provider(tmp_path: Path) -> None:
    value = _base()
    value["storage"] = {"provider": "modelscope", "private": True}
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ProtocolError, match="local or cloudstorage"):
        load_protocol(path)


def test_cloudstorage_protocol_requires_root_revision_and_local_lock_root(tmp_path: Path) -> None:
    value = _base()
    value["storage"] = {"provider": "cloudstorage", "private": True}
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ProtocolError, match="storage.root"):
        load_protocol(path)

    value["storage"] = {
        "provider": "cloudstorage",
        "private": True,
        "root": str(tmp_path / "cloud"),
        "revision": "core-v2",
        "lock_root": str(tmp_path / "locks"),
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    assert load_protocol(path).run_id == "test"


def test_protocol_snapshot_is_loadable_and_digest_bound(tmp_path: Path) -> None:
    value = _base()
    yaml_path = tmp_path / "protocol.yaml"
    yaml_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    protocol = load_protocol(yaml_path)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": protocol.run_id,
                "protocol_digest": protocol.digest,
                "protocol": value,
            }
        ),
        encoding="utf-8",
    )
    assert load_protocol(snapshot).digest == protocol.digest
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["protocol"]["run"]["id"] = "tampered"
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="snapshot digest"):
        load_protocol(snapshot)
