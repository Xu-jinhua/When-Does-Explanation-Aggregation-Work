from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any

import pytest
import yaml

from xai_ensemble.core.hashing import object_sha256
from xai_ensemble.core.io import atomic_write_json
from xai_ensemble.simple.adversarial import (
    _existing_shards,
    adversarial_artifact_digest,
    adversarial_source_identity,
    completed_adversarial_manifest,
    load_adversarial_shard,
    materialize_adversarial_model_inputs,
    run_adversarial_task,
    write_adversarial_shard,
)
from xai_ensemble.simple.artifacts import (
    ADVERSARIAL_SCHEMA_VERSION,
    ArtifactStore,
    adversarial_artifact_root,
)
from xai_ensemble.simple.config import load_experiment
from xai_ensemble.simple.data import DatasetBundle, LoadedModel
from xai_ensemble.simple.scheduler import (
    ADVERSARIAL_RESERVATION_BYTES,
    COMPUTE_EXCLUSIVE_JOB_KINDS,
    SimpleJobStore,
    submit_plan,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
ADVERSARIAL_CONDITION = {
    "id": "adversarial-sara-2-255",
    "kind": "adversarial",
    "kwargs": {
        "algorithm": "sara-repetto-v2",
        "epsilon": 2.0 / 255.0,
        "steps": 100,
        "learning_rate": 0.1,
        "classification_weight": 1e-4,
        "top_fraction": 0.1,
        "source_by_architecture": {
            "cnn": "DeepLift",
            "vit": "TransformerAttribution",
        },
        "batch_size_by_architecture": {"cnn": 32, "vit": 16},
    },
}


def _runtime_config(tmp_path: Path) -> dict:
    value = yaml.safe_load((CODE_ROOT / "configs/simple/example.yaml").read_text(encoding="utf-8"))
    value["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    value["storage"].update(
        {
            "remote_root": str(tmp_path / "artifacts"),
            "scratch_root": str(tmp_path / "scratch"),
            "spool_root": str(tmp_path / "spool"),
            "spool_max_gib": 1,
            "spool_min_free_gib": 0,
        }
    )
    value["runtime"].update(
        {
            "profile_directory": str(tmp_path / "profiles"),
            "database_path": str(tmp_path / "jobs.sqlite3"),
            "log_directory": str(tmp_path / "logs"),
        }
    )
    return value


def _write_config(path: Path, value: dict):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return load_experiment(path)


def test_formal_condition_adds_only_new_task_ids_and_keeps_scheduler_namespace(
    tmp_path: Path,
) -> None:
    expanded = load_experiment(CODE_ROOT / "configs/simple/paper-main.yaml")
    baseline_value = yaml.safe_load(
        (CODE_ROOT / "configs/simple/paper-main.yaml").read_text(encoding="utf-8")
    )
    baseline_value["methods_file"] = str(CODE_ROOT / "configs/simple/methods.yaml")
    baseline_value["conditions"] = [
        value for value in baseline_value["conditions"] if value["kind"] != "adversarial"
    ]
    baseline = _write_config(tmp_path / "paper-main-without-adversarial.yaml", baseline_value)

    assert expanded.scheduler_digest == baseline.scheduler_digest
    assert expanded.scheduler_digest == (
        "9a47f17a4f0a0e6f44094fe8fb976606e9c9bb47feb87040ae5f6a048768abb4"
    )
    assert expanded.phase1_digest == baseline.phase1_digest
    assert len(expanded.adversarial_tasks()) == 4
    assert len(expanded.phase1_tasks()) == 220
    assert len(expanded.phase2_tasks()) == 60
    assert {task.task_id for task in baseline.phase1_tasks()} == {
        task.task_id for task in expanded.phase1_tasks() if task.condition.kind != "adversarial"
    }
    assert {task.task_id for task in baseline.phase2_tasks()} == {
        task.task_id for task in expanded.phase2_tasks() if task.condition.kind != "adversarial"
    }


def test_existing_sqlite_plan_extends_with_attacks_without_changing_old_jobs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment.yaml"
    baseline_value = _runtime_config(tmp_path)
    baseline = _write_config(path, baseline_value)
    store = SimpleJobStore(
        baseline.runtime.database_path,
        experiment_digest=baseline.scheduler_digest,
    )
    assert submit_plan(baseline, store, include_phase2=True) == {
        "profile": 30,
        "phase2_profile": 2,
        "phase1": 22,
        "phase2": 2,
    }
    old_jobs = {
        job.job_id: (job.kind, job.command, job.dependencies, job.resource_ids)
        for job in store.jobs()
    }

    expanded_value = _runtime_config(tmp_path)
    expanded_value["conditions"].append(ADVERSARIAL_CONDITION)
    expanded = _write_config(path, expanded_value)
    assert expanded.scheduler_digest == baseline.scheduler_digest
    expanded_store = SimpleJobStore(
        expanded.runtime.database_path,
        experiment_digest=expanded.scheduler_digest,
    )
    counts = submit_plan(expanded, expanded_store, include_phase2=True)

    assert counts == {
        "profile": 30,
        "phase2_profile": 2,
        "phase1": 44,
        "phase2": 4,
        "adversarial": 2,
    }
    by_id = {job.job_id: job for job in expanded_store.jobs()}
    assert len(by_id) == len(old_jobs) + 26
    for job_id, specification in old_jobs.items():
        job = by_id[job_id]
        assert (job.kind, job.command, job.dependencies, job.resource_ids) == specification
    attack_jobs = [job for job in by_id.values() if job.kind == "adversarial"]
    assert len(attack_jobs) == 2
    assert "adversarial" in COMPUTE_EXCLUSIVE_JOB_KINDS
    assert {job.reservation_bytes for job in attack_jobs} == set(
        ADVERSARIAL_RESERVATION_BYTES.values()
    )
    for phase_task in expanded.phase1_tasks():
        if phase_task.condition.kind != "adversarial":
            continue
        attack_task = expanded.adversarial_task_for(
            dataset_id=phase_task.dataset.dataset_id,
            model_id=phase_task.model.model_id,
            split=phase_task.split,
            condition_id=phase_task.condition.condition_id,
        )
        assert (
            f"adversarial:{attack_task.task_id}"
            in by_id[f"phase1:{phase_task.task_id}"].dependencies
        )


def _published_attack(tmp_path: Path):
    torch = pytest.importorskip("torch")
    value = _runtime_config(tmp_path)
    value["conditions"].append(ADVERSARIAL_CONDITION)
    value["datasets"] = value["datasets"][:1]
    value["models"] = value["models"][:1]
    dataset_manifest = tmp_path / "dataset-manifest.json"
    checkpoint = tmp_path / "checkpoint.pt"
    dataset_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint-fixture")
    value["datasets"][0]["manifest_path"] = str(dataset_manifest)
    value["models"][0]["checkpoint_path"] = str(checkpoint)
    experiment = _write_config(tmp_path / "artifact-experiment.yaml", value)
    task = experiment.adversarial_tasks()[0]
    store = ArtifactStore(experiment)
    root = adversarial_artifact_root(task)
    clean = torch.full((2, 3, 4, 4), 0.5, dtype=torch.float32)
    deltas = torch.stack((torch.full_like(clean[0], 0.005), torch.full_like(clean[0], -0.005)))
    adversarial = clean + deltas
    indices = torch.tensor([10, 20], dtype=torch.int64)
    labels = torch.tensor([1, 0], dtype=torch.int64)
    targets = torch.tensor([1, 0], dtype=torch.int64)
    clean_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
    adversarial_logits = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32)
    tensors = {
        "indices": indices,
        "labels": labels,
        "targets": targets,
        "clean_logits": clean_logits,
        "adversarial_logits": adversarial_logits,
        "adversarial_images": adversarial,
        "deltas": deltas,
        "best_steps": torch.tensor([4, 8], dtype=torch.int64),
    }
    local_payload = tmp_path / "attack.safetensors"
    write_adversarial_shard(local_payload, task=task, tensors=tensors)
    payload_path, record_path = (
        "shards/shard-00000.safetensors",
        "shards/shard-00000.json",
    )
    published = store.publish(local_payload, posixpath.join(root, payload_path))
    source_identity = adversarial_source_identity(task)
    source_identity_digest = object_sha256(source_identity)
    record = {
        "schema_version": ADVERSARIAL_SCHEMA_VERSION,
        "task_digest": task.digest,
        "source_identity_digest": source_identity_digest,
        "shard_index": 0,
        "start": 0,
        "stop": 2,
        "count": 2,
        "row_indices_digest": object_sha256([10, 20]),
        "image_shape": [2, 3, 4, 4],
        "image_dtype": "float32",
        "delta_linf_max": 0.005,
        "prediction_preserved_count": 2,
        "unchanged_candidate_count": 0,
        "payload": {
            "relative_path": published.relative_path,
            "sha256": published.sha256,
            "size_bytes": published.size_bytes,
        },
    }
    local_record = tmp_path / "attack-record.json"
    atomic_write_json(local_record, record)
    store.publish(local_record, posixpath.join(root, record_path))
    manifest = {
        "schema_version": ADVERSARIAL_SCHEMA_VERSION,
        "status": "complete",
        "task_id": task.task_id,
        "task_digest": task.digest,
        "dataset": task.dataset.dataset_id,
        "model": task.model.model_id,
        "split": task.split,
        "condition": task.condition.condition_id,
        "algorithm": task.algorithm,
        "source_method": task.source_method,
        "precision": "fp32",
        "source_identity": dict(source_identity),
        "sample_count": 2,
        "shard_size": 512,
        "shards": [record],
    }
    manifest["artifact_digest"] = adversarial_artifact_digest(
        task, source_identity, manifest["shards"]
    )
    local_manifest = tmp_path / "attack-manifest.json"
    atomic_write_json(local_manifest, manifest)
    store.publish(local_manifest, posixpath.join(root, "manifest.json"), write_receipt=False)
    return experiment, task, store, tensors, clean, manifest


def test_completed_manifest_and_verified_partial_shard_are_skipped(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    experiment, task, store, _, _, manifest = _published_attack(tmp_path)
    source_digest = object_sha256(adversarial_source_identity(task))

    assert completed_adversarial_manifest(experiment, task, store=store) == manifest
    assert _existing_shards(
        store,
        adversarial_artifact_root(task),
        task=task,
        source_identity_digest=source_digest,
        bounds=((0, 2),),
        indices=torch.tensor([10, 20]),
    ) == {0: manifest["shards"][0]}
    # Completion is checked before CUDA/device setup, so this cannot regenerate.
    assert run_adversarial_task(experiment, task, device="cpu") == manifest


@pytest.mark.parametrize("streaming", [False, True])
def test_phase1_and_phase2_reuse_the_exact_saved_adversarial_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    torch = pytest.importorskip("torch")
    if streaming:
        monkeypatch.setenv("XAI_SIMPLE_STREAMING_INPUTS", "1")
        monkeypatch.setenv("XAI_SIMPLE_HOT_CACHE_GIB", "0.001")
        monkeypatch.setenv("XAI_SIMPLE_HOT_CACHE_ROOT", str(tmp_path / "hot"))
    experiment, task, store, tensors, clean, manifest = _published_attack(tmp_path)
    phase1_task = next(
        value
        for value in experiment.phase1_tasks()
        if value.condition.kind == "adversarial" and value.family == "Saliency"
    )
    bundle = DatasetBundle(
        dataset=None,
        raw_images=clean,
        labels=tensors["labels"],
        indices=tensors["indices"],
        sample_ids=("sample-10", "sample-20"),
        row_to_position={10: 0, 20: 1},
        raw_source_identity_digest="fixture-adversarial-clean-source",
    )
    loaded_model = LoadedModel(
        model=None,
        preprocessing={"input_size": 4},
        normalize=lambda images: images,
    )

    phase1_values = materialize_adversarial_model_inputs(
        experiment,
        phase1_task,
        bundle,
        loaded_model,
        store=store,
    )
    phase2_values = load_adversarial_shard(
        experiment,
        task,
        0,
        expected_indices=tensors["indices"],
        expected_labels=tensors["labels"],
        clean_images=clean,
        expected_targets=tensors["targets"],
        expected_adversarial_logits=tensors["adversarial_logits"],
        store=store,
        manifest=manifest,
    )

    torch.testing.assert_close(
        phase1_values.model_inputs[0 : len(bundle)],
        phase2_values["adversarial_images"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        phase2_values["adversarial_images"] - phase2_values["deltas"],
        clean,
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(phase1_values.targets, tensors["targets"])
    torch.testing.assert_close(
        phase1_values.adversarial_logits,
        tensors["adversarial_logits"],
    )
    assert torch.equal(phase2_values["clean_logits"].argmax(1), tensors["targets"])
    assert torch.equal(phase2_values["adversarial_logits"].argmax(1), tensors["targets"])



def test_streaming_adversarial_model_input_batches_do_not_pin_full_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrow prefetched batch must not retain the whole normalized shard.

    Regression test for the host-memory blow-up (OOM-killed, exit -9) seen on
    adversarial IntegratedGradients jobs: ``model_inputs`` used to normalize
    the full shard and return a view into it, so every batch queued by the
    byte-bounded prefetcher pinned an entire shard while being accounted as
    batch-sized bytes.
    """

    torch = pytest.importorskip("torch")
    monkeypatch.setenv("XAI_SIMPLE_STREAMING_INPUTS", "1")
    monkeypatch.setenv("XAI_SIMPLE_HOT_CACHE_GIB", "0.001")
    monkeypatch.setenv("XAI_SIMPLE_HOT_CACHE_ROOT", str(tmp_path / "hot"))
    experiment, task, store, tensors, clean, manifest = _published_attack(tmp_path)
    phase1_task = next(
        value
        for value in experiment.phase1_tasks()
        if value.condition.kind == "adversarial" and value.family == "Saliency"
    )
    bundle = DatasetBundle(
        dataset=None,
        raw_images=clean,
        labels=tensors["labels"],
        indices=tensors["indices"],
        sample_ids=("sample-10", "sample-20"),
        row_to_position={10: 0, 20: 1},
        raw_source_identity_digest="fixture-adversarial-clean-source",
    )

    def allocating_normalize(images: Any) -> Any:
        return (images - 0.5) / 0.25

    loaded_model = LoadedModel(
        model=None,
        preprocessing={"input_size": 4},
        normalize=allocating_normalize,
    )

    phase1_values = materialize_adversarial_model_inputs(
        experiment,
        phase1_task,
        bundle,
        loaded_model,
        store=store,
    )

    single = phase1_values.model_inputs[0:1]
    assert single.untyped_storage().nbytes() == single.numel() * single.element_size()
    torch.testing.assert_close(
        single,
        allocating_normalize(tensors["adversarial_images"][0:1]),
        rtol=0.0,
        atol=0.0,
    )
