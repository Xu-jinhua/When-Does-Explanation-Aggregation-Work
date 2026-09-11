"""CLI registration and executable handlers for Phase 0."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from xai_ensemble.core.hashing import object_sha256, stable_seed
from xai_ensemble.core.paths import resolve_full_matrix_runtime_path
from xai_ensemble.core.provenance import collect_provenance
from xai_ensemble.data.manifest import build_dataset_manifest, read_manifest, write_manifest
from xai_ensemble.data.partitions import (
    make_ind_partitions,
    make_overlap_partitions,
    make_reference_partition,
    read_partition_plan,
    write_partition_plan,
)
from xai_ensemble.data.specs import IMAGENET1K, get_dataset_spec
from xai_ensemble.phase0.checkpoint import (
    CheckpointManager,
    make_checkpoint_metadata,
    save_inference_checkpoint,
)
from xai_ensemble.phase0.config import LoaderConfig, TrainingConfig, default_training_config
from xai_ensemble.phase0.dataset import (
    ManifestIndexedDataset,
    build_dataloader,
    build_image_transform,
    build_raw_image_transform,
    load_hf_split,
    records_for_source,
)
from xai_ensemble.phase0.models import (
    INITIALIZATION_RECIPES,
    ModelBuildRequest,
    create_model,
    get_model_definition,
    initialization_for_recipe,
    resolved_preprocessing,
    validate_imagenet1k_class_map,
)
from xai_ensemble.phase0.statistics import (
    ImageMeanAccumulator,
    write_image_mean_artifact,
)
from xai_ensemble.phase0.trainer import Trainer, close_distributed


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return result




def _csv_nonnegative_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("Values must be non-negative integers")
    return result


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _build_manifest(args: argparse.Namespace) -> int:
    spec = get_dataset_spec(args.dataset)
    cache_dir = None if args.cache_dir is None else resolve_full_matrix_runtime_path(args.cache_dir)
    output = resolve_full_matrix_runtime_path(args.output)
    manifest = build_dataset_manifest(
        spec,
        splits=args.splits,
        cache_dir=cache_dir,
        streaming=args.streaming,
        token=args.token,
        hash_mode=args.hash_mode,
        require_expected_counts=not args.allow_count_mismatch,
    )
    write_manifest(manifest, output)
    audit = manifest.audit()
    _emit(
        {
            "path": str(output.resolve()),
            "fingerprint": manifest.fingerprint,
            "records": len(manifest.records),
            "split_sizes": audit.split_sizes,
            "cross_split_duplicate_contents": len(audit.duplicate_contents_across_splits),
        }
    )
    return 0


def _build_partitions(args: argparse.Namespace) -> int:
    manifest_path = resolve_full_matrix_runtime_path(args.manifest)
    output = resolve_full_matrix_runtime_path(args.output)
    matched_ind_partition = (
        None
        if args.matched_ind_partition is None
        else resolve_full_matrix_runtime_path(args.matched_ind_partition)
    )
    manifest = read_manifest(manifest_path)
    if args.kind == "ind":
        plan = make_ind_partitions(
            manifest,
            split=args.split,
            num_sources=args.num_sources,
            seed=args.seed,
            require_full_coverage=not args.allow_remainder,
        )
    elif args.kind == "overlap":
        class_quotas = None
        matched_ind_digest = None
        if matched_ind_partition:
            if args.samples_per_class is not None:
                raise ValueError(
                    "--samples-per-class cannot be combined with --matched-ind-partition"
                )
            matched = read_partition_plan(matched_ind_partition)
            matched.validate(manifest)
            if (
                matched.kind != "ind"
                or matched.split != args.split
                or len(matched.sources) != args.num_sources
                or matched.seed != args.seed
            ):
                raise ValueError("Matched IND plan differs in kind/split/source-count/seed")
            observed_labels = sorted(
                {record.label for record in manifest.records_for_split(args.split)}
            )
            # Drop-remainder IND plans omit classes with a zero source quota.
            class_count_maps = [
                {label: source.class_counts.get(label, 0) for label in observed_labels}
                for source in matched.sources
            ]
            if any(counts != class_count_maps[0] for counts in class_count_maps[1:]):
                raise ValueError("IND source class quotas are not identical across sources")
            class_quotas = class_count_maps[0]
            matched_ind_digest = matched.digest
        plan = make_overlap_partitions(
            manifest,
            split=args.split,
            num_sources=args.num_sources,
            seed=args.seed,
            samples_per_class=args.samples_per_class,
            class_quotas=class_quotas,
            mode=args.overlap_mode,
            matched_ind_digest=matched_ind_digest,
        )
        if matched_ind_partition:
            assert class_quotas is not None
            if any(
                {label: source.class_counts.get(label, 0) for label in class_quotas} != class_quotas
                for source in plan.sources
            ):
                raise RuntimeError("Matched OVERLAP class counts differ from IND quotas")
    else:
        plan = make_reference_partition(manifest, split=args.split, seed=args.seed)
    write_partition_plan(plan, output)
    _emit(
        {
            "path": str(output.resolve()),
            "kind": plan.kind,
            "strategy": plan.strategy,
            "digest": plan.digest,
            "source_count": len(plan.sources),
            "source_sizes": {source.source_id: source.size for source in plan.sources},
            "overlap_matrix": plan.overlap_matrix(),
        }
    )
    return 0


def _load_training_config(args: argparse.Namespace, family: str) -> TrainingConfig:
    if args.training_config:
        import yaml

        with Path(args.training_config).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        config = TrainingConfig.from_mapping(values)
    else:
        config = default_training_config(family, epochs=args.epochs)

    optimizer = config.optimizer
    if args.learning_rate is not None:
        optimizer = replace(optimizer, learning_rate=args.learning_rate)
    train_loader = replace(
        config.train_loader,
        batch_size=args.batch_size or config.train_loader.batch_size,
        num_workers=(config.train_loader.num_workers if args.workers is None else args.workers),
    )
    validation_loader = replace(
        config.validation_loader,
        batch_size=args.validation_batch_size or config.validation_loader.batch_size,
        num_workers=(
            config.validation_loader.num_workers if args.workers is None else args.workers
        ),
    )
    warmup = min(config.warmup_epochs, max(0, args.epochs - 1))
    return replace(
        config,
        epochs=args.epochs,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        warmup_epochs=warmup,
        amp=args.precision,
        seed=args.seed,
    )


def _selected_source(plan: Any, source_id: str | None) -> Any:
    if source_id is None:
        if len(plan.sources) != 1:
            raise ValueError("--source-id is required for a multi-source partition plan")
        return plan.sources[0]
    return plan.source(source_id)


def _source_training_seed(
    base_seed: int,
    *,
    dataset: str,
    model: str,
    source_id: str,
) -> int:
    """Return the matched IND/OVERLAP RNG seed for one source identity."""

    return int(
        stable_seed(
            base_seed,
            dataset,
            model,
            source_id,
            modulus=2**31,
        )
    )


def _class_balance_parameters(
    records: Sequence[Any], *, num_classes: int
) -> tuple[list[int], list[float]]:
    """Return train-only counts and normalized inverse-frequency weights."""

    if num_classes <= 1:
        raise ValueError("class balancing requires at least two classes")
    counts = [0 for _ in range(num_classes)]
    for record in records:
        label = int(record.label)
        if not 0 <= label < num_classes:
            raise ValueError(f"training label {label} is outside [0,{num_classes})")
        counts[label] += 1
    if any(count <= 0 for count in counts):
        raise ValueError(f"class balancing requires every training class; counts={counts}")
    raw = [1.0 / count for count in counts]
    mean = sum(raw) / len(raw)
    return counts, [value / mean for value in raw]


def _resolve_training_recipe(
    args: argparse.Namespace,
    *,
    partition_kind: str,
    num_classes: int,
) -> tuple[str, str, tuple[int, ...] | None, str | None]:
    """Resolve and cross-check the executable recipe carried by one job."""

    recipe = args.recipe
    if recipe is None:
        legacy = {
            "random": "random_scratch",
            "imagenet1k": "timm_finetune",
            "imagenet1k_subset": "timm_subset_logits",
        }
        recipe = legacy[args.initialization]
    if recipe not in INITIALIZATION_RECIPES:
        raise ValueError(f"Unknown training recipe: {recipe}")
    initialization = initialization_for_recipe(recipe)
    if args.initialization != initialization:
        raise ValueError(
            f"recipe={recipe} requires --initialization {initialization}, not {args.initialization}"
        )

    indices = args.class_map_indices
    digest = args.class_map_sha256
    if recipe == "timm_subset_logits":
        if partition_kind != "reference":
            raise ValueError("timm_subset_logits is only valid for a reference model")
        if args.epochs != 0:
            raise ValueError("timm_subset_logits requires --epochs 0")
        if indices is None or digest is None:
            raise ValueError(
                "timm_subset_logits requires --class-map-indices and --class-map-sha256"
            )
        validate_imagenet1k_class_map(
            indices,
            expected_classes=num_classes,
            expected_digest=digest,
        )
    else:
        if args.epochs <= 0:
            raise ValueError(f"{recipe} requires a positive --epochs value")
        if indices is not None or digest is not None:
            raise ValueError(
                "class-map arguments are executable inputs only for timm_subset_logits"
            )

    if partition_kind in {"ind", "overlap"} and recipe != "random_scratch":
        raise ValueError(
            "Locked IND/matched-OVERLAP source models must use the random_scratch recipe"
        )
    return recipe, initialization, indices, digest


def _train(args: argparse.Namespace) -> int:
    args.manifest = str(resolve_full_matrix_runtime_path(args.manifest))
    args.train_partition = str(resolve_full_matrix_runtime_path(args.train_partition))
    if args.validation_partition is not None:
        args.validation_partition = str(resolve_full_matrix_runtime_path(args.validation_partition))
    if args.cache_dir is not None:
        args.cache_dir = str(resolve_full_matrix_runtime_path(args.cache_dir))
    args.output = str(resolve_full_matrix_runtime_path(args.output))
    spec = get_dataset_spec(args.dataset)
    manifest = read_manifest(args.manifest)
    manifest.validate(spec, reject_cross_split_duplicates=False)
    train_plan = read_partition_plan(args.train_partition)
    train_plan.validate(manifest)
    train_source = _selected_source(train_plan, args.source_id)
    validation_plan = (
        None
        if args.validation_partition is None
        else read_partition_plan(args.validation_partition)
    )
    if validation_plan is not None:
        validation_plan.validate(manifest)
        validation_source = _selected_source(validation_plan, train_source.source_id)
        validation_records = records_for_source(
            manifest,
            validation_source,
            split=validation_plan.split,
        )
        validation_split = validation_plan.split
    else:
        validation_split = args.validation_split
        validation_records = manifest.records_for_split(validation_split)

    definition = get_model_definition(args.model)
    recipe, initialization, class_index_map, class_map_sha256 = _resolve_training_recipe(
        args,
        partition_kind=train_plan.kind,
        num_classes=spec.num_classes,
    )
    # Corresponding IND and matched-OVERLAP sources deliberately share their
    # initialization/training RNG; source IDs remain distinct.  The only
    # primary-control difference is therefore the partition overlap.
    derived_seed = _source_training_seed(
        args.seed,
        dataset=spec.key,
        model=args.model,
        source_id=train_source.source_id,
    )
    args.seed = derived_seed
    role = "reference" if train_plan.kind == "reference" else "source"
    partition_digest = object_sha256(
        {
            "train": train_plan.digest,
            "validation": None if validation_plan is None else validation_plan.digest,
            "source_id": train_source.source_id,
        }
    )
    provenance = dict(collect_provenance(args.project_root))
    provenance["initialization_recipe"] = recipe
    if class_index_map is not None:
        provenance["class_map_indices"] = list(class_index_map)
        provenance["class_map_sha256"] = class_map_sha256

    if recipe == "timm_subset_logits":
        request = ModelBuildRequest(
            model_key=args.model,
            num_classes=spec.num_classes,
            init_mode=initialization,
            seed=derived_seed,
            class_index_map=class_index_map,
            class_map_sha256=class_map_sha256,
        )
        model = create_model(request)
        metadata = make_checkpoint_metadata(
            run_id=args.run_id,
            task_id=(args.task_id or f"{train_plan.kind}-{args.model}-{train_source.source_id}"),
            role=role,
            source_id=train_source.source_id,
            dataset_id=spec.dataset_id,
            dataset_revision=spec.revision,
            dataset_spec_fingerprint=spec.fingerprint,
            dataset_manifest_fingerprint=manifest.fingerprint,
            partition_kind=train_plan.kind,
            partition_digest=partition_digest,
            model_key=args.model,
            model_provider=definition.provider,
            initialization=initialization,
            num_classes=spec.num_classes,
            training_config={
                "recipe": recipe,
                "epochs": 0,
                "class_map_indices": list(class_index_map or ()),
                "class_map_sha256": class_map_sha256,
            },
            protocol_digest=args.protocol_digest,
            seed=derived_seed,
            provenance=provenance,
        ).with_progress(
            completed_epochs=0,
            global_step=0,
            best_validation_top1=None,
            metrics={},
        )
        manager = CheckpointManager(args.output)
        output = save_inference_checkpoint(
            manager.inference_path,
            model=model,
            metadata=metadata,
        )
        _emit(
            {
                "recipe": recipe,
                "initialization": initialization,
                "epochs": 0,
                "class_map_sha256": class_map_sha256,
                "inference_checkpoint": str(output),
            }
        )
        return 0

    config = _load_training_config(args, definition.family)
    request = ModelBuildRequest(
        model_key=args.model,
        num_classes=spec.num_classes,
        init_mode=initialization,
        seed=config.seed,
    )
    model = create_model(request)
    preprocessing = resolved_preprocessing(model, definition)
    train_transform = build_image_transform(definition, training=True, preprocessing=preprocessing)
    validation_transform = build_image_transform(
        definition, training=False, preprocessing=preprocessing
    )
    train_hf = load_hf_split(
        spec,
        train_plan.split,
        cache_dir=args.cache_dir,
        keep_in_memory=args.keep_in_memory,
        token=args.token,
    )
    validation_hf = load_hf_split(
        spec,
        validation_split,
        cache_dir=args.cache_dir,
        keep_in_memory=args.keep_in_memory,
        token=args.token,
    )
    train_records = records_for_source(manifest, train_source, split=train_plan.split)
    train_dataset = ManifestIndexedDataset(
        train_hf,
        train_records,
        image_column=spec.image_column,
        label_column=spec.label_column,
        transform=train_transform,
    )
    validation_dataset = ManifestIndexedDataset(
        validation_hf,
        validation_records,
        image_column=spec.image_column,
        label_column=spec.label_column,
        transform=validation_transform,
    )

    class_balance = str(getattr(args, "class_balance", "none"))
    if class_balance not in {"none", "weighted_loss", "balanced_sampler"}:
        raise ValueError(f"unknown class-balance mode {class_balance!r}")
    class_counts: list[int] | None = None
    class_weights: list[float] | None = None
    sampler_override = None
    if class_balance != "none":
        class_counts, class_weights = _class_balance_parameters(
            train_records, num_classes=spec.num_classes
        )
        if class_balance == "balanced_sampler":
            import torch
            from torch.utils.data import WeightedRandomSampler

            sample_weights = torch.as_tensor(
                [class_weights[int(record.label)] for record in train_records],
                dtype=torch.double,
            )
            sampler_override = WeightedRandomSampler(
                sample_weights,
                num_samples=len(train_records),
                replacement=True,
                generator=torch.Generator().manual_seed(config.seed),
            )

    trainer = Trainer(config, device=args.device)
    try:
        context = trainer.context
        train_loader = build_dataloader(
            train_dataset,
            config.train_loader,
            training=True,
            seed=config.seed,
            distributed=context.enabled,
            rank=context.rank,
            world_size=context.world_size,
            sampler_override=sampler_override,
        )
        validation_loader = build_dataloader(
            validation_dataset,
            config.validation_loader,
            training=False,
            seed=config.seed,
            distributed=context.enabled,
            rank=context.rank,
            world_size=context.world_size,
        )
        training_config = {"recipe": recipe, **config.to_dict()}
        if class_balance != "none":
            training_config["class_balance"] = {
                "mode": class_balance,
                "train_class_counts": class_counts,
                "normalized_inverse_frequency_weights": class_weights,
            }
        metadata = make_checkpoint_metadata(
            run_id=args.run_id,
            task_id=args.task_id or f"{train_plan.kind}-{args.model}-{train_source.source_id}",
            role=role,
            source_id=train_source.source_id,
            dataset_id=spec.dataset_id,
            dataset_revision=spec.revision,
            dataset_spec_fingerprint=spec.fingerprint,
            dataset_manifest_fingerprint=manifest.fingerprint,
            partition_kind=train_plan.kind,
            partition_digest=partition_digest,
            model_key=args.model,
            model_provider=definition.provider,
            initialization=initialization,
            num_classes=spec.num_classes,
            training_config=training_config,
            protocol_digest=args.protocol_digest,
            seed=config.seed,
            provenance=provenance,
        )
        manager = CheckpointManager(args.output)
        resume = None if args.resume == "none" else args.resume
        result = trainer.fit(
            model,
            train_loader,
            validation_loader,
            checkpoint_manager=manager,
            checkpoint_metadata=metadata,
            resume_from=resume,
            class_weights=(class_weights if class_balance == "weighted_loss" else None),
        )
        if context.is_primary:
            _emit(asdict(result))
    finally:
        close_distributed(trainer.context)
    return 0


def _compute_means(args: argparse.Namespace) -> int:
    args.manifest = str(resolve_full_matrix_runtime_path(args.manifest))
    if args.cache_dir is not None:
        args.cache_dir = str(resolve_full_matrix_runtime_path(args.cache_dir))
    args.output = str(resolve_full_matrix_runtime_path(args.output))
    spec = get_dataset_spec(args.dataset)
    if not args.split.lower().startswith("train"):
        raise ValueError("formal dataset/class means must be computed from a train split")
    manifest = read_manifest(args.manifest)
    manifest.validate(spec, reject_cross_split_duplicates=False)
    records = manifest.records_for_split(args.split)
    definition = get_model_definition(args.model)
    probe_model = create_model(
        ModelBuildRequest(
            model_key=args.model,
            num_classes=spec.num_classes,
            init_mode="random",
            seed=args.seed,
        )
    )
    preprocessing = resolved_preprocessing(probe_model, definition)
    del probe_model
    transform = build_raw_image_transform(definition, preprocessing=preprocessing)
    source = load_hf_split(
        spec,
        args.split,
        cache_dir=args.cache_dir,
        keep_in_memory=args.keep_in_memory,
        token=args.token,
    )
    dataset = ManifestIndexedDataset(
        source,
        records,
        image_column=spec.image_column,
        label_column=spec.label_column,
        transform=transform,
    )
    loader = build_dataloader(
        dataset,
        LoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=False,
            persistent_workers=args.workers > 0,
            prefetch_factor=args.prefetch_factor,
            drop_last=False,
        ),
        training=False,
        seed=args.seed,
    )
    accumulator = ImageMeanAccumulator(spec.num_classes)
    for batch in loader:
        accumulator.update(batch["image"], batch["label"])
    values = accumulator.finalize()
    artifact = write_image_mean_artifact(
        args.output,
        values,
        dataset_id=spec.dataset_id,
        dataset_revision=spec.revision,
        dataset_manifest_fingerprint=manifest.fingerprint,
        model_id=args.model,
        preprocessing=preprocessing,
        source_split=args.split,
        protocol_digest=args.protocol_digest,
    )
    _emit(
        {
            "artifact_id": artifact.artifact_id,
            "manifest": str((Path(args.output) / "manifest.json").resolve()),
            "sample_count": values.sample_count,
            "class_counts": values.class_counts.tolist(),
            "dataset_mean_shape": list(values.dataset_mean.shape),
        }
    )
    return 0


def register_subcommands(subparsers: Any) -> None:
    """Register Phase 0 commands on an existing argparse subparser action."""

    manifest = subparsers.add_parser("build-manifest", help="scan a pinned HF dataset")
    manifest.add_argument("--dataset", default=IMAGENET1K.key)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--cache-dir")
    manifest.add_argument("--streaming", action="store_true")
    manifest.add_argument("--token")
    manifest.add_argument(
        "--hash-mode", choices=("decoded_rgb", "encoded_bytes"), default="decoded_rgb"
    )
    manifest.add_argument("--splits", type=_csv_strings, default=None)
    manifest.add_argument("--allow-count-mismatch", action="store_true")
    manifest.set_defaults(handler=_build_manifest, _handler=_build_manifest)


    partitions = subparsers.add_parser(
        "build-partitions", help="build class-stratified IND/OVERLAP/reference partitions"
    )
    partitions.add_argument("--manifest", required=True)
    partitions.add_argument("--output", required=True)
    partitions.add_argument("--kind", choices=("ind", "overlap", "reference"), required=True)
    partitions.add_argument("--split", default="train")
    partitions.add_argument("--num-sources", type=int, default=10)
    partitions.add_argument("--seed", type=int, default=20260714)
    partitions.add_argument("--allow-remainder", action="store_true")
    partitions.add_argument("--samples-per-class", type=int)
    partitions.add_argument("--matched-ind-partition")
    partitions.add_argument("--overlap-mode", choices=("shared", "independent"), default="shared")
    partitions.set_defaults(handler=_build_partitions, _handler=_build_partitions)

    train = subparsers.add_parser("train", help="train or resume one source/reference model")
    train.add_argument("--dataset", default=IMAGENET1K.key)
    train.add_argument("--manifest", required=True)
    train.add_argument("--train-partition", required=True)
    train.add_argument("--validation-partition")
    train.add_argument("--validation-split", default="validation")
    train.add_argument("--source-id")
    train.add_argument("--model", required=True)
    train.add_argument(
        "--recipe",
        choices=tuple(sorted(INITIALIZATION_RECIPES)),
        help="Executable recipe; formal jobs always set this explicitly",
    )
    train.add_argument(
        "--initialization",
        choices=("random", "imagenet1k", "imagenet1k_subset"),
        default="random",
    )
    train.add_argument(
        "--class-map-indices",
        type=_csv_nonnegative_ints,
        help="Ordered dataset-label to ImageNet-1K indices for subset logits",
    )
    train.add_argument("--class-map-sha256")
    train.add_argument("--training-config")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--validation-batch-size", type=int)
    train.add_argument("--workers", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--precision", choices=("auto", "off", "fp16", "bf16"), default="auto")
    train.add_argument("--seed", type=int, default=20260714)
    train.add_argument("--device")
    train.add_argument("--cache-dir")
    train.add_argument("--keep-in-memory", action="store_true")
    train.add_argument("--token")
    train.add_argument("--output", required=True)
    train.add_argument("--resume", default="auto", help="auto, none, or a checkpoint path")
    train.add_argument(
        "--class-balance",
        choices=("none", "weighted_loss", "balanced_sampler"),
        default="none",
        help="recovery-only train-set class balancing strategy",
    )
    train.add_argument("--run-id", default="phase0")
    train.add_argument("--task-id")
    train.add_argument("--protocol-digest", default="standalone")
    train.add_argument("--project-root", default=".")
    train.set_defaults(handler=_train, _handler=_train)


    means = subparsers.add_parser(
        "compute-means",
        help="compute immutable pixelwise train-split dataset/class mean images",
    )
    means.add_argument("--dataset", default=IMAGENET1K.key)
    means.add_argument("--manifest", required=True)
    means.add_argument("--model", required=True)
    means.add_argument("--split", default="train")
    means.add_argument("--batch-size", type=int, default=128)
    means.add_argument("--workers", type=int, default=8)
    means.add_argument("--prefetch-factor", type=int, default=2)
    means.add_argument("--seed", type=int, default=20260714)
    means.add_argument("--cache-dir")
    means.add_argument("--keep-in-memory", action="store_true")
    means.add_argument("--token")
    means.add_argument("--protocol-digest", default="standalone")
    means.add_argument("--output", required=True)
    means.set_defaults(handler=_compute_means, _handler=_compute_means)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    register_subcommands(parser.add_subparsers(dest="command", required=True))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args._handler(args))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
