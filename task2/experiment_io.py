"""Configuration merging, fingerprints, atomic checkpoints, and evaluation lock."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml


def deep_merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    base_path: str | Path, method_path: str | Path, overrides: dict | None = None
) -> dict:
    with Path(base_path).open("r", encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    with Path(method_path).open("r", encoding="utf-8") as stream:
        method = yaml.safe_load(stream)
    config = deep_merge(base, method)
    return deep_merge(config, overrides or {})


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def atomic_json_dump(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def lock_experiments(records: list[dict], path: str | Path) -> dict:
    """Freeze resolved configurations and selected checkpoints before label access."""
    locked_runs = []
    for record in records:
        checkpoint = Path(record["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for lock: {checkpoint}")
        locked_runs.append(
            {
                "run_name": record["run_name"],
                "config_fingerprint": record["config_fingerprint"],
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    payload = {"target_labels_consulted": False, "runs": locked_runs}
    destination = Path(path)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing["runs"] != locked_runs:
            raise RuntimeError(
                "A different experiment set is already locked. Do not revise it after "
                "target evaluation; use a new results directory for a new study."
            )
        return existing
    atomic_json_dump(payload, path)
    return payload


def validate_experiment_lock(path: str | Path, records: list[dict]) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    locked = {item["run_name"]: item for item in payload["runs"]}
    current = {item["run_name"]: item for item in records}
    if set(locked) != set(current):
        raise RuntimeError("The run set differs from the frozen experiment lock")
    for run_name, record in current.items():
        expected = locked[run_name]
        if record["config_fingerprint"] != expected["config_fingerprint"]:
            raise RuntimeError(f"Configuration changed after locking: {run_name}")
        if file_sha256(record["checkpoint_path"]) != expected["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed after locking: {run_name}")
    return payload


def mark_target_label_access(path: str | Path) -> dict:
    """Irreversibly record that the final label-aware stage has begun."""
    destination = Path(path)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if not payload.get("target_labels_consulted", False):
        payload["target_labels_consulted"] = True
        atomic_json_dump(payload, destination)
    return payload
