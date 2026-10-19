"""Crash-safe execution helpers for serial and sharded TDW experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import tempfile
import time


SUPPORTED_SHARD_COUNTS = (1, 2, 3)


def _write_json_file(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_json(path, payload):
    """Replace a JSON file atomically, never exposing a partial final file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{}.".format(target.name),
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(descriptor)
    try:
        _write_json_file(temporary, payload)
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def create_or_validate_json(path, payload):
    """Atomically create one shared JSON manifest or validate its exact value."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{}.".format(target.name),
        suffix=".candidate",
        dir=str(target.parent),
    )
    os.close(descriptor)
    created = False
    try:
        _write_json_file(temporary, payload)
        try:
            # A fully written hard link appears atomically. This avoids a
            # second shard observing a partially written shared manifest.
            os.link(temporary, target)
            created = True
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            "shared experiment manifest is unreadable; do not overwrite it: {}".format(
                error
            )
        ) from error
    if existing != payload:
        raise ValueError(
            "existing logical-run manifest differs; use the same full task/config "
            "arguments or a new run_id"
        )
    return created


def validate_shard_arguments(num_shards, shard_id):
    if isinstance(num_shards, bool) or int(num_shards) not in SUPPORTED_SHARD_COUNTS:
        raise ValueError("num_shards must be one of 1, 2, or 3")
    if isinstance(shard_id, bool) or int(shard_id) < 0 or int(shard_id) >= int(num_shards):
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return int(num_shards), int(shard_id)


def select_shard_episodes(data, episodes, num_shards, shard_id):
    """Assign complete matched pairs round-robin while preserving input order."""
    num_shards, shard_id = validate_shard_arguments(num_shards, shard_id)
    episodes = [int(value) for value in episodes]
    if num_shards == 1:
        return episodes
    groups = {}
    pair_order = []
    for episode in episodes:
        task = data[episode]
        if "matched_pair_id" not in task:
            raise ValueError(
                "sharded execution requires matched_pair_id for episode {}".format(
                    episode
                )
            )
        pair_id = str(task["matched_pair_id"])
        if pair_id not in groups:
            groups[pair_id] = []
            pair_order.append(pair_id)
        groups[pair_id].append(episode)
    selected_pairs = {
        pair_id
        for index, pair_id in enumerate(pair_order)
        if index % num_shards == shard_id
    }
    return [
        episode
        for episode in episodes
        if str(data[episode]["matched_pair_id"]) in selected_pairs
    ]


def stable_episode_seed(experiment_seed, run_id, episode):
    if isinstance(experiment_seed, bool):
        raise ValueError("experiment_seed must be a non-negative integer")
    experiment_seed = int(experiment_seed)
    if experiment_seed < 0:
        raise ValueError("experiment_seed must be a non-negative integer")
    digest = hashlib.sha256(
        "{}:{}:{}".format(experiment_seed, run_id, int(episode)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def seed_episode_randomness(experiment_seed, run_id, episode):
    """Make local fallback/exploration randomness invariant to shard count."""
    seed = stable_episode_seed(experiment_seed, run_id, episode)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        # NumPy is present in TDW runs; keeping this helper import-safe makes
        # the lightweight execution tests independent of simulator packages.
        pass
    return seed


class EpisodeClaim:
    """Advisory process lock preventing duplicate execution of one episode."""

    def __init__(self, episode_dir, episode, shard_id):
        self.path = Path(episode_dir) / ".episode.lock"
        self.episode = int(episode)
        self.shard_id = int(shard_id)
        self._stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise RuntimeError(
                "episode {} is already claimed by another worker".format(
                    self.episode
                )
            ) from error
        self._stream.seek(0)
        self._stream.truncate()
        json.dump(
            {
                "episode": self.episode,
                "shard_id": self.shard_id,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "claimed_at_unix": time.time(),
            },
            self._stream,
            indent=2,
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._stream is not None:
            try:
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._stream.close()
                self._stream = None

