"""
Test to verify the checkpoint manager keeps only the most recent N checkpoints.

Run:
    python tests/test_checkpoint_manager.py
"""

import tempfile
from collections import deque
from pathlib import Path
from typing import List


class MockCheckpointManager:
    """Mirrors the checkpoint logic from OffPolicyTrainer."""

    def __init__(self, log_path: Path, max_checkpoints: int = 10):
        self.log_path = log_path
        self.max_checkpoints = max_checkpoints
        self._checkpoint_history: deque[List[Path]] = deque()

    def save_checkpoint(self, files: List[Path]):
        for f in files:
            f.write_text(f"checkpoint {f.name}")
        self._checkpoint_history.append(files)
        if len(self._checkpoint_history) > self.max_checkpoints:
            old_files = self._checkpoint_history.popleft()
            for f in old_files:
                f.unlink(missing_ok=True)


def test_keeps_only_max_checkpoints():
    max_ckpts = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir)
        mgr = MockCheckpointManager(log_path, max_checkpoints=max_ckpts)

        all_files = []
        for i in range(6):
            policy = log_path / f"policy-{i}.pkl"
            value = log_path / f"value-{i}.pkl"
            mgr.save_checkpoint([policy, value])
            all_files.append((policy, value))

        # Only the last 3 checkpoints should exist
        for i in range(3):
            assert not all_files[i][0].exists(), f"policy-{i}.pkl should be deleted"
            assert not all_files[i][1].exists(), f"value-{i}.pkl should be deleted"

        for i in range(3, 6):
            assert all_files[i][0].exists(), f"policy-{i}.pkl should exist"
            assert all_files[i][1].exists(), f"value-{i}.pkl should exist"

        assert len(mgr._checkpoint_history) == max_ckpts
        print(f"PASS: Only last {max_ckpts} checkpoints kept out of 6")


def test_policy_only_no_value():
    max_ckpts = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir)
        mgr = MockCheckpointManager(log_path, max_checkpoints=max_ckpts)

        all_files = []
        for i in range(5):
            policy = log_path / f"policy-{i}.pkl"
            mgr.save_checkpoint([policy])
            all_files.append(policy)

        for i in range(3):
            assert not all_files[i].exists(), f"policy-{i}.pkl should be deleted"
        for i in range(3, 5):
            assert all_files[i].exists(), f"policy-{i}.pkl should exist"

        assert len(mgr._checkpoint_history) == max_ckpts
        print(f"PASS: Policy-only mode works correctly")


def test_fewer_than_max():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir)
        mgr = MockCheckpointManager(log_path, max_checkpoints=10)

        files = []
        for i in range(3):
            policy = log_path / f"policy-{i}.pkl"
            mgr.save_checkpoint([policy])
            files.append(policy)

        for f in files:
            assert f.exists(), f"{f.name} should still exist"

        assert len(mgr._checkpoint_history) == 3
        print("PASS: No deletions when under max_checkpoints")


if __name__ == "__main__":
    test_keeps_only_max_checkpoints()
    test_policy_only_no_value()
    test_fewer_than_max()
    print("\nAll checkpoint manager tests passed!")
