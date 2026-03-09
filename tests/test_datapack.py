"""Tests for datapack allocation functions."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import AllocationMode
from areal.infra import TrainController
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.utils.datapack import (
    _aggregate_traj_rtensor_layouts,
    _concat_localized_traj_lists,
    balanced_greedy_partition,
    concat_traj_dicts,
    concat_traj_results,
    data_parallel_merge,
    dispatch_traj_list,
    ffd_allocate,
    split_result_to_traj_list,
)

# =============================================================================
# Test Data Generators
# =============================================================================


def generate_bimodal_seqlens(
    n_long: int, n_short: int, long_range: tuple, short_range: tuple, seed: int = 42
):
    """Generate bimodal distribution of sequence lengths (common in RL with varied prompts)."""
    rng = np.random.default_rng(seed)
    long_seqs = rng.integers(long_range[0], long_range[1], size=n_long).tolist()
    short_seqs = rng.integers(short_range[0], short_range[1], size=n_short).tolist()
    all_seqs = long_seqs + short_seqs
    rng.shuffle(all_seqs)
    return list(all_seqs)


def generate_uniform_seqlens(n: int, low: int, high: int, seed: int = 42):
    """Generate uniformly distributed sequence lengths."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=n).tolist()


def generate_skewed_seqlens(n: int, max_len: int, skew: float = 2.0, seed: int = 42):
    """Generate skewed distribution (many short, few long - typical for chat/code)."""
    rng = np.random.default_rng(seed)
    # Use beta distribution to create skew
    samples = rng.beta(1, skew, size=n)
    return [int(s * max_len) + 1 for s in samples]


def generate_exponential_seqlens(n: int, scale: float = 500.0, seed: int = 42):
    """Generate exponentially distributed sequence lengths (common in NLP tasks)."""
    rng = np.random.default_rng(seed)
    samples = rng.exponential(scale=scale, size=n)
    # Clip to reasonable range and convert to int
    return [max(1, min(int(s), 8192)) for s in samples]


def generate_multimodal_seqlens(
    n: int, modes: list[tuple[int, int, float]], seed: int = 42
):
    """Generate multimodal distribution with multiple peaks.

    Args:
        n: Total number of sequences
        modes: List of (mean, std, weight) tuples for each mode
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    total_weight = sum(w for _, _, w in modes)
    seqlens = []

    for mean, std, weight in modes:
        count = int(n * weight / total_weight)
        samples = rng.normal(mean, std, size=count)
        seqlens.extend([max(1, int(s)) for s in samples])

    # Fill remaining with first mode
    while len(seqlens) < n:
        mean, std, _ = modes[0]
        seqlens.append(max(1, int(rng.normal(mean, std))))

    rng.shuffle(seqlens)
    return seqlens[:n]


def generate_power_law_seqlens(n: int, alpha: float = 2.0, seed: int = 42):
    """Generate power-law distributed sequence lengths (Zipf-like)."""
    rng = np.random.default_rng(seed)
    # Use Pareto distribution (power law)
    samples = (rng.pareto(alpha, size=n) + 1) * 100
    return [max(1, min(int(s), 8192)) for s in samples]


def generate_batch_realistic_seqlens(
    batch_size: int, prompt_range: tuple, response_range: tuple, seed: int = 42
):
    """Generate realistic batch with prompt+response patterns (typical in RL training).

    In RL training, each sample has a prompt (input) and response (generated).
    The total sequence length varies based on both components.
    """
    rng = np.random.default_rng(seed)
    prompts = rng.integers(prompt_range[0], prompt_range[1], size=batch_size)
    responses = rng.integers(response_range[0], response_range[1], size=batch_size)
    return [int(p + r) for p, r in zip(prompts, responses)]


def generate_code_seqlens(n: int, seed: int = 42):
    """Generate sequence lengths typical for code generation tasks.

    Code has characteristic length distribution:
    - Many short snippets (1-100 tokens)
    - Medium functions (100-500 tokens)
    - Fewer long functions (500-2000 tokens)
    - Rare very long files (2000+ tokens)
    """
    return generate_multimodal_seqlens(
        n=n,
        modes=[
            (50, 30, 0.4),  # Short snippets
            (250, 100, 0.35),  # Medium functions
            (800, 300, 0.2),  # Long functions
            (2000, 500, 0.05),  # Very long files
        ],
        seed=seed,
    )


def generate_chat_seqlens(n: int, seed: int = 42):
    """Generate sequence lengths typical for chat/conversation data.

    Chat has characteristic length distribution:
    - Many short messages (10-100 tokens)
    - Medium responses (100-500 tokens)
    - Fewer long explanations (500-1500 tokens)
    """
    return generate_multimodal_seqlens(
        n=n,
        modes=[
            (50, 30, 0.5),  # Short messages
            (200, 80, 0.35),  # Medium responses
            (800, 300, 0.15),  # Long explanations
        ],
        seed=seed,
    )


def generate_math_seqlens(n: int, seed: int = 42):
    """Generate sequence lengths typical for math problem solving.

    Math problems have:
    - Short problem statements (50-200 tokens)
    - Variable solution lengths (100-1000 tokens)
    """
    return generate_batch_realistic_seqlens(
        batch_size=n,
        prompt_range=(50, 200),
        response_range=(100, 800),
        seed=seed,
    )


class TestBalancedGreedyPartition:
    """Tests for balanced_greedy_partition function."""

    @pytest.mark.parametrize("K", [2, 4, 8])
    def test_basic_partition(self, K):
        """Test basic partition returns correct structure."""
        n = K * 10  # 10 items per group
        nums = list(range(100, 100 + n))

        groups = balanced_greedy_partition(nums, K)

        assert len(groups) == K
        # Each group should have n/K items
        for g in groups:
            assert len(g) == n // K

    def test_returns_indices_not_values(self):
        """Test that function returns indices, not values."""
        nums = [100, 200, 300, 400]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        # Groups should contain indices (0-3), not values (100-400)
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == [0, 1, 2, 3]

        # Verify we can use indices to get original values
        for g in groups:
            values = [nums[i] for i in g]
            assert all(100 <= v <= 400 for v in values)

    def test_preserves_all_indices(self):
        """Test that all indices are assigned exactly once."""
        nums = [50, 100, 150, 200, 250, 300]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(nums)))

    def test_equal_group_sizes(self):
        """Test that all groups have equal size."""
        nums = list(range(24))
        K = 4

        groups = balanced_greedy_partition(nums, K)

        expected_size = len(nums) // K
        for g in groups:
            assert len(g) == expected_size

    def test_raises_on_non_divisible(self):
        """Test error when n is not divisible by K."""
        nums = [1, 2, 3, 4, 5]
        K = 2

        with pytest.raises(ValueError, match="must be divisible by K"):
            balanced_greedy_partition(nums, K)

    def test_raises_on_too_few_items(self):
        """Test error when n < K."""
        nums = [1, 2, 3]
        K = 5

        with pytest.raises(ValueError, match="must be >= K"):
            balanced_greedy_partition(nums, K)

    def test_raises_on_empty_input(self):
        """Test error when input is empty."""
        nums = []
        K = 4

        with pytest.raises(ValueError, match="must be >= K"):
            balanced_greedy_partition(nums, K)

    def test_balances_sums(self):
        """Test that group sums are well balanced."""
        # Create values with high variance
        nums = [1000, 900, 800, 700, 100, 200, 300, 400]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        sums = [sum(nums[i] for i in g) for g in groups]

        # Sums should be reasonably balanced
        total = sum(nums)
        expected_avg = total / K
        # Each group should be within 20% of average
        for s in sums:
            assert abs(s - expected_avg) / expected_avg < 0.3

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_bimodal_distribution(self, seed):
        """Test with bimodal sequence lengths (typical for RL with varied prompts)."""
        n_long, n_short = 8, 24
        values = generate_bimodal_seqlens(
            n_long=n_long,
            n_short=n_short,
            long_range=(1000, 2000),
            short_range=(100, 400),
            seed=seed,
        )
        K = 4

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))

        # Verify balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        # Difference should be reasonable
        assert max_diff / avg_sum < 0.5

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_uniform_distribution(self, seed):
        """Test with uniformly distributed sequence lengths."""
        values = generate_uniform_seqlens(n=200, low=512, high=2048, seed=seed)
        K = 8

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        for g in groups:
            assert len(g) == 25  # 200 / 8

        # Verify balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        assert max_diff / avg_sum < 0.2

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_skewed_distribution(self, seed):
        """Test with skewed distribution (many short, few long - like chat data)."""
        values = generate_skewed_seqlens(n=160, max_len=4096, skew=3.0, seed=seed)
        K = 4

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))

    def test_edge_case_identical_values(self):
        """Test when all values are identical."""
        nums = [100] * 20
        K = 4

        groups = balanced_greedy_partition(nums, K)

        # All groups should have equal size and equal sum
        assert len(groups) == K
        sums = [sum(nums[i] for i in g) for g in groups]
        assert all(s == sums[0] for s in sums)

    def test_edge_case_two_values(self):
        """Test with extreme two-value distribution."""
        nums = [1000] * 4 + [1] * 4
        K = 2

        groups = balanced_greedy_partition(nums, K)

        # Each group should ideally have 2 large and 2 small
        sums = [sum(nums[i] for i in g) for g in groups]
        # Both sums should be close
        assert abs(sums[0] - sums[1]) <= 2  # Small difference due to 1s

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_realistic_dp_sizes(self, dp_size):
        """Test with realistic data parallel sizes."""
        n_seqs = dp_size * 16  # 16 sequences per DP rank
        values = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(1000, 2000),
            short_range=(200, 600),
            seed=42,
        )

        groups = balanced_greedy_partition(values, dp_size)

        assert len(groups) == dp_size
        for g in groups:
            assert len(g) == 16

        # Check balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / dp_size
        assert max_diff / avg_sum < 0.3

    def test_large_scale(self):
        """Test with larger number of items."""
        n = 1000
        K = 10
        values = generate_uniform_seqlens(n=n, low=100, high=1000, seed=42)

        groups = balanced_greedy_partition(values, K)

        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(n))

        # Check balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        assert max_diff / avg_sum < 0.1  # Should be very balanced

    def test_single_item_per_group(self):
        """Test edge case where each group gets exactly one item."""
        nums = [100, 200, 300, 400]
        K = 4

        groups = balanced_greedy_partition(nums, K)

        assert len(groups) == K
        for g in groups:
            assert len(g) == 1

    def test_deterministic(self):
        """Test that function is deterministic."""
        nums = [300, 100, 400, 200, 500, 600]
        K = 2

        groups1 = balanced_greedy_partition(nums, K)
        groups2 = balanced_greedy_partition(nums, K)

        # Should produce same result
        assert groups1 == groups2


class TestFFDAllocate:
    """Tests for existing ffd_allocate function to ensure no regression."""

    def test_basic_allocation(self):
        """Test basic FFD allocation."""
        values = [100, 200, 300, 150, 250]
        capacity = 500
        min_groups = 2

        groups = ffd_allocate(values, capacity, min_groups)

        assert len(groups) >= min_groups
        # All indices should be present
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))
        # Each group should respect capacity
        for g in groups:
            total = sum(values[i] for i in g)
            assert total <= capacity

    def test_respects_min_groups(self):
        """Test that min_groups constraint is respected."""
        values = [50] * 10
        capacity = 1000
        min_groups = 4

        groups = ffd_allocate(values, capacity, min_groups)

        assert len(groups) >= min_groups

    def test_raises_on_value_exceeds_capacity(self):
        """Test error when a value exceeds capacity."""
        values = [100, 600, 200]
        capacity = 500

        with pytest.raises(RuntimeError, match="larger than capacity"):
            ffd_allocate(values, capacity, min_groups=1)

    def test_raises_on_insufficient_values(self):
        """Test error when not enough values for min_groups."""
        values = [100, 200]
        capacity = 500
        min_groups = 5

        with pytest.raises(RuntimeError, match="smaller than min_groups"):
            ffd_allocate(values, capacity, min_groups)


# =============================================================================
# Integration Tests: dispatch_traj_list
# =============================================================================


class TestDispatchTrajList:
    """Integration tests for dispatch_traj_list with balanced_greedy_partition.

    These tests verify that trajectory lists (list[dict[str, RTensor]]) are
    partitioned into equal-size groups for different DP ranks.
    """

    def _create_traj_list(self, seqlens: list[int]) -> list[dict[str, RTensor]]:
        """Helper to create a list of trajectory dicts (one per sequence)."""
        traj_list = []
        for slen in seqlens:
            shard = TensorShardInfo(
                size=1, seqlens=[slen], shard_id="test", node_addr=""
            )
            data = torch.zeros(1, slen)
            traj_list.append({"input_ids": RTensor(shard=shard, data=data)})
        return traj_list

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_uniform_distribution(self, dp_size):
        """Test that uniform distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=1000, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        assert len(splits) == dp_size
        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size, (
                f"DP rank {i} got {len(group)} trajectories, expected {expected_size}"
            )
            assert len(group_indices[i]) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_bimodal_distribution(self, dp_size):
        """Test that bimodal distribution splits into equal-size groups."""
        n_seqs = dp_size * 20
        seqlens = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(1000, 2000),
            short_range=(100, 400),
            seed=42,
        )
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_skewed_distribution(self, dp_size):
        """Test that skewed distribution splits into equal-size groups."""
        n_seqs = dp_size * 24
        seqlens = generate_skewed_seqlens(n=n_seqs, max_len=2000, skew=3.0, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_exponential_distribution(self, dp_size):
        """Test that exponential distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_exponential_seqlens(n=n_seqs, scale=500.0, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_code_distribution(self, dp_size):
        """Test that code-like distribution splits into equal-size groups."""
        n_seqs = dp_size * 32
        seqlens = generate_code_seqlens(n=n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_chat_distribution(self, dp_size):
        """Test that chat-like distribution splits into equal-size groups."""
        n_seqs = dp_size * 24
        seqlens = generate_chat_seqlens(n=n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_math_distribution(self, dp_size):
        """Test that math problem distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_math_seqlens(n=n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_power_law_distribution(self, dp_size):
        """Test that power-law distribution splits into equal-size groups."""
        n_seqs = dp_size * 20
        seqlens = generate_power_law_seqlens(n=n_seqs, alpha=2.0, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000, 2024, 3141, 9999])
    def test_equal_split_various_seeds(self, seed):
        """Test equal split consistency across many random seeds."""
        dp_size = 4
        n_seqs = 64
        seqlens = generate_bimodal_seqlens(
            n_long=16,
            n_short=48,
            long_range=(500, 1500),
            short_range=(50, 300),
            seed=seed,
        )
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    def test_all_indices_preserved_after_dispatch(self):
        """Test that all original indices are preserved after dispatch."""
        dp_size = 4
        n_seqs = 100
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        traj_list = self._create_traj_list(seqlens)

        _, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        all_indices = sorted(i for g in group_indices for i in g)
        assert all_indices == list(range(n_seqs))

    def test_token_balance_across_dp_ranks(self):
        """Test that total tokens are reasonably balanced across DP ranks."""
        dp_size = 4
        seqlens = generate_bimodal_seqlens(
            n_long=20,
            n_short=60,
            long_range=(1000, 2000),
            short_range=(100, 300),
            seed=42,
        )
        traj_list = self._create_traj_list(seqlens)

        _, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        tokens_per_rank = [sum(seqlens[i] for i in g) for g in group_indices]
        avg_tokens = sum(tokens_per_rank) / dp_size
        max_diff = max(tokens_per_rank) - min(tokens_per_rank)

        assert max_diff / avg_tokens < 0.3, (
            f"Token imbalance too high: {max_diff / avg_tokens:.2%}"
        )

    @pytest.mark.parametrize(
        "batch_size,dp_size",
        [
            (32, 2),
            (64, 4),
            (128, 8),
            (256, 8),
            (512, 8),
            (1024, 8),
        ],
    )
    def test_large_batch_equal_split(self, batch_size, dp_size):
        """Test equal split for various large batch sizes."""
        seqlens = generate_uniform_seqlens(n=batch_size, low=100, high=2000, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = batch_size // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    def test_dispatch_preserves_trajectory_content(self):
        """Test that dispatched trajectories contain the original dict content."""
        dp_size = 2
        seqlens = [100, 200, 300, 400]
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        # Every dispatched trajectory should be a dict with 'input_ids' RTensor
        for group in splits:
            for traj in group:
                assert isinstance(traj, dict)
                assert "input_ids" in traj
                assert isinstance(traj["input_ids"], RTensor)

    def test_dispatch_with_multi_key_trajectories(self):
        """Test dispatch with trajectory dicts containing multiple RTensor keys."""
        dp_size = 2
        n_seqs = 8
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        traj_list = []
        for slen in seqlens:
            shard = TensorShardInfo(
                size=1, seqlens=[slen], shard_id="test", node_addr=""
            )
            traj_list.append(
                {
                    "input_ids": RTensor(shard=shard, data=torch.zeros(1, slen)),
                    "labels": RTensor(shard=shard, data=torch.ones(1, slen)),
                }
            )

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for group in splits:
            assert len(group) == expected_size
            for traj in group:
                assert "input_ids" in traj
                assert "labels" in traj

    def test_dispatch_fallback_for_non_rtensor_dicts(self):
        """Test that dicts without RTensors get seqlen=1 fallback."""
        dp_size = 2
        traj_list = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "foo"},
            {"text": "bar"},
        ]

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=dp_size)

        assert len(splits) == dp_size
        total = sum(len(g) for g in splits)
        assert total == 4


class TestTrainControllerDispatchIntegration:
    """Integration tests for TrainController._dispatch_inputs with equal-size splits.

    These tests simulate the full dispatch flow from TrainController to verify
    that batches are split into equal sizes for different DP ranks.
    """

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler for testing."""
        scheduler = Mock()
        scheduler.async_call_engine = AsyncMock(return_value=None)
        return scheduler

    @pytest.fixture
    def train_config(self):
        """Create a TrainEngineConfig for testing."""
        return TrainEngineConfig(
            scheduling_spec=(
                SchedulingSpec(cpu=4, gpu=1, mem=16000, port_count=2, cmd="dummy"),
            )
        )

    def _create_traj_list(self, seqlens: list[int]) -> list[dict[str, RTensor]]:
        """Helper to create a list of trajectory dicts (one per sequence)."""
        traj_list = []
        for slen in seqlens:
            shard = TensorShardInfo(
                size=1, seqlens=[slen], shard_id="test", node_addr=""
            )
            data = torch.zeros(1, slen)
            traj_list.append({"input_ids": RTensor(shard=shard, data=data)})
        return traj_list

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_train_controller_dispatch_equal_split(
        self, mock_scheduler, train_config, dp_size
    ):
        """Test TrainController dispatches batches to equal-size DP groups."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        # Create batch with sequences that should split equally
        n_seqs = dp_size * 16
        seqlens = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(500, 1500),
            short_range=(100, 300),
            seed=42,
        )
        traj_list = self._create_traj_list(seqlens)

        split_args, split_kwargs, group_indices = controller._dispatch_inputs(traj_list)

        # Verify equal split
        expected_size = n_seqs // dp_size
        assert len(group_indices) == dp_size
        for i, g in enumerate(group_indices):
            assert len(g) == expected_size, (
                f"DP rank {i} got {len(g)} sequences, expected {expected_size}"
            )

    @pytest.mark.parametrize(
        "generator_name,generator_func",
        [
            ("uniform", lambda n, seed: generate_uniform_seqlens(n, 100, 1000, seed)),
            (
                "bimodal",
                lambda n, seed: generate_bimodal_seqlens(
                    n // 4, n - n // 4, (800, 1500), (100, 300), seed
                ),
            ),
            ("skewed", lambda n, seed: generate_skewed_seqlens(n, 2000, 3.0, seed)),
            (
                "exponential",
                lambda n, seed: generate_exponential_seqlens(n, 500.0, seed),
            ),
            ("code", lambda n, seed: generate_code_seqlens(n, seed)),
            ("chat", lambda n, seed: generate_chat_seqlens(n, seed)),
            ("math", lambda n, seed: generate_math_seqlens(n, seed)),
            ("power_law", lambda n, seed: generate_power_law_seqlens(n, 2.0, seed)),
        ],
    )
    def test_train_controller_dispatch_all_distributions(
        self, mock_scheduler, train_config, generator_name, generator_func
    ):
        """Test TrainController dispatch with all distribution types."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        n_seqs = dp_size * 20
        seqlens = generator_func(n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        _, _, group_indices = controller._dispatch_inputs(traj_list)

        expected_size = n_seqs // dp_size
        for i, g in enumerate(group_indices):
            assert len(g) == expected_size, (
                f"Distribution '{generator_name}': DP rank {i} got {len(g)} "
                f"sequences, expected {expected_size}"
            )

    def test_train_controller_dispatch_preserves_indices(
        self, mock_scheduler, train_config
    ):
        """Test that all sequence indices are preserved after dispatch."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        n_seqs = 100
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        traj_list = self._create_traj_list(seqlens)

        _, _, group_indices = controller._dispatch_inputs(traj_list)

        all_indices = sorted(i for g in group_indices for i in g)
        assert all_indices == list(range(n_seqs))

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_train_controller_dispatch_deterministic(
        self, mock_scheduler, train_config, seed
    ):
        """Test that dispatch is deterministic for the same input."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        seqlens = generate_bimodal_seqlens(
            n_long=16,
            n_short=48,
            long_range=(500, 1500),
            short_range=(100, 300),
            seed=seed,
        )
        traj_list = self._create_traj_list(seqlens)

        _, _, group_indices1 = controller._dispatch_inputs(traj_list)

        # Dispatch again with same data
        traj_list2 = self._create_traj_list(seqlens)
        _, _, group_indices2 = controller._dispatch_inputs(traj_list2)

        assert group_indices1 == group_indices2


class TestConcatTrajDicts:
    """Test concat_traj_dicts and _concat_localized_traj_lists utilities."""

    def test_concat_single_dict_returns_same(self):
        d = {"x": torch.randn(2, 3)}
        result = concat_traj_dicts([d])
        assert result is d

    def test_concat_two_dicts_pads_and_cats(self):
        d1 = {"x": torch.ones(2, 3), "y": torch.ones(2, 5)}
        d2 = {"x": torch.zeros(1, 4), "y": torch.zeros(1, 5)}
        result = concat_traj_dicts([d1, d2])
        assert result["x"].shape == (3, 4)  # padded to max dim-1
        assert result["y"].shape == (3, 5)
        assert torch.allclose(result["x"][:2, :3], torch.ones(2, 3))
        assert torch.allclose(result["x"][2:, :4], torch.zeros(1, 4))

    def test_concat_empty_returns_empty(self):
        assert concat_traj_dicts([]) == {}

    def test_concat_list_values(self):
        d1 = {"ids": [1, 2]}
        d2 = {"ids": [3, 4, 5]}
        result = concat_traj_dicts([d1, d2])
        assert result["ids"] == [1, 2, 3, 4, 5]

    def test_concat_localized_traj_lists_walks_structure(self):
        """_concat_localized_traj_lists concats list[dict] inside args."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(1, 3)
        args = [
            [{"x": t1}, {"x": t2}],  # traj list → should be concatenated
            42,  # scalar → unchanged
        ]
        result = _concat_localized_traj_lists(args)
        assert isinstance(result[0], dict)  # was list, now single dict
        assert result[0]["x"].shape == (3, 3)
        assert result[1] == 42

    def test_concat_localized_traj_lists_concats_all_list_dicts(self):
        """_concat_localized_traj_lists concats ANY list[dict], including non-tensor.

        The guard against concatenating chat messages is at the call site
        (rpc_server.py only calls this for TrainEngine), not in this function.
        """
        # Even message dicts get concatenated — caller must guard
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = _concat_localized_traj_lists(messages)
        assert isinstance(result, dict)
        assert result["role"] == "user"

        # Tensor dicts also get concatenated
        t1 = torch.randn(1, 10)
        t2 = torch.randn(1, 10)
        traj_list = [{"input_ids": t1}, {"input_ids": t2}]
        result = _concat_localized_traj_lists(traj_list)
        assert isinstance(result, dict)
        assert result["input_ids"].shape == (2, 10)


class TestAggregateTrajRtensorLayouts:
    """Test _aggregate_traj_rtensor_layouts for correct seqlens aggregation."""

    def test_aggregates_seqlens_from_traj_list(self):
        """Simulates the RPC server receiving list[dict[str, RTensor]] from controller.

        raw_args structure: [list[dict[str, RTensor]]]
        After aggregation, extract_layout should find an RTensor with ALL seqlens.
        """
        # Build raw_args as the RPC server sees them after deserialization:
        # [list_of_traj_dicts]  (positional args tuple)
        traj0 = {
            "input_ids": RTensor(
                shard=TensorShardInfo(
                    size=1, seqlens=[128], shard_id="a", node_addr="w0"
                ),
                data=torch.empty(0, device="meta"),
            ),
        }
        traj1 = {
            "input_ids": RTensor(
                shard=TensorShardInfo(
                    size=1, seqlens=[256], shard_id="b", node_addr="w0"
                ),
                data=torch.empty(0, device="meta"),
            ),
        }
        traj2 = {
            "input_ids": RTensor(
                shard=TensorShardInfo(
                    size=1, seqlens=[64], shard_id="c", node_addr="w0"
                ),
                data=torch.empty(0, device="meta"),
            ),
        }
        raw_args = [[traj0, traj1, traj2]]  # [list[dict[str, RTensor]]]

        result = _aggregate_traj_rtensor_layouts(raw_args)

        # Should be [aggregated_RTensor]
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], RTensor)
        assert result[0].shard.seqlens == [128, 256, 64]
        assert result[0].shard.size == 3

    def test_end_to_end_merge_rejects_rtensors(self):
        """RTensors should not be merged via data_parallel_merge (use _reorder_traj_results)."""
        # 4 trajectories, 2 DP groups
        traj_list = []
        for i, sl in enumerate([128, 256, 64, 192]):
            traj_list.append(
                {
                    "input_ids": RTensor(
                        shard=TensorShardInfo(
                            size=1, seqlens=[sl], shard_id=f"s{i}", node_addr="w0"
                        ),
                        data=torch.empty(0, device="meta"),
                    ),
                }
            )

        splits, group_indices = dispatch_traj_list(traj_list, dp_size=2)

        # Simulate what each worker returns: an RTensor with aggregated seqlens
        results = []
        for group_trajs in splits:
            worker_seqlens = []
            for d in group_trajs:
                for v in d.values():
                    if isinstance(v, RTensor):
                        worker_seqlens.extend(v.shard.seqlens)
                        break
            n_trajs = len(group_trajs)
            results.append(
                RTensor(
                    shard=TensorShardInfo(
                        size=n_trajs,
                        seqlens=worker_seqlens,
                        shard_id="result",
                        node_addr="w0",
                    ),
                    data=torch.randn(n_trajs, 10),
                )
            )

        # RTensors should be rejected by data_parallel_merge
        with pytest.raises(TypeError, match="RTensors should not be merged"):
            data_parallel_merge(results)


class TestSplitResultToTrajList:
    """Tests for split_result_to_traj_list."""

    def test_split_tensor(self):
        """Split a batched tensor into per-traj tensors."""
        batched = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = split_result_to_traj_list(batched, n_trajs=3)
        assert isinstance(result, list)
        assert len(result) == 3
        torch.testing.assert_close(result[0], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(result[2], torch.tensor([[5.0, 6.0]]))

    def test_split_dict(self):
        """Split a batched dict into per-traj dicts."""
        batched = {
            "logp": torch.tensor([1.0, 2.0]),
            "values": torch.tensor([3.0, 4.0]),
            "scalar": "shared",
        }
        result = split_result_to_traj_list(batched, n_trajs=2)
        assert isinstance(result, list)
        assert len(result) == 2
        torch.testing.assert_close(result[0]["logp"], torch.tensor([1.0]))
        torch.testing.assert_close(result[1]["values"], torch.tensor([4.0]))
        assert result[0]["scalar"] == "shared"

    def test_split_none(self):
        """None input returns None."""
        assert split_result_to_traj_list(None, n_trajs=3) is None

    def test_split_passthrough(self):
        """Non-tensor/dict/None input is returned as-is."""
        assert split_result_to_traj_list(42, n_trajs=2) == 42


class TestConcatTrajResults:
    """Tests for concat_traj_results."""

    def test_concat_tensor_list(self):
        """Concat list[Tensor] back into single tensor."""
        tensors = [torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])]
        result = concat_traj_results(tensors)
        torch.testing.assert_close(result, torch.tensor([[1.0], [2.0], [3.0]]))

    def test_concat_dict_list(self):
        """Concat list[dict] back into single dict."""
        dicts = [
            {"a": torch.tensor([[1.0]])},
            {"a": torch.tensor([[2.0]])},
        ]
        result = concat_traj_results(dicts)
        assert isinstance(result, dict)
        torch.testing.assert_close(result["a"], torch.tensor([[1.0], [2.0]]))

    def test_concat_none_list(self):
        """List of Nones returns None."""
        assert concat_traj_results([None, None]) is None

    def test_concat_non_list(self):
        """Non-list input passes through."""
        t = torch.tensor([1.0])
        assert concat_traj_results(t) is t

    def test_concat_empty(self):
        """Empty list passes through."""
        assert concat_traj_results([]) == []

    def test_roundtrip_tensor(self):
        """split then concat is identity for tensors."""
        original = torch.randn(4, 8)
        split = split_result_to_traj_list(original, n_trajs=4)
        restored = concat_traj_results(split)
        torch.testing.assert_close(restored, original)

    def test_roundtrip_dict(self):
        """split then concat is identity for dicts."""
        original = {"a": torch.randn(3, 5), "b": torch.randn(3, 2)}
        split = split_result_to_traj_list(original, n_trajs=3)
        restored = concat_traj_results(split)
        torch.testing.assert_close(restored["a"], original["a"])
        torch.testing.assert_close(restored["b"], original["b"])


# =============================================================================
# Wave 1 & 2 Fix Verification Tests
# =============================================================================


class TestConcatTrajDictsKeyValidation:
    """Tests for M4: concat_traj_dicts key validation."""

    def test_consistent_keys_succeeds(self):
        """concat_traj_dicts works when all dicts have same keys."""
        dicts = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3, 4])}]
        result = concat_traj_dicts(dicts)
        assert "a" in result

    def test_inconsistent_keys_raises(self):
        """concat_traj_dicts raises ValueError on mismatched keys."""
        dicts = [{"a": torch.tensor([1])}, {"b": torch.tensor([2])}]
        with pytest.raises(ValueError):
            concat_traj_dicts(dicts)

    def test_extra_key_in_later_dict_raises(self):
        """concat_traj_dicts raises ValueError when later dict has extra key."""
        dicts = [
            {"a": torch.tensor([1])},
            {"a": torch.tensor([2]), "b": torch.tensor([3])},
        ]
        with pytest.raises(ValueError):
            concat_traj_dicts(dicts)

    def test_missing_key_in_later_dict_raises(self):
        """concat_traj_dicts raises ValueError when later dict misses key."""
        dicts = [
            {"a": torch.tensor([1]), "b": torch.tensor([2])},
            {"a": torch.tensor([3])},
        ]
        with pytest.raises(ValueError):
            concat_traj_dicts(dicts)


class TestSplitResultRefSafety:
    """Tests for H3: split_result_to_traj_list reference safety."""

    def test_non_tensor_values_are_copied(self):
        """Non-tensor values in split results are copies, not shared references."""
        shared_list = [1, 2, 3]
        result = {"data": torch.tensor([[1.0], [2.0]]), "meta": shared_list}
        splits = split_result_to_traj_list(result, n_trajs=2, traj_batch_size=1)
        # Modifying one split's meta should not affect the other
        splits[0]["meta"].append(99)
        assert 99 not in splits[1]["meta"], (
            "Non-tensor values should be independent copies"
        )

    def test_non_tensor_dict_values_are_copied(self):
        """Non-tensor dict values should be shallow-copied per split."""
        shared_dict = {"key": "value"}
        result = {"data": torch.tensor([[1.0], [2.0]]), "config": shared_dict}
        splits = split_result_to_traj_list(result, n_trajs=2, traj_batch_size=1)
        # The config should be a copy, not the same object
        assert splits[0]["config"] is not splits[1]["config"]

    def test_scalar_values_preserved_across_splits(self):
        """Scalar values should be preserved in all splits."""
        result = {"data": torch.tensor([[1.0], [2.0]]), "count": 42}
        splits = split_result_to_traj_list(result, n_trajs=2, traj_batch_size=1)
        assert splits[0]["count"] == 42
        assert splits[1]["count"] == 42


class TestTrajListBatchConversion:
    """Tests for M2: traj_list_to_batch and batch_to_traj_list."""

    def test_traj_list_to_batch_basic(self):
        """traj_list_to_batch concatenates trajectory dicts."""
        from areal.utils.datapack import traj_list_to_batch

        traj_list = [
            {"input_ids": torch.tensor([[1, 2, 3]])},
            {"input_ids": torch.tensor([[4, 5, 6]])},
        ]
        batched, traj_batch_size = traj_list_to_batch(traj_list)
        assert batched["input_ids"].shape == (2, 3)
        assert traj_batch_size == 1

    def test_traj_list_to_batch_rejects_non_list(self):
        """traj_list_to_batch raises AssertionError for non-list input."""
        from areal.utils.datapack import traj_list_to_batch

        with pytest.raises(AssertionError):
            traj_list_to_batch({"a": 1})

    def test_traj_list_to_batch_rejects_non_dict_elements(self):
        """traj_list_to_batch raises AssertionError for list of non-dicts."""
        from areal.utils.datapack import traj_list_to_batch

        with pytest.raises(AssertionError):
            traj_list_to_batch(["not", "dicts"])

    def test_batch_to_traj_list_basic(self):
        """batch_to_traj_list splits batched dict back into trajectory list."""
        from areal.utils.datapack import batch_to_traj_list

        batched = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        splits = batch_to_traj_list(batched, n_trajs=2, traj_batch_size=1)
        assert len(splits) == 2
        assert splits[0]["input_ids"].shape == (1, 2)
        assert splits[1]["input_ids"].shape == (1, 2)

    def test_batch_to_traj_list_with_traj_batch_size(self):
        """batch_to_traj_list handles traj_batch_size > 1 (group_size)."""
        from areal.utils.datapack import batch_to_traj_list

        batched = {"x": torch.tensor([[1], [2], [3], [4]])}
        splits = batch_to_traj_list(batched, n_trajs=2, traj_batch_size=2)
        assert len(splits) == 2
        assert splits[0]["x"].shape == (2, 1)
        assert splits[1]["x"].shape == (2, 1)

    def test_roundtrip(self):
        """traj_list_to_batch and batch_to_traj_list are inverses."""
        from areal.utils.datapack import batch_to_traj_list, traj_list_to_batch

        traj_list = [
            {"x": torch.tensor([[1.0, 2.0]])},
            {"x": torch.tensor([[3.0, 4.0]])},
        ]
        batched, traj_batch_size = traj_list_to_batch(traj_list)
        recovered = batch_to_traj_list(
            batched, n_trajs=2, traj_batch_size=traj_batch_size
        )
        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0]["x"], traj_list[0]["x"])
        torch.testing.assert_close(recovered[1]["x"], traj_list[1]["x"])

    def test_roundtrip_multiple_keys(self):
        """Roundtrip preserves multiple keys per trajectory dict."""
        from areal.utils.datapack import batch_to_traj_list, traj_list_to_batch

        traj_list = [
            {"a": torch.tensor([[1.0]]), "b": torch.tensor([[2.0]])},
            {"a": torch.tensor([[3.0]]), "b": torch.tensor([[4.0]])},
        ]
        batched, traj_batch_size = traj_list_to_batch(traj_list)
        recovered = batch_to_traj_list(
            batched, n_trajs=2, traj_batch_size=traj_batch_size
        )
        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0]["a"], traj_list[0]["a"])
        torch.testing.assert_close(recovered[0]["b"], traj_list[0]["b"])
        torch.testing.assert_close(recovered[1]["a"], traj_list[1]["a"])
        torch.testing.assert_close(recovered[1]["b"], traj_list[1]["b"])


class TestDataParallelMergeTypes:
    """Tests for M3: data_parallel_merge list/tuple handling."""

    def test_merge_list_of_lists_preserves_type(self):
        """data_parallel_merge preserves list type for list-of-lists."""
        results = [[42], [43]]
        merged = data_parallel_merge(results)
        assert isinstance(merged, list)

    def test_merge_list_of_tuples_preserves_type(self):
        """data_parallel_merge preserves tuple type for list-of-tuples."""
        results = [(42,), (43,)]
        merged = data_parallel_merge(results)
        assert isinstance(merged, tuple)

    def test_merge_nested_lists(self):
        """data_parallel_merge handles nested list structures."""
        results = [[[1, 2]], [[3, 4]]]
        merged = data_parallel_merge(results)
        assert isinstance(merged, list)
        assert isinstance(merged[0], list)

    def test_merge_nested_tuples(self):
        """data_parallel_merge handles nested tuple structures."""
        results = [((1,), (2,)), ((3,), (4,))]
        merged = data_parallel_merge(results)
        assert isinstance(merged, tuple)

    def test_merge_empty_returns_none(self):
        """data_parallel_merge returns None for empty results."""
        merged = data_parallel_merge([])
        assert merged is None

    def test_merge_dict_of_lists(self):
        """data_parallel_merge handles dicts containing list values."""
        results = [{"a": [1, 2]}, {"a": [3, 4]}]
        merged = data_parallel_merge(results)
        assert isinstance(merged, dict)
        assert isinstance(merged["a"], list)


class TestIsTrajList:
    """Tests for H1: TrainController._is_traj_list tensor check."""

    def test_rejects_non_tensor_dict_list(self):
        """_is_traj_list returns False for list of dicts without tensor values."""
        assert not TrainController._is_traj_list([{"key": "string_value"}])

    def test_rejects_dict_with_int_values(self):
        """_is_traj_list returns False for list of dicts with int values."""
        assert not TrainController._is_traj_list([{"count": 42}])

    def test_rejects_dict_with_nested_dict(self):
        """_is_traj_list returns False for list of dicts with nested dict."""
        assert not TrainController._is_traj_list([{"config": {"key": "value"}}])

    def test_accepts_torch_tensor_dict_list(self):
        """_is_traj_list returns True for list of dicts with torch.Tensor values."""
        assert TrainController._is_traj_list([{"logits": torch.tensor([1.0])}])

    def test_accepts_rtensor_dict_list(self):
        """_is_traj_list returns True for list of dicts with RTensor values."""
        shard = TensorShardInfo(size=1, seqlens=[5], shard_id="test", node_addr="")
        rtensor = RTensor(shard=shard, data=torch.zeros(1, 5))
        assert TrainController._is_traj_list([{"input_ids": rtensor}])

    def test_rejects_empty_list(self):
        """_is_traj_list returns False for empty list."""
        assert not TrainController._is_traj_list([])

    def test_rejects_non_list(self):
        """_is_traj_list returns False for non-list input."""
        assert not TrainController._is_traj_list({"key": "value"})

    def test_rejects_list_of_non_dicts(self):
        """_is_traj_list returns False for list of non-dicts."""
        assert not TrainController._is_traj_list([1, 2, 3])

    def test_accepts_dict_with_mixed_tensor_and_non_tensor(self):
        """_is_traj_list returns True when at least one value is tensor."""
        assert TrainController._is_traj_list(
            [{"data": torch.tensor([1.0]), "meta": "string"}]
        )


class TestAggregateRTensorLayoutsNoRTensor:
    """Tests for H2: _aggregate_traj_rtensor_layouts with no RTensor."""

    def test_handles_dict_without_rtensor(self):
        """_aggregate_traj_rtensor_layouts handles dicts without RTensor values."""
        # Dict with only non-RTensor values should return list unchanged (recurse)
        result = _aggregate_traj_rtensor_layouts([{"key": "string_value"}])
        # When no RTensor found in first dict, it returns list as-is
        assert isinstance(result, list)

    def test_handles_nested_non_rtensor_structure(self):
        """_aggregate_traj_rtensor_layouts handles nested structures without RTensor."""
        nested = {"outer": [{"key": 42}]}
        result = _aggregate_traj_rtensor_layouts(nested)
        assert isinstance(result, dict)
        assert "outer" in result

    def test_handles_tuple_without_rtensor(self):
        """_aggregate_traj_rtensor_layouts handles tuples without RTensor."""
        result = _aggregate_traj_rtensor_layouts(({"a": 1}, {"b": 2}))
        assert isinstance(result, tuple)

    def test_returns_scalar_unchanged(self):
        """_aggregate_traj_rtensor_layouts returns scalars unchanged."""
        assert _aggregate_traj_rtensor_layouts(42) == 42
        assert _aggregate_traj_rtensor_layouts("string") == "string"

    def test_skips_dicts_without_rtensor_in_list(self):
        """When some dicts in list lack RTensor, they contribute 0 seqlens."""
        # Mixed list: one dict with RTensor, one without
        shard = TensorShardInfo(size=1, seqlens=[100], shard_id="test", node_addr="")
        rtensor = RTensor(shard=shard, data=torch.zeros(1, 100))
        obj = [
            {"input_ids": rtensor},
            {"meta": "no rtensor here"},  # No RTensor in this dict
        ]
        result = _aggregate_traj_rtensor_layouts(obj)
        # Result should be RTensor with only seqlens from first dict
        assert isinstance(result, RTensor)
        assert result.shard.seqlens == [100]
        assert result.shard.size == 1
