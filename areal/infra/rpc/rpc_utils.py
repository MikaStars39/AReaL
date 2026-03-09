"""Shared utilities for RPC server implementations."""

from __future__ import annotations

from typing import Any

import torch

from areal.infra.rpc.rtensor import RTensor, TensorShardInfo


def remotize_traj_list(
    result: list[dict | torch.Tensor | None], node_addr: str
) -> list[Any]:
    """Remotize a per-trajectory list result from a TrainEngine call.

    Each trajectory dict or tensor gets its own RTensor shard so the controller
    can re-dispatch without data duplication.

    Parameters
    ----------
    result : list[dict | torch.Tensor | None]
        List of trajectory results (dicts with tensors, tensors, or None)
    node_addr : str
        Node address for shard storage (e.g., "host:port" or "" for Ray backend)

    Returns
    -------
    list[Any]
        List with tensors converted to RTensors
    """
    remotized_result = []
    for traj_result in result:
        if traj_result is None:
            remotized_result.append(None)
            continue
        if isinstance(traj_result, torch.Tensor):
            # Plain tensor result (e.g., from compute_logp)
            shard = TensorShardInfo(
                size=traj_result.shape[0],
                seqlens=[int(traj_result.shape[1])] if traj_result.ndim > 1 else [1],
                shard_id="",
                node_addr=node_addr,
            )
            traj_layout = RTensor(shard=shard, data=torch.empty(0, device="meta"))
            remotized_result.append(
                RTensor.remotize(traj_result, traj_layout, node_addr=node_addr)
            )
            continue
        # Dict result (e.g., from compute_advantages): use attention_mask for layout
        traj_layout = RTensor.extract_layout(
            traj_result,
            layouts=dict(args=None, kwargs=None),
            node_addr=node_addr,
        )
        if traj_layout is not None:
            remotized_result.append(
                RTensor.remotize(traj_result, traj_layout, node_addr=node_addr)
            )
        else:
            remotized_result.append(traj_result)
    return remotized_result
