import enum

from torch import nn as nn

from query_strategies.batchBALD import multi_bald
import dataclasses
import typing
@dataclasses.dataclass
class AcquisitionBatch:
    indices: typing.List[int]
    scores: typing.List[float]
    orignal_scores: typing.Optional[typing.List[float]]


def acquire_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    min_candidates_per_acquired_item,
    min_remaining_percentage,
    initial_percentage,
    reduce_percentage,
    device=None,
) -> AcquisitionBatch:
    target_size = max(
        min_candidates_per_acquired_item * b, len(available_loader.dataset) * min_remaining_percentage // 100
    )

    return multi_bald.compute_multi_bald_batch(
        bayesian_model=bayesian_model,
        available_loader=available_loader,
        num_classes=num_classes,
        k=k,
        b=b,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        device=device,)

