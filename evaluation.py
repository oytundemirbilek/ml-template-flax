# Metrics and Loss
import jax.numpy as jnp
from jax import Array


class MockLoss:
    def __init__(
        self,
        batch_first: bool = True,
        reduction: str = "mean",
        another_loss_weight: float = 1.0,
    ) -> None:
        self.another_loss_weight = another_loss_weight
        self.batch_first = batch_first
        self.reduction = reduction

    def compute(
        self,
        pred_data: Array,
        target_data: Array,
    ) -> Array:
        if self.batch_first:
            loss = jnp.sum(jnp.sqrt(jnp.square(pred_data - target_data)), axis=(1, 2))
        else:
            loss = jnp.sum(jnp.sqrt(jnp.square(pred_data - target_data)))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError(
                "Batch reduction options are only 'mean' or 'sum'"
            )
