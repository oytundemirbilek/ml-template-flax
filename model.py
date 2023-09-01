# Our solution
from typing import List
from jax import Array
from flax.linen import Module, Sequential, Dense, relu, compact


class MockModel(Module):
    """ """

    in_features: int
    out_features: int
    batch_size: int
    layer_sizes: List[int]

    @compact
    def forward(self, input_data: Array) -> Array:
        return Sequential(
            [
                Dense(self.layer_sizes[0]),
                relu,
                Dense(self.layer_sizes[1]),
                relu,
                Dense(self.out_features),
            ]
        )(input_data)
