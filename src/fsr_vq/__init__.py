"""Function-Space Residual Vector Quantization primitives."""

from .formats import deployable_payload_nbytes
from .candidate_graph import (
    LinearFunctionQuadratic,
    diagonal_residual_target_distances,
    linear_function_quadratic,
    residual_target_distances,
)
from .residual import (
    HardLinearVectorState,
    LinearTransitionProposal,
    exact_linear_transition_delta,
)

__all__ = [
    "HardLinearVectorState",
    "LinearFunctionQuadratic",
    "LinearTransitionProposal",
    "diagonal_residual_target_distances",
    "deployable_payload_nbytes",
    "exact_linear_transition_delta",
    "linear_function_quadratic",
    "residual_target_distances",
]
