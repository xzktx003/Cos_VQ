"""Accepted hard-transition trajectory invariants for FSR-VQ."""

from __future__ import annotations

import math
from typing import Any


class AcceptedTransitionTrajectory:
    """Track a strictly improving train-and-validation hard-state trajectory."""

    def __init__(self, *, initial_train_loss: float, initial_validation_loss: float) -> None:
        if not math.isfinite(initial_train_loss) or not math.isfinite(
            initial_validation_loss
        ):
            raise ValueError("initial trajectory losses must be finite")
        self.initial_train_loss = float(initial_train_loss)
        self.initial_validation_loss = float(initial_validation_loss)
        self.current_train_loss = float(initial_train_loss)
        self.current_validation_loss = float(initial_validation_loss)
        self.records: list[dict[str, Any]] = []

    @property
    def accepted_transitions(self) -> int:
        return len(self.records)

    def append(
        self,
        *,
        train_loss: float,
        validation_loss: float,
        metadata: dict[str, Any],
    ) -> None:
        train_loss = float(train_loss)
        validation_loss = float(validation_loss)
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise ValueError("trajectory losses must be finite")
        if train_loss >= self.current_train_loss:
            raise ValueError("accepted train loss must strictly decrease")
        if validation_loss >= self.current_validation_loss:
            raise ValueError("accepted validation loss must strictly decrease")
        self.records.append(
            {
                "step": len(self.records) + 1,
                "source_train_loss": self.current_train_loss,
                "train_loss": train_loss,
                "source_validation_loss": self.current_validation_loss,
                "validation_loss": validation_loss,
                "train_improvement": self.current_train_loss - train_loss,
                "validation_improvement": (
                    self.current_validation_loss - validation_loss
                ),
                "metadata": dict(metadata),
            }
        )
        self.current_train_loss = train_loss
        self.current_validation_loss = validation_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_train_loss": self.initial_train_loss,
            "initial_validation_loss": self.initial_validation_loss,
            "current_train_loss": self.current_train_loss,
            "current_validation_loss": self.current_validation_loss,
            "accepted_transitions": self.accepted_transitions,
            "train_relative_improvement": (
                self.initial_train_loss - self.current_train_loss
            )
            / max(abs(self.initial_train_loss), 1e-30),
            "validation_relative_improvement": (
                self.initial_validation_loss - self.current_validation_loss
            )
            / max(abs(self.initial_validation_loss), 1e-30),
            "records": self.records,
        }
