"""Model entry points for redback-aux-models."""

redback_model_type = "supernova"

from .nagy_vinko import (
    nagy_vinko,
    nagy_vinko_bolometric,
    nagy_vinko_component_bolometric,
)

__all__ = [
    "nagy_vinko",
    "nagy_vinko_bolometric",
    "nagy_vinko_component_bolometric",
]
