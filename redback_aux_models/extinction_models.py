"""Extinction wrappers for auxiliary redback models."""

import redback.transient_models.extinction_models as redback_extinction_models
import redback.utils as redback_utils


@redback_utils.citation_wrapper("redback-aux-models")
def extinction_with_nagy_vinko_base_model(time, av_host, **kwargs):
    """Apply redback's generic extinction machinery to the plugin Nagy-Vinko model."""
    kwargs.setdefault("base_model", "nagy_vinko")
    return redback_extinction_models.extinction_with_function(time=time, av_host=av_host, **kwargs)
