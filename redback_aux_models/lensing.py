"""Gravitational-lensing wrapper models for redback plugins."""

import numpy as np

import redback.sed as redback_sed
import redback.transient_models.extinction_models as redback_extinction_models
import redback.utils as redback_utils

_MODEL_TYPES = (
    "afterglow",
    "general_synchrotron",
    "integrated_flux_afterglow",
    "kilonova",
    "magnetar_driven",
    "shock_powered",
    "stellar_interaction",
    "supernova",
    "tde",
)


def _strip_lensing_parameters(kwargs, nimages):
    clean_kwargs = kwargs.copy()
    for image_index in range(1, nimages + 1):
        clean_kwargs.pop(f"dt_{image_index}", None)
        clean_kwargs.pop(f"mu_{image_index}", None)
    return clean_kwargs


def _perform_lensing(time, flux_density_function, nimages, **kwargs):
    """Sum delayed and magnified image fluxes from a base model evaluator."""
    time = np.atleast_1d(time)
    lensed_output = np.zeros_like(time, dtype=float)

    for image_index in range(1, nimages + 1):
        dt = kwargs.get(f"dt_{image_index}", 0.0)
        mu = kwargs.get(f"mu_{image_index}", 1.0 if image_index == 1 else 0.0)
        shifted_time = time - dt
        valid = shifted_time > 0
        if np.any(valid):
            lensed_output[valid] += mu * np.atleast_1d(flux_density_function(shifted_time[valid]))

    return lensed_output


def _evaluate_lensing_model(time, nimages=2, model_type=None, **kwargs):
    """Evaluate a base model with multiple gravitationally-lensed images."""
    base_model = kwargs["base_model"]
    if base_model in ["thin_shell_supernova", "homologous_expansion_supernova"]:
        kwargs["base_model"] = kwargs.get("submodel", "arnett_bolometric")

    temp_kwargs = _strip_lensing_parameters(kwargs, nimages)
    function = redback_extinction_models._get_correct_function(base_model=base_model, model_type=model_type)

    if kwargs["output_format"] == "flux_density":
        temp_kwargs["output_format"] = "flux_density"

        def evaluate_base_model(shifted_time):
            return function(shifted_time, **temp_kwargs)

        return _perform_lensing(
            time=time,
            flux_density_function=evaluate_base_model,
            nimages=nimages,
            **kwargs,
        )

    temp_kwargs["output_format"] = "spectra"
    time_obs = time
    spectra_tuple = function(np.atleast_1d(time), **temp_kwargs)

    total_mu = 0.0
    for image_index in range(1, nimages + 1):
        total_mu += kwargs.get(f"mu_{image_index}", 1.0 if image_index == 1 else 0.0)

    # Redback spectra-mode models are evaluated on their own internal time grid.
    # We therefore apply the total magnification in spectra mode. Use
    # flux_density mode when the individual image time delays matter.
    return redback_sed.get_correct_output_format_from_spectra(
        time=time_obs,
        time_eval=spectra_tuple.time,
        spectra=total_mu * spectra_tuple.spectra,
        lambda_array=spectra_tuple.lambdas,
        **kwargs,
    )


@redback_utils.citation_wrapper("redback-aux-models")
def lensing_with_function(time, nimages=2, **kwargs):
    """Apply gravitational lensing to any base model discoverable by redback."""
    return _evaluate_lensing_model(time=time, nimages=nimages, model_type=None, **kwargs)


def _make_lensing_wrapper(model_type):
    @redback_utils.citation_wrapper("redback-aux-models")
    def wrapper(time, nimages=2, **kwargs):
        return _evaluate_lensing_model(time=time, nimages=nimages, model_type=model_type, **kwargs)

    wrapper.__name__ = f"lensing_with_{model_type}_base_model"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = f"Apply gravitational lensing to {model_type} base models."
    return wrapper


for _model_type in _MODEL_TYPES:
    globals()[f"lensing_with_{_model_type}_base_model"] = _make_lensing_wrapper(_model_type)


__all__ = ["lensing_with_function"] + [
    f"lensing_with_{model_type}_base_model" for model_type in _MODEL_TYPES
]
