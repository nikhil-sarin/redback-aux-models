import numpy as np

import redback.model_library as model_library
import redback.priors as priors
from redback_aux_models.priors import get_lensing_priors


def test_lensing_wrappers_are_registered_as_base_models():
    for name in [
        "lensing_with_function",
        "lensing_with_supernova_base_model",
        "lensing_with_kilonova_base_model",
        "lensing_with_afterglow_base_model",
    ]:
        assert name in model_library.all_models_dict
        assert name in model_library.base_models_dict


def test_lensing_priors_are_provided_by_plugin():
    prior = priors.get_priors("lensing_three_images")
    assert set(prior) == {"dt_1", "mu_1", "dt_2", "mu_2", "dt_3", "mu_3"}
    generated = get_lensing_priors(nimages=4)
    assert "dt_4" in generated
    assert "mu_4" in generated


def test_supernova_lensing_wrapper_evaluates_builtin_base_model():
    time = np.array([10.0, 20.0, 30.0])
    params = dict(
        base_model="arnett",
        output_format="flux_density",
        frequency=np.ones_like(time) * 4.8e14,
        redshift=0.01,
        f_nickel=0.1,
        mej=2.0,
        vej=5000.0,
        kappa=0.1,
        kappa_gamma=0.03,
        temperature_floor=5000.0,
        dt_1=0.0,
        mu_1=1.0,
        dt_2=5.0,
        mu_2=0.5,
    )
    function = model_library.all_models_dict["lensing_with_supernova_base_model"]
    output = function(time, nimages=2, **params)
    assert output.shape == time.shape
    assert np.all(np.isfinite(output))
