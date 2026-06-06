"""Prior provider entry point for redback-aux-models."""

from importlib import resources

from bilby.core.prior import DeltaFunction, LogUniform, PriorDict, Uniform

_PRIOR_FILES = {
    "nagy_vinko": "nagy_vinko.prior",
    "nagy_vinko_bolometric": "nagy_vinko_bolometric.prior",
    "lensing_two_images": "lensing_two_images.prior",
    "lensing_three_images": "lensing_three_images.prior",
    "lensing_four_images": "lensing_four_images.prior",
}


def get_priors(model_name):
    """Return a PriorDict for auxiliary redback models, or None if unknown."""
    filename = _PRIOR_FILES.get(model_name)
    if filename is None:
        return None
    with resources.as_file(resources.files(__package__) / "priors" / filename) as prior_file:
        return PriorDict(filename=str(prior_file))


def get_lensing_priors(nimages=2, dt_min=0.0, dt_max=1000.0, mu_min=0.1, mu_max=100.0):
    """Return a PriorDict for gravitational-lensing delay and magnification parameters."""
    priors = PriorDict()
    for image_index in range(1, nimages + 1):
        dt_name = f"dt_{image_index}"
        mu_name = f"mu_{image_index}"
        if image_index == 1:
            priors[dt_name] = DeltaFunction(0.0, name=dt_name, latex_label=rf"$\Delta t_{image_index}$ (days)")
        else:
            priors[dt_name] = Uniform(dt_min, dt_max, name=dt_name, latex_label=rf"$\Delta t_{image_index}$ (days)")
        priors[mu_name] = LogUniform(mu_min, mu_max, name=mu_name, latex_label=rf"$\mu_{image_index}$")
    return priors
