import redback.model_library as model_library
import redback.priors as priors
from redback.transient_models.extinction_models import _get_correct_function


def test_plugin_models_are_registered():
    assert "nagy_vinko" in model_library.all_models_dict
    assert "nagy_vinko_bolometric" in model_library.all_models_dict
    assert "nagy_vinko_component_bolometric" in model_library.all_models_dict
    assert model_library.plugin_module_model_types["redback_aux_models"] == {"supernova"}
    assert model_library.modules_dict["redback_aux_models"] == {
        "nagy_vinko": model_library.all_models_dict["nagy_vinko"],
        "nagy_vinko_bolometric": model_library.all_models_dict["nagy_vinko_bolometric"],
        "nagy_vinko_component_bolometric": model_library.all_models_dict["nagy_vinko_component_bolometric"],
    }


def test_plugin_priors_are_registered():
    prior = priors.get_priors("nagy_vinko_bolometric")
    assert "core_radius" in prior
    assert "shell_radius" in prior
    assert "nickel_mass" in prior


def test_extinction_can_resolve_plugin_model():
    function = _get_correct_function("nagy_vinko")
    assert function is model_library.all_models_dict["nagy_vinko"]
    typed_function = _get_correct_function("nagy_vinko", model_type="supernova")
    assert typed_function is model_library.all_models_dict["nagy_vinko"]
    assert "extinction_with_nagy_vinko_base_model" in model_library.base_models_dict
