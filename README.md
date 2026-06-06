# redback-aux-models

Auxiliary model plugins for redback.

This package registers Nagy-Vinko/LC2 supernova models through redback's plugin entry point system.
Install in editable mode while developing:

```bash
pip install -e /Users/nikhil/Documents/postdoc/redback-aux-models
```

Registered models:

- `nagy_vinko`
- `nagy_vinko_bolometric`
- `nagy_vinko_component_bolometric`

Priors are provided through the `redback.model.priors` plugin entry point.
For extinction, either use `extinction_with_function(base_model="nagy_vinko", ...)` or the plugin wrapper `extinction_with_nagy_vinko_base_model`.

Additional registered base-model wrappers:

- `lensing_with_function`
- `lensing_with_supernova_base_model`
- `lensing_with_kilonova_base_model`
- `lensing_with_afterglow_base_model`
- other typed lensing wrappers matching redback model types

Lensing priors are available as `lensing_two_images`, `lensing_three_images`, and `lensing_four_images`, or by importing `get_lensing_priors` from `redback_aux_models.priors`.
