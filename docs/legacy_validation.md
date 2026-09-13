# Legacy projection and X/Y-WCS diagnostic policy

For the current release, legacy fixed-source sampling is **not** part of the ordinary scientific command-line contract.

The installed conversion commands are:

```text
xmm-region
xmm-region-batch
```

Both use the current adaptive, versioned `project_selection()` projection contract. They do not accept `--samples`; passing that option is an argument error. FITS-backed detector geometry remains the ordinary scientific output representation.

The low-level implementation modules still retain `ProjectionRule(samples=...)` and fixed-sampling helpers for controlled developer comparisons and historical investigations. Those code paths do not define the supported release CLI behaviour and are not guaranteed to support every first-class semantic path such as sector/panda/epanda geometry.

The historical `xmm_region_tool.validate_cli` X/Y-WCS comparison code is likewise retained as developer/scientific diagnostic code, but **`xmm-region-validate` is not installed as a console script**. It uses the older source/fixed-sampling path and must not be interpreted as validation of current adaptive `xmm-region` output.

The independent event X/Y-WCS comparison remains scientifically useful for studying coordinate-realisation differences. It is not the detector-space projection-accuracy metric and is not a substitute for the release validator.

The installed real-SAS consumer validation command is:

```text
xmm-region-sas-validate
```

It validates the generated FITS REGION artifact through real `evselect`, compares exact selected row identity/order, and remains fail-closed when SAS and the package ASC reference evaluator disagree.

A future release may promote a modernised X/Y-WCS diagnostic back into the installed surface only if it consumes the same current `CelestialSelection`, source snapshot, adaptive/versioned projection rule and frozen SAS context as ordinary conversion. Until then, legacy fixed-sampling code is deliberately an internal/developer boundary.
