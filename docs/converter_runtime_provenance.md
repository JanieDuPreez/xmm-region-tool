# Converter/runtime provenance

Durable projection evidence records the Python converter/runtime that created the `ProjectionProvenance` snapshot. This is supporting execution provenance and is deliberately separate from semantic projection identity.

## Recorded producer evidence

`ProjectionProvenance.evidence_record()` includes `converter_runtime` with schema `xmm-region-tool.converter-runtime/v1`. The record contains:

- the installed `xmm-region-tool` distribution version;
- a deterministic SHA256 over the installed `xmm_region_tool` Python source files;
- Python implementation and version;
- Astropy version;
- NumPy version;
- `regions` version.

The source hash provides a build/source discriminator for development and editable installs without requiring a `.git` directory or embedding a local checkout path. Installed distribution metadata is the authoritative package-version source; top-level `xmm_region_tool.__version__` is derived from that same metadata rather than a second manually synchronized version string.

If distribution metadata is unavailable in an unsupported raw source-tree import, the explicit fallback is `0+unknown`. Supported installed environments should have distribution metadata.

## Semantic identity and supporting evidence

The converter/runtime record is not included in `projection_identity_sha256`. A runtime upgrade that executes the same explicitly versioned scientific algorithm and produces the same semantic/detector geometry therefore does not automatically invalidate semantic identity. Material algorithm changes remain responsible for changing their method/refinement/schema version.

The runtime record is instead included in the separately integrity-bound supporting-evidence record used by managed FITS-backed artifacts. Recording a runtime version/source hash is provenance; the managed cross-binding of supporting evidence to the FITS artifact is the integrity mechanism.

## Historical evidence

Some retained real-SAS validation predates artifact-embedded converter/runtime provenance. Those observations remain evidence for the SAS/event/geometry behaviour they actually recorded, but they must not be retroactively described as carrying converter provenance that did not exist at the time.

Current release-candidate artifacts generated through the managed path include the converter/runtime record so they can be tied to the installed package/runtime that produced them.

## Release packaging

The external-test release line uses an explicit prerelease distribution version (`0.1.0rc2`). Exact installed source hashing remains useful as an additional source/build discriminator, especially for editable or locally rebuilt environments. The durable provenance contract does not treat the human-readable version string alone as an exact source identifier.
