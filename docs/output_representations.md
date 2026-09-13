# Output representation contract

For the current scientific release, **FITS-backed detector geometry is the only ordinary supported output representation**.

The installed `xmm-region` and `xmm-region-batch` commands therefore accept only `--representation fits`. The top-level `xmm_region_tool.convert_selection_batch()` API does not expose a representation selector at all; it always writes a FITS REGION companion plus the short SAS `region(...)` wrapper.

This is a deliberate scientific/provenance boundary, not merely a preference. The FITS-backed path has retained real-SAS exact-row and ESAS-consumer evidence, carries detector/celestial/projection bindings, and its exact geometry bytes can be checksum-bound independently of the location-dependent text wrapper.

## Direct expression output

Implementation modules retain `selection_expression()` and `write_esas_regionfile(..., representation="expression")` for debugging and comparison. They are **not managed/durable scientific output** and are intentionally absent from the top-level supported API and installed command-line interfaces.

The direct expression serializer formats detector coordinates to six decimal places before SAS consumes them. The resulting text therefore encodes a slightly quantized detector geometry rather than the exact full-precision `DetectorSelection` that participates in the projection identity. The expression file is also not promoted to the managed artifact class or claimed to have FITS-grade byte binding.

`representation="auto"` likewise remains an implementation-module compatibility/debug facility only. It may choose expression based on text length and must not be used as an ordinary scientific output contract.

A future release may promote direct expression output only after it defines and binds the serialized geometry/bytes explicitly and retains focused real-SAS evidence for inclusion, subtractive Boolean composition, and a real `mosspectra`/`pnspectra` consumer path.
