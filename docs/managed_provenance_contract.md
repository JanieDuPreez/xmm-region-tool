# Managed provenance completeness

The package distinguishes low-level projection-evidence serialization from the stronger persisted managed-artifact contract.

## Managed artifacts

`write_bound_detector_geometry()` is the supported writer for detector geometry that will be cached, staged or reused by another application. Its sidecar is expected to retain the complete detailed evidence that the managed writer emits. `load_bound_detector_geometry()` therefore fails closed unless that sidecar contains all of the following:

- the canonical projection record and projection identity;
- the detailed calibration record;
- the exact CIF file SHA256 used as captured execution evidence;
- the full ordered `constituent_evidence` audit, including the writer-emitted `name`, `sha256`, `cif_md5` and `size` fields for every selected constituent;
- the detailed SAS producer record;
- the detector-geometry file and semantic geometry bindings;
- the converter/runtime producer record;
- the separately integrity-bound supporting execution/output evidence record.

The managed loader recomputes calibration, producer, context and projection semantic identities rather than trusting copied digest fields.

## Semantic identity versus supporting evidence integrity

`projection_identity_sha256` remains the semantic identity of the requested projection operation and its material inputs. It binds the exact event bytes, semantic event/celestial identities, semantic SAS context/calibration/producer identities and the explicit versioned projection rule. It deliberately does not turn post-execution output or diagnostics into semantic cache inputs.

Managed FITS-backed artifacts therefore carry a distinct `xmm-region-tool.supporting-evidence/v1` record. Its canonical SHA256 is written both to the sidecar and to the FITS REGION header as `XMRGEVD`. The record binds:

- the semantic projection identity it supports;
- the detector-geometry identity produced by the projection;
- actual command/command-history evidence;
- relevant execution-environment evidence;
- achieved refinement diagnostics;
- the historical calibration execution record, including exact CIF evidence;
- the detailed SAS producer record;
- the immutable Python converter/runtime producer record;
- optional original DS9/source-file SHA256 evidence when a source file was supplied.

The exact FITS geometry file SHA256 is recorded after `XMRGEVD` is inserted, so a normal one-file FITS edit is detected by the file digest while a sidecar-only edit of a guaranteed supporting field is detected by the supporting-evidence digest. Reload/staging also cross-check duplicated FITS provenance (`XMRGCIF`, `XMRGCAL`, `ESKYVER`, `SASVERS`, optional `XMRGSHA`, and `XMRGEVD`) against the validated sidecar record.

This is ordinary corruption/stale-state detection through mutually checked records and hashes. It is not a digital signature and does not claim protection against a coordinated hostile rewrite of every mutually checked artifact with newly recomputed hashes.

A matching `projection_identity_sha256` alone therefore does not authenticate the current FITS bytes, detector-output identity or achieved diagnostics. Those claims belong to the complete managed artifact/evidence contract.

## Complete detector-to-celestial source binding

Durable detector geometry is not allowed to become source-anonymous. Before `write_detector_geometry()` writes an artifact that can be promoted to the managed-safe class, the completed `ProjectionResult` must carry the normal projection invariant:

```text
DetectorSelection.source_geometry_sha256
    == ProjectionProvenance.celestial_geometry_sha256
```

Both values must be canonical lower-case 64-hex SHA256 digests. A missing source digest, malformed digest or mismatch fails before any durable geometry is written. Normal `project_selection()` results satisfy this automatically because the detector selection is bound to the input `CelestialSelection.geometry_sha256` at projection time.

This check deliberately belongs to durable-artifact promotion rather than the `DetectorSelection` value type itself. Low-level/debug detector selections may remain source-unbound (`source_geometry_sha256=None`) where useful, and partial projection evidence can still be inspected without pretending to be a managed artifact. What is forbidden is promoting such an unbound detector selection to a reusable artifact that carries `XMRGCEL`/projection provenance as though the source relationship had been established.

The celestial digest identifies semantic source geometry. It is distinct from the detector geometry digest, which identifies the camera/exposure-specific projected tessellation, and neither identity is replaced by the other.

## Calibration evidence and live-context ownership

The exact CIF file SHA256 is execution/frozen-context evidence rather than semantic calibration identity. Two CIF byte streams with identical CALINDEX semantics and identical selected CCF/replacement bytes may therefore have the same calibration identity. A live projection context still fails its endpoint validation if its exact CIF bytes are persistently changed during the projection lifecycle; a previously projected artifact may remain semantically reusable with a later byte-different but semantically equivalent CIF, while its historical stored exact-CIF evidence must remain internally coherent.

This is endpoint drift detection, not an execution-owned immutable calibration copy. The SAS executable, active CIF, selected CCF constituents, explicit replacements and their resolution namespace must remain stable for the complete `project_selection()` call. A concurrent A -> B -> A substitution between SAS calls is outside the package guarantee and can evade pre/post validation. The managed/shared-workflow ownership rule is defined in `docs/sas_context_stability.md`.

## Canonical calibration records

A self-consistent hash is not by itself proof that a detailed calibration block could have been emitted by the real calibration reader. The managed boundary therefore validates the complete canonical record before recomputing `calibration-identity/v2`.

In particular:

- `calindex_sha256`, `cif_file_sha256`, constituent SHA256 values, replacement SHA256 values and the stored calibration identity must be canonical lower-case 64-hex digests;
- canonical constituent/replacement records contain exactly `name` and `sha256`; arbitrary extra or missing fields are rejected;
- CCF names are non-empty basenames, not path-bearing lookalikes;
- selected constituents are unique and basename-sorted, matching `read_calibration_identity()` output;
- replacements are unique but retain the observed `SAS_CCFFILES` order rather than being sorted;
- each constituent audit record contains exactly `name`, `sha256`, `cif_md5` and `size`, and its ordered `(name, sha256)` pair must match the corresponding canonical constituent;
- audit MD5/size values retain their own strict type/digest constraints;
- unexpected top-level calibration-evidence fields are rejected rather than silently becoming unvalidated hash inputs.

The in-memory calibration value objects enforce the reciprocal constructor contract. `CalibrationReplacement` and `CalibrationConstituent` normalize and validate canonical basenames/digests; `CalibrationIdentity` snapshots caller-owned replacement/constituent/search-path iterables into immutable tuples, rejects duplicate basenames, canonicalizes constituent order and normalizes paths/digests. A caller therefore cannot mutate an input list after construction and change the identity of a supposedly frozen `CalibrationIdentity` or `SasProjectionContext`.

Persisted JSON remains independently validated even though package-created objects are canonical: files are an untrusted boundary and are not assumed to have passed the constructors that originally wrote them.

## Public event-binding boundary

The top-level managed-safe API does not require callers to construct or import `EventIdentity`. Ordinary callers supply `event_file=` to `write_bound_detector_geometry()` and `materialize_bound_sas_regionfile()`. Each operation re-reads that path through the canonical science-event role and requires both the resulting semantic event identity and the current exact-file SHA256 to match the authoritative projection/artifact binding before delegating to the managed implementation.

This stronger writer closes the forgery gap that would exist if a caller-constructed `EventIdentity` and a caller-constructed `ProjectionResult` could merely present matching digest fields. The event path is evidence location, not part of semantic event or projection identity.

For compatibility, the top-level functions still accept the historical `event_identity=` spelling. That object is not trusted as semantic evidence: only its `.path` is retained, and identity plus exact bytes are re-derived from the file before promotion or reuse. A forged instrument, ObsID, exposure or pointing value in the supplied object therefore cannot become a managed FITS binding.

As with the projection lifecycle itself, these repeated path checks fail closed on persistent generation changes but do not claim immutable-copy isolation from a concurrent ABA replacement that changes and restores the same path between checks. The narrower caller-owned path-stability contract is documented in `docs/science_event_contract.md`.

## Low-level/debug evidence

`write_projection_evidence()` is a lower-level serialization helper. Callers may omit `calibration_identity` and/or `sas_producer`, in which case the resulting sidecar intentionally contains only the evidence that was supplied. This is useful for debugging and inspection, but such a partial sidecar does **not** satisfy the managed persisted-artifact contract and must not be accepted by `load_bound_detector_geometry()` as a fully provenance-bound artifact.

The complete FITS-backed path with calibration and SAS-producer evidence also emits the supporting-evidence record described above. Partial/debug serializers do not fabricate that guarantee when required source records are absent.

## Projection-time ownership of exact calibration evidence

The semantic calibration identity intentionally excludes the exact CIF file SHA256 when byte differences do not change CALINDEX semantics or the selected CCF/replacement bytes. Exact CIF bytes are instead captured execution evidence.

That distinction does not permit historical execution evidence to be replaced later. `project_selection()` therefore snapshots the complete `CalibrationIdentity.evidence_record()` into immutable `ProjectionProvenance` at projection time. This snapshot is deliberately excluded from `canonical_projection_record()` and therefore does not change semantic `projection/v3` identity.

When detector geometry is materialised with calibration metadata, any caller-supplied `CalibrationIdentity` must match both the semantic identity and the complete projection-time evidence snapshot. The persisted managed sidecar uses the authoritative snapshot from the `ProjectionResult`, rather than reconstructing historical evidence from a later semantically-equivalent calibration object.

Consequently, two CIF files may have the same semantic calibration and projection identities while a result actually produced under CIF A cannot later be labelled as having executed under CIF B. During later reuse of an already-projected artifact, a semantically equivalent current CIF B is allowed; the historical A evidence remains fixed inside the validated artifact/evidence pair rather than being relabelled as B.
