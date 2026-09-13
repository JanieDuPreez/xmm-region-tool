# Real-SAS validation contract and retained evidence

This document records the public acceptance contract and retained real-SAS evidence for `xmm-region-tool`.

Mocked SAS tests, synthetic FITS tests and code inspection validate package logic only. Claims about SAS behaviour below come from real SAS 22.1.0 runs on A3376 and A133.

## Core invariants

For one semantic celestial selection and one exact EPIC event product:

1. detector geometry belongs only to that exact observation/exposure/instrument event artifact;
2. projection accuracy is defined by an explicit detector-space error rule, not by tuning event counts;
3. event X/Y-WCS membership is an independent diagnostic, not the authoritative projection metric;
4. real SAS `selectlib`/`evselect` interpretation is compared against the package reference evaluator by exact selected-row identity, and any disagreement is fail-closed rather than hidden by matching counts;
5. `mosspectra`/`pnspectra` must accept the generated short wrapper and FITS geometry directly;
6. FOV and projection-domain behaviour must be demonstrated without silent clipping or invented coordinates;
7. calibration and producer identity must be reproducible and bound to the artifact.

## Projection contract

The default projection path is adaptive detector-space chord refinement.

### Current estimator: `adaptive-detector-chord/v2`

For every semantic celestial boundary interval `[a,b]`:

1. project its endpoints with `esky2det`;
2. evaluate the true semantic celestial boundary at dyadic interior fractions `1/4`, `1/2` and `3/4`;
3. project those probes;
4. measure each probe's Euclidean perpendicular distance from the detector-space endpoint chord;
5. accept the interval only if the largest sampled probe error is at or below tolerance;
6. otherwise bisect and repeat, reusing cached dyadic projections.

The diagnostics are:

```text
interior_probe_fractions = [0.25, 0.5, 0.75]
max_accepted_probe_error
max_observed_probe_error
```

They describe the sampled estimator, not an analytic proof of the continuous maximum between probes. The v2 rule removes the known midpoint-only blind spot in which endpoints and midpoint can lie on one detector chord while quarter points deviate.

The default tolerance is:

```text
detector_tolerance = 1.0 native DET unit = 0.05 arcsec
```

The converter fails explicitly when the requested criterion cannot be met within configured depth/vertex limits or when projection produces non-finite, degenerate, self-intersecting or topology-changing detector geometry.

`--samples N` is legacy/debug functionality and is not part of the installed release CLI surface. See `docs/legacy_validation.md`.

### Historical v1 evidence

The older midpoint-only `adaptive-detector-chord/v1` matrix remains historical evidence only and is not relabelled as v2 validation. A central A3376 MOS1 circle gave:

```text
2.0 / 1.0 DET -> 128 vertices, max accepted midpoint error ~0.723 DET
0.5 / 0.25 DET -> 256 vertices, max accepted midpoint error ~0.181 DET
0.125 DET -> 512 vertices, max accepted midpoint error ~0.045 DET
```

### Current v2 real-SAS evidence

The focused v2 gate used A133 ObsID `0723802001` with real MOS1 S001, MOS2 S002 and pn U002 ESAS event products. For each camera, 2-arcmin circles were projected at the pointing centre, 8 arcmin off axis and 12.5 arcmin near the FOV edge at 1.0, 0.5 and 0.25 DET tolerances.

Across all nine camera/position cases:

- 1.0 DET used 128 vertices with `max_accepted_probe_error` between `0.722981` and `0.723719` DET;
- 0.5 DET used 256 vertices with `max_accepted_probe_error` between `0.180881` and `0.181541` DET;
- 0.25 DET produced the same 256-vertex geometry as 0.5 DET because the accepted residual was already below 0.25 DET;
- the symmetric 1.0 -> 0.5 polyline difference was `0.722981` to `0.723719` DET;
- the symmetric 0.5 -> 0.25 difference was exactly `0.0` DET;
- no central, off-axis, FOV-near or camera-specific topology/projection failure occurred.

The 1.0 -> 0.5 refinement is nested and the identical 0.5/0.25 geometry is a stable refinement fixed point. This is the retained empirical sanity check for `adaptive-detector-chord/v2` over the tested XMM projection domain.

## Independent DS9 source semantics

Real-SAS consumer tests begin after DS9 text has been converted into semantic celestial paths, so source-format semantics are validated separately.

Controlled TAN-WCS tests establish that, for tested XMM-scale parametric geometry through 30 arcmin, the package's WCS-independent celestial interpretation agrees with DS9's local/reference-plane construction to better than 1 native DET unit for circle, rotated ellipse and rotated box. Additional tests cover FK5 polygon edge behaviour, a Galactic rotated ellipse and frame-angle handling, and deliberately multi-degree stress geometry.

DS9 `panda(...)` semantics are grounded in DS9's implementation/documentation: angle wrap, equal start/stop meaning 360 degrees, and `nangle × nradius` cell decomposition are implemented explicitly.

A rotation-sign defect discovered during validation invalidated the earlier **source-semantic** acceptance of rotated ellipse/box/epanda cases while leaving the corresponding SAS-consumer observations intact. Fresh post-correction source-semantic validation passed:

- asymmetric MOS1 `epanda`: visual source match + exact downstream SAS rows `4229/4229`;
- asymmetric pn `epanda`: visual source match + exact downstream SAS rows `9771/9771`;
- rotated non-square MOS1 box: visual source match + exact downstream SAS rows `27331/27331`;
- ordinary rotated ellipse and elliptical-annulus visual checks passed.

The accepted directed DS9 conversion is `theta_native = 180 deg - theta_DS9`. For undirected ellipse/box axes, canonical `-theta_DS9` is equivalent modulo 180 degrees; non-full `epanda` retains the directed-axis distinction.

See `docs/ds9_semantics.md` for the current public source-language contract and its limits.

## Observation/camera matrix

Retained real-data validation includes:

- A3376 ObsID `0151900101`: MOS1, MOS2 and pn;
- A3376 ObsID `0504140101`: a second pointing/attitude;
- A133 ObsID `0144310101`: MOS1 and pn comparison/validation cases;
- A133 ObsID `0723802001`: MOS1, MOS2 and pn adaptive-v2 convergence cases.

Detector products from different cameras, exposures or observations are never treated as interchangeable.

## Shape and Boolean-composition matrix

Historical real-SAS exact-row consumer validation includes at least one of each required composition:

- circle;
- rotated ellipse;
- rotated box/rectangle;
- arbitrary polygon;
- circular annulus;
- elliptical annulus;
- rectangular/box annulus;
- two disjoint inclusions;
- inclusion plus exclusion;
- exclusion-only selection;
- excluded annulus Boolean expansion;
- central sector annulus;
- off-axis/rotated sector annulus;
- multi-cell DS9 panda grid;
- elliptical-profile / asymmetric `epanda` products.

For rotated DS9 shapes, older exact-row results are consumer-fidelity evidence only; source-semantic acceptance uses the corrected directed-angle checks above.

Representative historical v1 A3376 `0151900101` / MOS1 S007 results include:

- elliptical annulus: max accepted midpoint error `0.98439958` DET; expected/SAS `24074/24074`; exact row sequence yes;
- rectangular/box annulus: max accepted midpoint error `0.00013210` DET; expected/SAS `25008/25008`; exact row sequence yes;
- two disjoint inclusions: max accepted midpoint error `0.60202648` DET; expected/SAS `18750/18750`; exact row sequence yes;
- central wrapped sector annulus (`330° -> 60°`): `10655/10655`; exact row sequence yes;
- off-axis rotated sector annulus: `7295/7295`; exact row sequence yes;
- 2×2 panda grid: `23660/23660`; exact row sequence yes.

These observations are retained for the rows and artifacts actually tested. They are not evidence that SAS reproduces formal ASC boundary semantics at every mathematically exact polygon vertex.

## Calibrated geometry and topology

The current adaptive-v2 projection criterion has real MOS1/MOS2/pn central/off-axis/FOV-near convergence evidence as described above.

Topology validation independently rejects invalid hole relationships before projection geometry is returned: a hole wholly outside its outer component, a subtractive boundary containing its outer component, nested/redundant internal holes, touching boundaries and malformed detector polygons fail explicitly. See `docs/detector_topology.md`.

## Independent event X/Y-WCS diagnostic

The historical/internal X/Y-WCS diagnostic formerly exposed as `xmm-region-validate` compares source membership through the event-list X/Y WCS against detector geometry. It is not installed in the current release and remains diagnostic only.

A3376 showed a real field-wide difference between the event X/Y-WCS sky realisation and SAS DET->sky coordinates, with median separation about 0.156 arcsec in a representative sample. Zero X/Y-WCS row disagreement is therefore not a release requirement and must not be forced with an arbitrary event-count tolerance. See `docs/legacy_validation.md`.

## Real-SAS FITS REGION interpretation

The installed validator tests consumer fidelity for detector rows presented to SAS:

```bash
xmm-region-sas-validate detector-region.fits \
  --event-file EVENTS.fits \
  --source-region REGION.reg
```

A compatible result requires:

- expected selected count equals SAS selected count;
- selected event-row sequence matches exactly;
- compared science columns are unchanged;
- include/exclude and Boolean expansion behave as intended.

A matching count with different rows is a failure.

### Exact polygon-vertex limitation in SAS 22.1.0

Focused A133 ObsID `0723802001` / MOS1 S001 testing found a deliberate boundary case where real SAS 22.1.0 does not reproduce the formal ASC-FITS-REGION rule at exact serialized polygon vertices.

The six deliberate cases were:

| case | ASC/package | SAS 22.1 |
| --- | --- | --- |
| shell interior | selected | selected |
| outside outer | rejected | rejected |
| outer edge, not vertex | selected | selected |
| outer vertex | selected | **rejected** |
| hole edge, not vertex | rejected | rejected |
| hole vertex | rejected | **selected** |

The package expected case indices `[0, 2, 3]`; SAS selected `[0, 2, 5]`. Counts happened to match (`3/3`), but exact row identity failed and the validator correctly reported incompatibility.

Accepted producer evidence for this behaviour is:

```text
SAS: 22.1.0-a8f2c2afa-20250304
evselect: evselect-3.71.3 [22.1.0-a8f2c2afa-20250304]
evselect executable SHA256: 4222a9250982b64f18edeea512253e5a51ee0ac57cb7169e5f6349d52b91f77a
producer identity SHA256: 6b5df03ea52f91c9935558046e2bf93e49735d30d951ef044eb51dc760f07549
```

A representation-only experiment then tested whether the production writer's redundant repeated closing vertex caused the discrepancy. Re-encoding the same polygons with unique vertices did **not** alter either side: package membership remained `[0, 2, 3]` and SAS remained `[0, 2, 5]`. The source event SHA256 was unchanged before and after the probe.

Therefore redundant closure serialization is not the cause, and unique-vertex serialization is not a justified mitigation. Production serialization remains unchanged. Formal ASC semantics remain authoritative for the package evaluator; the validator remains exact-row and fail-closed; no epsilon expansion/shrinkage, event-count tuning or vertex perturbation is introduced.

The retained consumer evidence is consequently complete for the rows and shape matrix actually tested, with this explicit SAS 22.1.0 exact-vertex limitation. It is not a claim of universal exact-vertex agreement between SAS and ASC-FITS-REGION semantics.

### Artifact integrity is a separate contract

Exact-row agreement does not prove that a FITS file is still the exact detector artifact originally produced. For generated products with a supported manifest, use:

```bash
xmm-region-check detector-region.fits \
  --event-file EVENTS.fits \
  --manifest xmm-region-manifest.json
```

The manifest-backed checker verifies the current FITS SHA-256 plus detector/supporting-evidence bindings against the projection-evidence sidecar, then checks event/celestial/projection bindings. Header-only checking remains useful for diagnostics but explicitly does not verify geometry bytes.

See `docs/managed_provenance_contract.md` for the durable artifact/evidence contract.

## ESAS consumer tasks

Real consumer chains completed successfully:

```text
MOS: mosspectra -> mosback
pn:  pnspectra  -> pnback
```

Generated FITS-backed wrappers were accepted directly; `convregion` and giant command-line expressions were not required.

The high-level extraction-cell workflow was exercised on A133 across MOS1, MOS2 and pn: one DS9 two-bin annulus was automatically split into two cells, independently projected for all three cameras, validated against manifest bindings, and generated `r001` wrappers completed real `mosspectra`/`pnspectra` runs successfully.

A later representative exact-candidate integration smoke used A133 ObsID `0723802001` MOS1 S001. A two-cell DS9 annulus converted through the high-level `xmm-region` CLI, `xmm-region-check` verified the bound artifact/evidence chain, and real SAS 22.1.0 `mosspectra` consumed the generated `r001` wrapper successfully with exit 0 and normal spectrum/response/image products. This is a direct ESAS consumer smoke; it does not redefine any earlier exact-row evidence.

## FOV and projection domain

The converter calls `esky2det` with `checkfov=no`.

Real A3376 tests demonstrate that ordinary FOV-crossing geometry can return a complete finite detector boundary without clipping. One deliberate MOS1 edge-crossing circle selected `13385/13385` exact rows under real SAS.

`checkfov=no` does not remove the telescope-coordinate mathematical domain. Real SAS 22.1.0 raises `thetaGreaterThan90` for sufficiently out-of-domain sky coordinates. Package behaviour is fail-closed: non-zero exit, named SAS cause preserved, and no false detector/provenance artifact produced.

## SAS dataset/blockspec path handling

The current release uses unquoted SAS DAL/selectlib path syntax only for the deliberately conservative resolved-path class:

```text
[A-Za-z0-9._/-]+
```

Science event/calinfoset paths, package-created SAS temporary datasets, FITS REGION paths in `region(<blockspec>,DETX,DETY)`, managed wrapper geometry paths and real-`evselect` validation datasets are checked against the same policy before SAS consumption. Whitespace and structural/unvalidated characters such as `+`, `:`, `[`, `]`, `,`, `(` and `)` fail closed rather than being guessed safe. See `docs/sas_path_safety.md`.

## Calibration and producer provenance

Semantic projection/calibration identity binds:

- exact event-file SHA256 plus semantic event identity;
- semantic celestial geometry;
- path-independent CALINDEX semantics;
- exact SHA256 of every CIF-selected CCF constituent;
- explicit replacement constituent bytes;
- exact `esky2det` executable bytes/version and SAS release;
- the actual projection/refinement rule.

The exact CIF file SHA256 is recorded separately as execution evidence. A frozen `SasProjectionContext` revalidates the captured calibration and producer state immediately before and after projection, subject to the caller-owned lifecycle boundary documented in `docs/sas_context_stability.md`.

A real A133 audit resolved all 128 active CIF constituents with zero missing files, zero MD5 mismatches and zero FSIZE mismatches. A subsequent real conversion using the resulting calibration identity selected the same `5450/5450` MOS1 B5 rows exactly.

See `docs/calindex_semantics.md`, `docs/context_suitability.md`, `docs/science_event_contract.md`, and `docs/converter_runtime_provenance.md` for the public identity/evidence contracts.

## A133 legacy annulus comparison

The generic calibrated boundary projection was compared with an older approximation that projected only the annulus centre and used fixed circular DET radii of 1200 DET units per arcmin.

For B5/B6/Lmid the two methods were extremely close but not mathematically identical. The generic method preserved exact partition closure while changing only boundary-level rows:

- MOS1 Lmid: 4 rows changed versus the legacy approximation;
- pn Lmid: 9 rows changed versus the legacy approximation.

This supports the calibrated celestial-boundary method as the exact source-geometry realisation while quantifying the previous approximation rather than treating it as incorrect science by fiat.

## Managed artifact reuse

For managed/library reuse, the recommended path is:

```text
project_selection(...)
  -> ProjectionResult
  -> write_bound_detector_geometry(...)
  -> BoundDetectorGeometryArtifact
  -> materialize_bound_sas_regionfile(...)
```

The wrapper is not written until exact event bytes, semantic event identity, calibration identity, producer identity, geometry bytes, requested celestial science-cell identity, projection identity, FITS binding headers and projection-evidence sidecar all agree. Persisted managed artifacts are rehydrated through `load_bound_detector_geometry(...)` rather than by reconstructing trusted binding metadata manually.

See `docs/managed_provenance_contract.md` and `docs/managed_execution_contract.md`.

## Validation summary

The retained release evidence establishes:

- independent DS9 semantics with documented source-WCS limitations;
- adaptive-v2 1-DET projection with focused real MOS1/MOS2/pn central/off-axis/FOV-near convergence evidence;
- exact event/calibration/producer binding;
- real shape/camera/observation consumer evidence;
- exact-row real-SAS validation that fails closed, including the documented SAS 22.1.0 exact-vertex limitation rather than claiming universal exact-vertex agreement;
- successful MOS and pn ESAS consumer chains;
- fail-closed FOV/domain behaviour;
- a conservative tested SAS path grammar;
- explicit separation of mocked software tests from real-SAS scientific evidence.

These claims are intentionally limited to the behaviours and evidence recorded above; they do not turn synthetic tests into SAS evidence or generalise exact-row observations beyond the products and rows actually tested.
