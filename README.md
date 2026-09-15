# xmm-region-tool

`xmm-region-tool` converts celestial extraction regions into **observation-, exposure- and camera-specific XMM-Newton EPIC DETX/DETY region products** for SAS/ESAS tasks such as `mosspectra` and `pnspectra`.

## Release status

The current public-test release line is `0.1.0rc2`. It is intended for external testing before the first stable `0.1.0` release.

After the release candidate is published to PyPI, install it with:

```bash
pip install --pre xmm-region-tool
```

or pin the exact candidate:

```bash
pip install xmm-region-tool==0.1.0rc2
```

Python 3.10 or newer is required. Scientific conversion also requires an initialized XMM-Newton SAS environment with valid calibration/observation context.

## Why this exists

A celestial region cannot be reused as one DETX/DETY region for every EPIC product. The sky-to-detector mapping depends on the exact observation, exposure, instrument, pointing and calibration context. Transforming only the region centre is also insufficient because size, orientation and the complete boundary must follow the calibrated mapping.

The tool does not use `convregion` at runtime. It uses SAS `esky2det` for the calibrated sky-to-detector transform and materialises complex detector selections as FITS Spatial Region tables.

## Quick start

Given an ESAS working directory containing a DS9 file and unambiguous filtered camera products such as:

```text
mos1S001-allevc.fits
mos2S002-allevc.fits
pnS003-allevc.fits
pnS003-allevcoot.fits
profile.reg
```

run:

```bash
xmm-region profile.reg
```

When neither `--event-file` nor `--instrument` is supplied, the command selects every available MOS1/MOS2/pn camera with exactly one canonical `*-allevc.fits` candidate. Ambiguity within a camera, or finding no canonical products, fails closed rather than guessing.

Restrict automatic discovery explicitly when needed:

```bash
xmm-region profile.reg --instrument mos1 mos2 pn
```

Or provide exact event products directly:

```bash
xmm-region profile.reg \
  --event-file mos1S001-allevc.fits \
  --event-file pnS003-allevc.fits
```

Explicit event paths bypass filename discovery, so scientifically valid non-canonical event filenames remain usable.

For pn, current real `espfilt` products use the OOT sibling spelling `P-allevcoot.fits`. The older `P-allevc-oot.fits` spelling remains supported for compatibility; if both spellings are present for the same science event, discovery fails closed.

See `docs/cli_examples.md` for more examples.

## Extraction cells and outputs

The ordinary `xmm-region` workflow treats a DS9 file as a set of science extraction cells:

- each positive top-level region is an independent cell;
- a circular or elliptical multi-annulus produces one cell per adjacent radial bin;
- `panda(...)` and `epanda(...)` produce their angular × radial cells independently;
- negative/excluded regions are applied to every positive extraction cell;
- an exclusion-only selection remains one selection;
- `--combine` explicitly requests the full union selection.

Every `(cell, event file)` pair is projected independently. Detector geometry is never reused between MOS1, MOS2, pn, exposures or observations.

For a file called `profile.reg`, automatic products are named deterministically, for example:

```text
profile-r001-mos1.txt
profile-r001-mos1.fits
profile-r001-mos1.provenance.json
profile-r001-mos2.txt
profile-r001-pn.txt
profile-r002-mos1.txt
...
profile-xmm-region-manifest.json
```

The short `.txt` file is the SAS region wrapper; the FITS file contains the detector geometry; the provenance sidecar records the projection evidence. `--output-dir` controls automatic output placement. `--output NAME.txt` is available when exactly one extraction cell and one event product are targeted.

The default CLI output is compact. Use `--verbose` for full paths, detector-refinement diagnostics, manifest paths and copy/paste ESAS guidance; use `--quiet` for scripting.

## Supported celestial geometry

The supported source surface includes:

- circle;
- ellipse;
- box/rectangle;
- polygon;
- circular annulus and DS9 circular multi-annulus;
- elliptical annulus and DS9 multi-ring elliptical annulus;
- rectangular/box annulus;
- circular sector annulus;
- elliptical sector annulus;
- DS9 `panda(...)` and `epanda(...)` angular/radial grids;
- DS9 include/exclude regions, including exclusion-only selections.

`bpanda(...)` is not currently supported and is rejected explicitly rather than approximated silently. Unsupported or ambiguous DS9 semantics likewise fail closed instead of producing a partial science selection.

See `docs/ds9_semantics.md` and `docs/detector_topology.md` for the geometry contracts.

## Projection model

Celestial science geometry is kept separate from detector tessellation:

```text
semantic celestial selection
    -> explicit projection/refinement rule
    -> detector polygon geometry
```

The installed scientific CLIs use the versioned `adaptive-detector-chord/v2` rule. For each semantic boundary interval, the converter projects the endpoints and true interior boundary points at fractions 1/4, 1/2 and 3/4 through `esky2det`. It measures each projected probe's perpendicular detector-space distance from the endpoint chord and recursively bisects until the largest sampled probe error is at or below the requested tolerance.

The default tolerance is **1 native DET unit = 0.05 arcsec**. The reported error is the explicit sampled three-probe estimator, not a claim of an analytic continuous maximum between probes.

The converter fails explicitly if the criterion cannot be met within its bounded refinement limits or if projection produces non-finite, degenerate, self-intersecting or topology-changing detector geometry.

Legacy fixed-source sampling remains internal/development-only and is not part of the installed scientific CLI contract. See `docs/legacy_validation.md`.

## Calibration, event identity and provenance

A projected detector artifact is tied to the exact science context that produced it. The semantic projection identity binds, among other things:

- exact event-file bytes plus semantic event identity;
- semantic celestial geometry;
- path-independent calibration/CALINDEX semantics;
- exact selected CCF/replacement bytes;
- SAS/`esky2det` producer identity;
- the explicit versioned projection rule.

Managed FITS-backed artifacts also integrity-bind the emitted detector geometry and supporting execution evidence. This is designed for scientific reproducibility, stale-state detection and ordinary corruption/replacement detection; it is not a digital signature and does not claim protection against a hostile same-machine writer able to replace and restore all trusted files during execution.

See:

- `docs/science_event_contract.md`;
- `docs/context_suitability.md`;
- `docs/calindex_semantics.md`;
- `docs/managed_provenance_contract.md`;
- `docs/sas_context_stability.md`;
- `SECURITY.md`.

## Managed Python API

The top-level package exposes the supported managed workflow. For example:

```python
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

from xmm_region_tool import (
    SasProjectionContext,
    circle_selection,
    materialize_bound_sas_regionfile,
    project_selection,
    write_bound_detector_geometry,
)

event_file = Path("mos1S001-allevc.fits")
context = SasProjectionContext.from_environment()
selection = circle_selection(
    SkyCoord(10.0 * u.deg, -9.0 * u.deg, frame="icrs"),
    30.0 * u.arcsec,
)

projected = project_selection(
    selection,
    calinfoset=event_file,
    context=context,
)

bound = write_bound_detector_geometry(
    "managed-region.fits",
    projected,
    event_file=event_file,
    calibration_identity=context.calibration,
    sas_producer=context.producer,
)

materialize_bound_sas_regionfile(
    "managed-region.txt",
    bound,
    event_file=event_file,
    calibration_identity=context.calibration,
    sas_producer=context.producer,
    expected_celestial_geometry_sha256=selection.geometry_sha256,
    expected_projection_identity_sha256=projected.provenance.projection_identity_sha256,
)
```

Persisted managed artifacts can be reloaded with `load_bound_detector_geometry()` before reuse.

## Validation and known SAS boundary limitation

`xmm-region-sas-validate` is the real-SAS consumer-fidelity validator. It compares exact selected event-row identity/order between the package's ASC reference semantics and real `evselect`, rather than accepting matching row counts alone.

Real SAS 22.1.0 validation covers MOS1/MOS2/pn, multiple observations, central/off-axis/FOV-crossing regions, rotated ellipses and boxes, polygons, annuli, include/exclude forms, circular and elliptical sector products, `panda(...)` and `epanda(...)`. Generated wrappers have also been consumed successfully by real `mosspectra`/`mosback` and `pnspectra`/`pnback` chains.

A focused boundary test identified a SAS 22.1.0 difference at deliberately exact serialized polygon vertices: formal ASC/package semantics and SAS agree for tested interiors and non-vertex edges, but SAS gives the opposite decision for the tested first outer vertex and first subtractive-hole vertex. Re-encoding the same polygon without the redundant repeated closing coordinate did not change the SAS behaviour. The package therefore keeps formal ASC semantics authoritative and the validator fails closed on an exact-row mismatch rather than perturbing geometry to imitate the SAS quirk.

See `docs/real_sas_validation.md` for the retained evidence and limits of the claim.

## Path and resource safety

SAS/selectlib dataset syntax is handled conservatively. Paths supplied to SAS are restricted to a tested safe resolved-path character class rather than being quoted or escaped speculatively. Generated publication is symlink-aware and atomic, output namespaces reject aliases/collisions with protected inputs, subprocess diagnostics are bounded, and terminal rendering sanitises untrusted control characters.

The optional real-SAS validator also bounds accepted event-table workloads before materialising the table. See `docs/sas_path_safety.md`, `docs/subprocess_diagnostics.md`, `docs/terminal_output_safety.md`, and `docs/validator_event_workload_evidence_2026-09-13.md`.

## Development

```bash
uv sync --locked --extra test
uv run --locked pytest
uv run --locked ruff check src tests
```

The repository CI additionally checks Python 3.10 and 3.13, the declared minimum scientific dependency stack, a freshly built wheel outside the source checkout, and focused security regressions.

Real SAS behaviour must be validated in an initialized SAS environment. Mocked SAS tests validate package orchestration and geometry logic only; they are not evidence about SAS itself.

## Security

Please follow `SECURITY.md` for private vulnerability reporting and for the generated-artifact/privacy and trusted-filesystem boundaries.
