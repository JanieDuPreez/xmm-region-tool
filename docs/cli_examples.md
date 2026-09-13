# High-level CLI examples

The ordinary workflow is one command that can split a DS9 file into independent extraction cells and project every cell independently for one or more EPIC cameras.

## Current ESAS working directory

The SAS 22 ESAS cookbook identifies the `P-allevc.fits` event list produced by `espfilt` as the filtered event product used for subsequent work. For pn, the current real `espfilt` OOT product spelling is `P-allevcoot.fits`. The older/hyphenated `P-allevc-oot.fits` spelling remains supported as a compatibility form.

Given a working directory containing, for example:

```text
mos1S001-allevc.fits
mos2S002-allevc.fits
pnS003-allevc.fits
pnS003-allevcoot.fits
profile.reg
```

ordinary use can select every unambiguous canonical camera product automatically:

```bash
xmm-region profile.reg
```

The equivalent explicit camera selection is:

```bash
xmm-region profile.reg --instrument mos1 mos2 pn
```

When `--instrument` is omitted, every available MOS1/MOS2/pn camera with exactly one canonical `*-allevc.fits` candidate is selected. Discovery remains fail-closed: more than one canonical candidate for any available camera is an error rather than a guess, and finding no discovered camera products is also an error. Use `--instrument` to restrict automatic discovery to a chosen subset, or `--event-file` to supply exact paths explicitly.

For a DS9 file containing three positive extraction cells, expected wrapper names are:

```text
profile-r001-mos1.txt
profile-r001-mos2.txt
profile-r001-pn.txt
profile-r002-mos1.txt
...
```

Each wrapper has its own camera-specific FITS geometry and provenance sidecar.

For a DS9 `panda(...)`, every angular/radial cell is an independent extraction cell. A DS9 multi-annulus likewise produces one extraction cell per adjacent radial bin. Negative/excluded DS9 regions are applied to each positive extraction cell.

## Explicit event files

Exact event products can always be supplied explicitly:

```bash
xmm-region profile.reg \
  --event-file mos1S001-allevc.fits \
  --event-file pnS003-allevc.fits
```

Explicit event paths bypass canonical filename discovery, so non-canonical but scientifically valid event products remain usable.

## One exact output override

`--output NAME.txt` is retained as a low-level convenience only when exactly one extraction cell and one event product are targeted. Multi-cell or multi-camera runs use deterministic automatic names in `--output-dir`.

## Projection mode

The installed `xmm-region` and `xmm-region-batch` commands use adaptive detector-space refinement only. They do not expose or accept `--samples` in the supported user-facing contract.

Legacy fixed-source sampling remains available only through internal/development APIs for controlled comparisons. It is deliberately narrower than the supported semantic geometry surface and must not be used as evidence for the normal adaptive scientific projection contract. See `docs/legacy_validation.md` for the exact boundary.

## ESAS guidance

Normal output includes the generated region products and the event/region arguments for `mosspectra` or `pnspectra`. For pn, the current `P-allevcoot.fits` sibling is included automatically in the suggested `pnspectra` invocation when present. The compatibility spelling `P-allevc-oot.fits` is also accepted. If both spellings exist for the same science event, OOT discovery fails closed rather than choosing one.

The tool does not invent analysis-dependent CCD/quadrant selections, source-removal choices, image settings or energy settings. Those remain the user's normal ESAS parameters.

## Quiet scripting

```bash
xmm-region profile.reg --instrument mos1 pn --quiet
```

`--quiet` suppresses normal progress and usage guidance. Errors still go to stderr and the exit status remains non-zero on failure.
