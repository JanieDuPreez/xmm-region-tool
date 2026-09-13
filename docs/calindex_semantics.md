# CALINDEX semantic identity

This document defines which parts of a SAS Calibration Index File belong to the path-independent scientific calibration identity. File-resolution logic determines which constituent bytes CAL resolves; this contract defines which CALINDEX row fields are semantic once that active set is known.

## Real SAS 22.1.0 CIF evidence

A real A133 CIF produced for the supported SAS 22.1.0 environment was inspected read-only. Its `CALINDEX` table has 128 rows and the current column spelling/type is:

- `EXTSEQU`: `32A`;
- `EXTSEQID`: `256A`;
- `CREATOR`: `64A`;
- no `EXTSEQUID` column.

Representative active rows contain `EXTSEQID` values such as `LINCOORD FOV CORNER`, `BORESIGHT OM_ANGVAR EMOS1_ANGVAR ...`, and `MISCDATA`. These are extension-content identifiers, not presentation metadata. `CREATOR` values instead identify tooling/path/version provenance.

Current SAS `cifbuild` documentation and the XMM calibration-index format describe `EXTSEQID` as the extension-sequence identifiers. The current `cifdiff` algorithm sorts `CalIndex` objects with `CcfConstituent::compare` before set comparison, supporting a constituent-collection interpretation rather than physical FITS-row-order identity.

References:

- https://xmm-tools.cosmos.esa.int/external/sas/current/doc/cifbuild.pdf
- https://xmm-tools.cosmos.esa.int/external/sas/current/doc/cifdiff.pdf
- https://xmm-tools.cosmos.esa.int/external/xmm_obs_info/odf/data/docs/XMM-SOC-GEN-ICD-0024.pdf

## `xmm-region-tool.calindex/v3`

The v3 CALINDEX semantic digest includes fields that identify/select/index calibration content:

- `TELESCOP`;
- `SCOPE`;
- `TYPEID`;
- `ISSUE`;
- `VALDATE`;
- `VALDATE-END`;
- normalized basename form of `FNAME`;
- `EXTSEQU`;
- `EXTSEQID`.

`EXTSEQID` is deliberately **not** aliased from `EXTSEQUID`: the supported real SAS-22 CIF emits `EXTSEQID`, and no real compatibility spelling has been demonstrated.

Before hashing, semantic rows are sorted by their canonical JSON representation. Duplicate rows remain duplicates, but harmless physical FITS row reordering does not create a different calibration identity.

## Evidence/validation fields that are not semantic selectors

The following CALINDEX fields are excluded from the v3 semantic digest:

- `DATE` — constituent creation date;
- `SUBDATE` — constituent submission date;
- `FSIZE` — constituent-size evidence;
- `MD5` — constituent integrity signature;
- `CREATOR` — tool/path provenance.

This does **not** mean these fields are ignored. `FSIZE` and `MD5` remain fail-closed validation evidence: when present, the resolved constituent must match them before its SHA-256 is accepted. Exact selected constituent bytes are independently SHA-256-bound in `CalibrationIdentity`, so including size/MD5 again in the scientific CALINDEX digest would create redundant identity differences without changing selected calibration bytes. `DATE`, `SUBDATE`, and `CREATOR` are retained as CALINDEX/container provenance rather than selection/index semantics.

## Downstream schema relationship

The CALINDEX semantic-hash schema is `xmm-region-tool.calindex/v3`.

`xmm-region-tool.calibration-identity/v2` binds the semantic CALINDEX digest plus exact selected constituent/replacement bytes. The nested `calindex_sha256` changes when CALINDEX semantic meaning changes. SAS projection-context and projection canonical records bind the resulting calibration-identity SHA-256 without embedding presentation-only CALINDEX fields.

Historical detector-coordinate observations created under older CALINDEX hash schemas remain evidence for the detector coordinates and CCF bytes actually tested; new contexts use the current v3 semantic digest.
