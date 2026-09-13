# SAS dataset and blockspec path safety

The current release deliberately does **not** quote or escape arbitrary filesystem paths passed to SAS dataset/selectlib parameters. SAS DAL and selectlib use structural dataset/blockspec syntax, so treating every POSIX filename character as literal would make the package claim semantics that have not been validated.

## Supported resolved-path grammar

Every filesystem path interpolated into the supported unquoted SAS-facing forms is first expanded/resolved and must match:

```text
[A-Za-z0-9._/-]+
```

In words: ASCII letters, digits, `.`, `_`, `-`, and `/` only.

This intentionally rejects whitespace and structural/unvalidated punctuation including `+`, `:`, `[`, `]`, `,`, `(`, `)`, and other punctuation. The restriction is conservative rather than an assertion that every rejected character is necessarily unusable in every SAS parser.

## Covered parser boundaries

The same shared validator applies to:

- science event paths that become `esky2det calinfoset=<dataset>`;
- package-created `esky2det intab=<dataset>:INPUT` temporary tables;
- FITS REGION paths interpolated into `region(<blockspec>,DETX,DETY)` wrappers;
- managed `materialize_sas_regionfile()` geometry paths;
- real-SAS `evselect` REGION blockspecs;
- package-created temporary event/filter datasets used by real-SAS validation.

Generic FITS reading is not subject to this restriction merely because a path exists. The restriction is applied when a path crosses a SAS parser boundary.

SAS-facing temporary directories are also checked. `TemporaryDirectory` is created beneath the validated active temporary root and package-defined names use only the supported character class, preventing an unusual `TMPDIR` from silently introducing unvalidated DAL syntax.

## Provenance consequences

The path itself is execution location, not semantic geometry or projection identity. Rejecting an unsafe path therefore does not alter identities for an otherwise identical artifact at a supported location.

For already-safe paths, emitted wrapper text and SAS argv values are unchanged. This is a rejection-only support-boundary restriction and does not invalidate retained A133/A3376 detector-geometry or consumer evidence.

## Future expansion

Support for spaces or structural punctuation may be added only after a quoting/escaping or execution-owned staging scheme is defined and directly exercised through the relevant SAS parsers. At minimum, widening support should probe whitespace plus `+`, `:`, `[`, `]`, `,`, `(` and `)` across the affected dataset/blockspec surfaces before the public path contract changes.
