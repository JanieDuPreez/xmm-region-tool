# Science event / `esky2det` calinfoset contract

The package distinguishes **generic event identity inspection** from the stricter role of an EPIC science event used as the normal `esky2det` calibration-information product.

## Generic identity

`read_event_identity()` identifies one unambiguous canonical `EVENTS` table and reconciles material duplicate identity values across that table and the primary header. Partial or derived products may still be inspectable through this layer when optional pointing/calibration fields are absent.

## Science-calinfoset role

`project_selection()` requires the supplied event product to pass `read_science_calinfoset_identity()` before the first `esky2det` subprocess is started.

For this role the **primary header** must contain non-empty values for:

- `TELESCOP` identifying an XMM product;
- supported EPIC `INSTRUME`;
- `DATE-OBS`;
- `RA_PNT`;
- `DEC_PNT`;
- `PA_PNT`.

The primary-header ownership is deliberate. It matches the documented `calinfostyle=set` contract used by the normal projection path. Values copied only into a secondary extension do not make an otherwise incomplete primary header acceptable as a science calinfoset.

`DATE-OBS` must be a complete FITS observation date-time and must parse as a valid UTC FITS timestamp. The pointing values must already be finite under the generic event-identity contract and additionally satisfy the physical role bounds:

```text
0 <= RA_PNT < 360 deg
-90 <= DEC_PNT <= 90 deg
0 <= PA_PNT < 360 deg
```

Although SAS documents an inclusive 360-degree endpoint for corresponding user parameters, this package deliberately uses the canonical half-open RA/position-angle domain and rejects an exact `360` header value rather than silently wrapping it to zero. Arbitrary finite wrap spellings are never normalized into apparently valid calibration metadata.

The `esky2det` invocation records `calinfostyle=set` explicitly together with `calinfoset=<exact event product>`; it does not rely on the SAS task default.

## Projection-time event snapshot and lifecycle stability

Before projection, `read_science_calinfoset_snapshot()` hashes the exact event bytes, reads and validates the science-event identity, then hashes the bytes again. Projection proceeds only when those two hashes agree. This prevents the accepted semantic identity from one persistent file generation being paired with the exact SHA-256 of another generation during the initial snapshot.

`project_selection()` also hashes the original event path after projection and requires it to match the accepted pre-projection SHA-256. Ordinary persistent changes to the event file during projection therefore fail closed before a successful `ProjectionResult` is returned.

This is deliberately **not** an execution-owned immutable copy of the event product. Every adaptive `esky2det` call still resolves the caller-owned `calinfoset` path. The caller must therefore keep that path and its underlying bytes stable for the complete projection lifecycle, from the initial snapshot until `project_selection()` returns. In particular, the before/after hashes do not prove safety against concurrent in-place mutation or an ABA sequence where the path temporarily changes to different bytes between checks and is restored to the original bytes before the final hash. The package does not claim that every SAS subprocess saw one generation under such concurrent mutation.

If stronger concurrent-mutation isolation is required, it needs an execution-owned immutable snapshot, or an equivalent mechanism, used as the `calinfoset` for every SAS call; additional hashes around individual subprocesses would not by themselves prove that property.

## Workflow ownership after projection

A successful `ProjectionResult` is authoritative for the event generation that produced its detector geometry. Callers may pre-read an `EventIdentity` and an exact event SHA-256 for naming, context selection or invocation-level policy, but those preflight records do not become authoritative merely because projection later succeeds.

Before the main CLI or batch API materialises a successful product, `validate_projection_event_generation()` requires all of the following:

- the caller's pre-read semantic event identity matches `ProjectionProvenance.event_identity_sha256`;
- when the workflow froze an invocation/item generation, its exact SHA-256 matches `ProjectionProvenance.event_file_sha256`;
- a fresh science-role snapshot of the current event path still has the same semantic identity and exact SHA-256 as the completed projection.

The standalone CLI freezes one exact SHA-256 per event path before iterating over extraction cells. Consequently one invocation cannot quietly project cell 1 from generation A and cell 2 from same-identity generation B and report both as successful. Batch success entries likewise record the projection-owned exact SHA-256 and semantic identity; failed entries do not present a mere preflight SHA as successful projection evidence.

Durable detector-geometry promotion independently re-reads the real event artifact from `EventIdentity.path`. A caller-constructed `EventIdentity` and a caller-constructed matching provenance digest therefore cannot vouch for each other when the on-disk event says something different. The current event bytes must also still equal the exact SHA-256 recorded by the `ProjectionResult`.

These checks share the caller-owned lifecycle-stability limitation described above: they fail ordinary persistent generation drift, but are not advertised as an adversarial concurrent-mutation/ABA isolation mechanism.

## Real-SAS validation event snapshot

`validate_fits_region_with_evselect()` uses a stronger execution boundary for its row-level comparison. After the generated region has been bound successfully to the current event, the validator captures an execution-owned temporary copy and verifies that copy has the accepted exact event SHA-256. It then injects the temporary row-id column into a derivative of that snapshot.

Both the package's expected DETX/DETY membership calculation and SAS `evselect table=<snapshot-with-row-id>:EVENTS` operate on that same derivative. A persistent replacement of the caller's event path between the initial binding check and snapshot capture therefore fails before SAS, and expected membership cannot accidentally be computed from one event generation while SAS filters another.

## pn science event versus OOT event

A pn cleaned OOT list is a different workflow role from the normal pn science event list even though both are structurally similar XMM event products. Generic identity inspection therefore remains able to read OOT files, but the science role rejects the known current/compatibility OOT product spellings:

```text
*-allevcoot.fits
*-allevc-oot.fits
```

`pn_oot_event_file()` may still resolve those names as an `ootevtfile` sibling. A resolved sibling must be a physically distinct file from the science event and must not be byte-identical to it; matching instrument, ObsID and exposure are necessary but not sufficient evidence of a valid role pairing.

This is deliberately a fail-closed **known-product-name** guard, not a cryptographic OOT classifier. An arbitrarily renamed OOT event list can be indistinguishable from the normal list using the currently retained event headers, so the package does not claim to detect such a renamed artifact.

## Evidence boundary

Synthetic tests prove that incomplete, malformed or physically invalid calibration metadata fails before `esky2det` is invoked and that generic event inspection can remain less restrictive where appropriate. They also prove that known OOT names cannot enter the normal science role, that aliased/identical science/OOT sibling files are rejected, that a persistent A -> B event-file replacement during the initial identity/SHA snapshot fails closed, that same-identity exact-byte drift cannot cross batch/CLI materialisation boundaries, that multi-cell CLI runs retain one frozen generation, that durable artifact promotion re-reads the real event, and that validator expected/SAS membership is based on one captured generation. They do **not** prove protection against the concurrent ABA/in-place mutation case described above, nor do they prove which header cards a real SAS installation reads.

Real-event acceptance was checked separately on canonical MOS/pn ESAS `P-allevc.fits` products: the required calibration-information fields are present in the primary header and ordinary real projection succeeds with explicit `calinfostyle=set`. That real evidence, rather than mocked tests, grounds the supported primary-header/calinfoset contract.
