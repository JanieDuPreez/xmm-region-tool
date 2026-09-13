# SAS projection-context lifecycle stability

`SasProjectionContext` is a frozen **identity/evidence snapshot** for the SAS producer and calibration state used by managed projection. It is not an execution-owned copy of the SAS installation or calibration tree.

## What the package guarantees

A context captures:

- an immutable copy of the relevant environment mapping;
- the active calibration identity, including exact selected CCF/replacement bytes and exact active CIF bytes as execution evidence;
- the exact `esky2det` executable path, SHA256 and producer/version identity.

`project_selection()` validates that captured state immediately before projection and validates it again after projection. A persistent replacement or modification of the active CIF/CCF/replacement state or `esky2det` executable is therefore detected before a successful `ProjectionResult` is returned.

The same checks also reject a manually constructed context whose captured calibration/producer records do not agree with the actual execution state at validation time.

Path relocation with semantically/materially identical bytes remains valid where the identity contracts intentionally permit it.

### Deterministic public executable resolution

The supported top-level `SasProjectionContext.from_environment(...)` boundary does not silently consult an uncaptured ambient `PATH` when a caller supplies an explicit environment mapping. For name-based `esky2det` resolution, that explicit mapping must itself contain a non-empty `PATH`. An absolute executable path is independent of `PATH`.

This rule is deliberately stricter than low-level/debug implementation helpers. It ensures that repeating the same explicit public environment mapping cannot select a different named executable solely because the process-wide ambient `PATH` changed.

### Complete producer evidence for durable output

Low-level/debug `SasProducerIdentity` values may remain partial, but supported durable/managed scientific outputs require complete producer evidence before publication or reuse:

- one canonical non-empty `esky2det` executable basename;
- a canonical non-empty task version/build string;
- the exact lower-case 64-hex SHA256 of the `esky2det` executable bytes;
- a separately captured canonical non-empty SAS release/build identifier.

The package does **not** infer the SAS release from arbitrary task-banner text when `sasversion` evidence is absent. A context with no separate SAS release/build identifier may still be useful for explicit low-level/debug work, but it cannot be promoted through the supported durable scientific boundary.

## What the package does not guarantee

Adaptive projection may invoke `esky2det` many times. Each invocation executes the caller-owned `context.esky2det_path`, and SAS/CAL resolves the caller-owned CIF/CCF/replacement paths recorded by the context environment.

The package does **not** stage or copy those files into an execution-owned immutable tree. Consequently, pre/post validation cannot prove isolation from a concurrent A -> B -> A substitution such as:

```text
pre-validation sees generation A
one or more SAS calls consume A
another process temporarily replaces executable/calibration bytes with B
one or more SAS calls consume B
that process restores exact generation A
post-validation sees A and therefore passes
```

The package therefore does not claim that endpoint validation proves every intermediate SAS subprocess consumed one immutable generation under adversarial or uncontrolled concurrent filesystem mutation.

## Caller / workflow responsibility

For the complete lifetime of `project_selection()`, callers must ensure that the SAS execution state represented by the context cannot change concurrently. This responsibility covers at least:

- `context.esky2det_path`;
- the active CIF referenced by `SAS_CCF`;
- every CIF-selected CCF constituent resolved through `SAS_CCFPATH` or equivalent SAS fallback search roots;
- every explicit `SAS_CCFFILES` replacement;
- the filesystem namespace needed to resolve those files consistently.

For ordinary single-user SAS installations this is normally satisfied by treating the SAS installation and calibration repository as read-only/stable during analysis.

Managed, shared, cache-backed or concurrent workflows must provide an equivalent lifecycle guarantee themselves, for example by using a private/read-only installation or by holding an external lifecycle lock that spans the entire `project_selection()` call. A lock taken only around context capture or only around final validation is insufficient.

The package does not currently implement a calibration-tree/executable staging layer because the documented local scientific-tool threat model does not justify copying an entire SAS/CCF installation merely to defend against a concurrent ABA mutation. A workflow that requires stronger isolation should introduce an execution-owned immutable snapshot, or an equivalent mechanism, and run every SAS call against that snapshot.

## Relation to event-file lifecycle

This is intentionally analogous to `docs/science_event_contract.md`.

The event contract likewise detects persistent generation drift but does not claim that repeated hashes of a caller-owned path provide immutable-copy isolation against a concurrent ABA replacement. Both contracts therefore place the same responsibility on a managed/shared workflow: caller-owned execution inputs must remain stable for the complete operation unless the package explicitly stages immutable execution-owned copies.

## Separate lifecycle boundaries

Observation-association/CIF-suitability checks answer whether the captured calibration context belongs to the event observation; they do not make the underlying SAS/calibration files immutable during execution.

Likewise, the managed detector-geometry staging contract addresses the later validation-to-SAS-consumption boundary after projection. Event generation stability has its own contract in `docs/science_event_contract.md`.

These boundaries are separate:

```text
SAS/calibration inputs stable during projection
context belongs to the event observation
event bytes stable/coherent during projection
persisted geometry copied/verified for SAS consumption
```

See `docs/context_suitability.md`, `docs/science_event_contract.md`, and `docs/managed_execution_contract.md` for the corresponding contracts.

## Evidence boundary

Synthetic tests prove the package's endpoint validation and fail-closed behaviour for persistent calibration/executable changes. Real A133 testing demonstrates that an ordinary stable SAS 22.1.0 environment passes context validation before and after real projection and that returned provenance identities match the captured context.

Neither synthetic endpoint tests nor that real evidence establish detection of an intentionally timed concurrent A -> B -> A mutation between individual `esky2det` calls. The package explicitly does not make that claim.
