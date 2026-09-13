# SAS calibration-context suitability

Before detector projection, the package must establish two distinct facts:

1. the active SAS observation context belongs to the event observation;
2. the active CIF was built for the correct observation-level date.

These checks are intentionally distinct from semantic calibration identity. ODF files are evidence used to reject stale or cross-observation execution state; their paths and bytes have not been shown to affect `esky2det` detector geometry directly and therefore do not enter semantic projection/context identity.

## Real A133 evidence

The production rule is based on the real A133 ObsID `0723802001` SAS 22.1.0 fixture rather than an assumed equality between CIF and exposure timestamps.

| Evidence | Value |
| --- | --- |
| CIF `OBSVDATE` | `2013-06-08T14:52:32` |
| CIF `ANALDATE` | `2026-07-13T13:45:48` |
| original ODF `*SUM.ASC` ObsID | `0723802001` |
| original ODF observation start | `2013-06-08T14:52:32` |
| generated `*SUM.SAS` ObsID | `0723802001` |
| generated `*SUM.SAS` observation start | `2013-06-08T15:09:56` |
| MOS1 `S001` event `DATE-OBS` | `2013-06-08T15:10:41` |
| MOS2 `S002` event `DATE-OBS` | `2013-06-08T15:10:41` |
| pn `U002` event `DATE-OBS` | `2013-06-08T18:19:00` |

PRIMARY and EVENTS headers agreed within each of the three event products. The exposure timestamps differ across instruments, as expected for one observation containing several exposures.

The important result is that CIF `OBSVDATE` equals the **original ODF observation start exactly**. It does not equal the later generated summary start or the individual MOS/pn exposure starts.

Consequently the package must not use weaker predicates such as:

- CIF `OBSVDATE == event DATE-OBS`;
- CIF `OBSVDATE == generated *SUM.SAS start`;
- all cameras sharing one `DATE-OBS`;
- CIF/event dates merely occurring on the same calendar day;
- CIF/event times being within an arbitrary tolerance.

## Standard SAS/ODF workflow

When `SAS_ODF` is available, `project_selection()` proves suitability before invoking `esky2det`.

If `SAS_ODF` points at a generated `*SUM.SAS`:

1. parse its internal fixed-width `OBSERVATION` record;
2. require its ObsID to equal the event ObsID;
3. follow its recorded `PATH` to the original ODF directory;
4. require exactly one original `*SUM.ASC` there;
5. parse the original ODF `OBSERVATION` record;
6. require its ObsID to equal the event ObsID;
7. require its observation start to equal CIF `OBSVDATE` exactly after canonical FITS-time parsing.

If `SAS_ODF` already points at the original ODF directory or `*SUM.ASC`, the original observation record provides both association and material-date evidence directly.

A matching generated-summary ObsID alone is deliberately insufficient. A stale CIF from observation A combined with `SAS_ODF` and an event from observation B must still fail the material date/ODF check.

## ODF-independent CIF construction

SAS also permits CIF construction with an explicitly supplied observation date rather than an active ODF. In that case the package does not infer an observation start from an exposure `DATE-OBS`, because the real A133 fixture demonstrates that those quantities are different.

Callers instead provide an explicit `ObservationRecord` when constructing `SasProjectionContext`:

```python
from xmm_region_tool import ObservationRecord, SasProjectionContext

observation = ObservationRecord(
    obs_id="0723802001",
    scheduled_start="2013-06-08T14:52:32",
    scheduled_end="2013-06-09T16:34:12",
)

context = SasProjectionContext.from_environment(
    observation_evidence=observation,
)
```

This declared record is assertion evidence, not a replacement calibration identity. It is checked against three independent facts available at projection time:

- declared ObsID must equal event ObsID;
- declared observation start must equal CIF `OBSVDATE`;
- event `DATE-OBS` must lie inside the declared observation interval.

If neither `SAS_ODF` evidence nor an explicit observation record is available, projection fails closed.

## Stale-state protection and identity

Suitability evidence is captured immediately before projection and again after projection. For file-backed ODF evidence, exact summary-file SHA256 values are included in the temporary evidence snapshot. If the association/suitability evidence changes during projection, the result is rejected.

The following remain deliberately excluded from `SasProjectionContext.canonical_record()` and projection identity:

- `SAS_ODF` path;
- generated summary bytes;
- original ODF-summary bytes;
- explicit `ObservationRecord` evidence.

This preserves path relocation and avoids turning non-material workflow state into a false detector-geometry identity difference. Material calibration selection remains bound through the calibration/CALINDEX/constituent identity.

## Validation scope

The date/association relationship above is grounded in a real A133 SAS 22.1.0 fixture. Automated tests exercise failure cases and ordering, including stale-CIF and wrong-observation-summary rejection before any detector projection call. Those synthetic tests verify software behaviour; they do not substitute for the real fixture evidence and do not claim to validate SAS detector geometry.
