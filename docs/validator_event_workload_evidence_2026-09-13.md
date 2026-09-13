# Real-SAS validator event-workload evidence (2026-09-13)

The optional `xmm-region-sas-validate` path is bounded against pathological FITS event tables before canonical EVENTS data are materialised.

A metadata-only survey of 84 retained canonical event products from the A133/A3376 validation corpus found:

- maximum EVENTS rows (`NAXIS2`): 1,087,832;
- maximum declared EVENTS payload (`NAXIS1*NAXIS2+PCOUNT`): 48,952,440 bytes;
- maximum complete file size in the surveyed set: 103,921,920 bytes;
- all surveyed products had `PCOUNT=0` and no `THEAP`.

The validator therefore uses deliberately generous limits of:

- 5,000,000 EVENTS rows;
- 256 MiB declared EVENTS payload.

These are approximately 4.6× and 5.5× the observed maxima. They are validator availability limits, not scientific event-selection limits, and do not apply to normal `xmm-region` projection.

Before accessing `event_hdu.data`, the validator checks `NAXIS1`, `NAXIS2`, `PCOUNT`, and any `THEAP` for bounded integer/layout semantics, independently enforces the signed-int32 row-ID domain, and rejects workloads outside the validator policy. Accepted products then use the same temporary `XMMRGROW` int32 row-ID representation and exact-row comparison semantics.

This hardening does not change the accepted temporary event representation, detector geometry, event identity, SAS arguments or exact-row comparison semantics. Existing real-SAS evidence for accepted ordinary products therefore remains applicable.
