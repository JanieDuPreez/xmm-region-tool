# Detector component and hole topology

The detector-space model distinguishes two different Boolean concepts:

- a top-level `DetectorRegion(include=True/False, ...)` is one independent Boolean component;
- a boundary after the first inside one `DetectorRegion` has `subtract=True` and is an **internal hole of that component**.

Those concepts must not be conflated. A top-level excluded component is allowed to overlap or cross an included component because the intended selection is ordinary Boolean geometry such as `(science) AND NOT (mask)`. Internal subtractive holes have a stronger structural contract.

## Canonical detector-polygon invariants

Every detector boundary accepted for projection output or durable managed materialisation must be a real finite simple polygon:

1. at least three finite XY vertices;
2. no zero-length or numerically unresolved adjacent edge;
3. no non-adjacent self-intersection;
4. non-zero, numerically resolved enclosed area.

The edge/area degeneracy checks are scale-relative to the polygon's own detector-space extent. There is deliberately no fixed minimum science-region size: a very small but numerically resolved triangle remains valid, while an exactly collinear triangle is rejected.

Self-intersection is diagnosed before the area check. A bow-tie therefore reports the more specific crossing failure even though its signed shoelace area can also cancel to zero.

## Canonical internal-hole invariants

Each `DetectorRegion` is validated independently:

1. every boundary satisfies the detector-polygon invariants above;
2. boundaries within one component may not intersect or touch one another;
3. every subtractive boundary must be **strictly contained** by the same component's outer boundary;
4. a subtractive boundary outside the outer boundary, or one that contains the outer boundary, is invalid;
5. multiple holes must be disjoint and non-nested;
6. duplicate/redundant holes are rejected rather than silently canonicalised.

The same internal-hole rules apply when the top-level component itself has `include=False`, for example an excluded annulus. They do not impose any containment or non-intersection requirement between separate top-level included/excluded components.

For simple non-intersecting polygons, containment is classified with an explicit point-in-polygon test after edge-intersection/touch checks. A representative hole vertex must classify strictly inside the parent outer polygon. Boundary contact is rejected by the preceding crossing/touch check rather than being treated as containment.

## Validation ownership

`src/xmm_region_tool/topology.py` owns the reusable detector-topology predicate and raises `DetectorTopologyError`.

The normal SAS projection path runs this predicate before returning detector geometry and translates failures into the existing `SasConversionError` surface.

The durable artifact boundary runs **the same predicate again** before `write_detector_geometry()` can publish FITS geometry. `write_bound_detector_geometry()` uses that durable writer, so a caller cannot manually construct malformed `DetectorSelection`/`ProjectionResult` objects, recompute matching geometry/provenance hashes, and have the managed workflow bless them merely because the digests are internally consistent. Topology validity and identity integrity are separate requirements.

Managed topology failures are translated to `ArtifactMaterializationError` before any output file is written. This revalidation is intentional even for ordinary `project_selection()` results: persisted or caller-supplied value objects are an independent trust boundary.

## Regression scope

Automated regressions cover:

- a valid internal hole;
- a hole fully outside its outer component;
- a subtractive polygon fully containing its outer component;
- nested/redundant holes;
- touching outer/hole boundaries;
- a valid internal hole of an excluded top-level component;
- a separate top-level exclusion mask crossing an included component, which remains valid;
- adjacent duplicate detector vertices;
- bow-tie/self-intersecting polygons;
- exactly collinear zero-area polygons;
- a very small but numerically resolved triangle;
- hash-consistent malformed `ProjectionResult` objects rejected by the managed writer before output;
- valid hash-consistent managed geometry still materialising normally.

These checks change malformed-geometry rejection and defensive revalidation only. They do not alter valid celestial geometry, detector coordinates, adaptive tessellation, FITS REGION Boolean serialisation, calibration identity or SAS arguments. Previously validated ordinary detector polygons therefore remain applicable evidence for the accepted valid-geometry path.
