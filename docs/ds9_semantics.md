# DS9 celestial-semantics contract

`xmm-region-tool` treats user-authored DS9 sky regions as scientific input, but it does not use DS9 itself as the projection engine. The conversion chain is therefore:

```text
DS9 text -> WCS-independent semantic celestial selection -> adaptive SAS detector projection
```

The real-SAS validation programme establishes the second arrow. This document defines the first arrow and, importantly, the limits of what can be inferred from a standalone DS9 region file.

## Initial-release source contract

A bare DS9 region file does **not** preserve the source image WCS on which a marker was authored. SAOImage DS9 maps celestial marker lengths and angles through that active WCS/reference system (`mapLenToRef` / `mapAngleToRef`), whereas this package deliberately stores path-independent celestial geometry.

The initial release therefore does **not** claim that standalone DS9 text can reproduce arbitrary DS9 display/reference-plane geometry to a universal sub-arcsecond bound. Instead, the adapter gives supported source syntax one deterministic WCS-independent celestial interpretation:

- circles are spherical small circles;
- ellipses are local celestial ellipses with the validated DS9 angle conversion;
- boxes are defined by local celestial vertices;
- polygon and box edges are shortest great-circle arcs;
- panda/epanda are represented by the corresponding circular/elliptical celestial sector boundaries.

Unsupported or ambiguous **source-language semantics** still fail closed. Examples include malformed/lost parameters, unsupported composite grouping, invalid annular topology and bare `ecliptic`. What the parser cannot do from text alone is decide whether an otherwise supported celestial marker happened to lie in a source-image WCS configuration where DS9's local reference-plane realization differs from this WCS-independent interpretation by more than a chosen display-space tolerance.

This distinction supersedes the earlier idea of using a universal 30-arcmin radius cutoff. No positive radius-only cutoff can make arbitrary source-WCS display equivalence fail closed because marker position, WCS scale/orientation and projection are missing variables.

## Controlled-WCS evidence and its scope

`tests/test_ds9_semantics.py` compares semantic celestial boundaries against DS9-style local/reference-plane constructions on controlled TAN WCS projections. Those tests do not use SAS or the detector converter as the reference calculation.

The historical centred-TAN matrix uses one native XMM DET unit as its comparison budget:

```text
1 DET = 0.05 arcsec
```

Centred controlled cases include:

- an ICRS circle with radius 30 arcmin;
- an ICRS rotated ellipse with 30 arcmin semimajor radius;
- an ICRS rotated box;
- an FK5 polygon, including the geodesic/TAN edge convention;
- a Galactic rotated ellipse;
- a rotated FK4/B1950 ellipse through the same accepted frame path.

Those centred cases remain useful positive evidence. They are not a proof that every marker of the same size on every source WCS is within 0.05 arcsec.

`tests/test_ds9_source_domain_wave_b.py` deliberately characterizes the missing-WCS limitation: the same 30-arcmin spherical circle that is below 0.05 arcsec in the centred TAN construction exceeds that comparison budget after the marker centre is moved away from CRVAL while retaining the same reference scale. This is why the initial release documents a WCS-independent celestial interpretation rather than pretending a radius-only parser rule can reconstruct arbitrary DS9 display geometry.

The earlier deliberate stress evidence is retained as characterization rather than an executable universal cutoff: a 5-degree circle differs from the centred planar construction by more than 1 arcsec, and a 40-arcmin-semimajor rotated ellipse was measured at about 0.108 arcsec in the corresponding controlled setup.

## Supported celestial frames

For standalone DS9 text the initial release accepts the celestial frame spellings whose frame meaning is available without source-image epoch state:

- `icrs`;
- `fk5` / `j2000`;
- `fk4` / `b1950`;
- `galactic`.

`ecliptic` is deliberately rejected. Stable DS9 ties the ecliptic equinox to the active image/WCS epoch, while a standalone region file does not preserve that epoch. Silently mapping bare `ecliptic` to Astropy's default J2000 ecliptic frame would therefore change physical celestial geometry while appearing successful. Programmatic callers remain free to provide an explicit Astropy ecliptic frame/equinox outside the DS9-text adapter.

Pixel/image coordinate systems are not accepted by the sky-region workflow.

## DS9 ellipse and box size conventions

DS9 ellipse arguments are semiaxis radii. For example,

```text
ellipse(x,y,240",120",angle)
```

has semiaxes 240 and 120 arcsec. `astropy-regions` exposes the corresponding sky ellipse using full `width` and `height`, and the adapter accounts for that conversion.

DS9 box arguments are full width and height.

### Nested ellipse serialization

DS9 region files serialize marker dimensions at finite decimal precision. Nested ellipses intended to share one ellipticity can therefore contain slightly different numeric axis ratios after being saved.

The existing bounded reconciliation is confined to the DS9-text adapter. It preserves each authored first semiaxis and finds one deterministic common second/first-axis ratio only when the required correction fits within the residual used by the historical centred-TAN compatibility calculation. Real rounded `epanda` and multi-ring/profile fixtures need only about 0.001 arcsec of repair.

That residual calculation is **not** evidence of a universal 0.05-arcsec DS9 display-equivalence guarantee. It is an internal compatibility bound for the WCS-independent interpretation, grounded in the centred controlled construction in which it was established. The off-CRVAL characterization above demonstrates why it must not be generalized to arbitrary missing source-WCS context. Programmatic elliptical-annulus and sector constructors remain strict and do not receive DS9 serialization repair.

### Box annuli

Stable DS9 box annuli use one common width/height aspect-ratio family. The DS9 adapter enforces that source-specific rule without imposing it on the neutral programmatic rectangle-annulus API.

For supported box annuli:

- explicit boundary pairs are sorted by the first width before cells are formed, matching DS9's annulus ordering;
- duplicate first-axis boundaries are rejected as zero-area extraction cells;
- `(0,0)` is permitted only as the innermost boundary and produces an ordinary central box rather than a degenerate subtractive polygon;
- partially zero pairs such as `(0,h)` are rejected;
- compact `n=N` uses the same linear width/height interpolation as DS9 and canonicalizes to the equivalent explicit boundary list;
- materially inconsistent width/height ratios fail closed.

No separate source-geometry rounding allowance is introduced for box aspect ratios.

## Angles and coordinate frames

The package's semantic/programmatic angle convention and DS9's celestial directed-angle convention are deliberately separate.

The package's native local tangent-plane convention is:

```text
0 deg  = increasing longitude
90 deg = increasing latitude
```

Real A133 DS9/MOS1 visual validation established that a directed DS9 celestial ray uses the reflected longitude direction:

```text
theta_native = 180 deg - theta_DS9
```

Because this is a reflection, handedness reverses. A DS9 sector maps to the same physical positive-sweep native sector by reflecting both boundaries and exchanging start/stop:

```text
start_native = 180 deg - stop_DS9
stop_native  = 180 deg - start_DS9
```

For an ordinary ellipse or rectangle the axis is undirected and has 180-degree rotational symmetry. Therefore `180 deg - theta` and `-theta` describe the same footprint, and the adapter retains `-theta` as the canonical undirected representation. A non-full `epanda` is different: its cuts attach to a directed end of the ellipse major axis, so the fully directed conversion is retained.

Physical orientation is transformed when normalizing geometry to ICRS; a numerical source-frame angle is never simply copied between celestial frames.

### Panda/epanda fuzzy angular classification

Stable DS9 8.7 does not classify panda angular intervals from the raw textual difference. Degree angle tokens are first normalized modulo one turn, mapped to the reference-frame angle, and `BaseMarker::setAngles()` then uses `FLT_EPSILON` fuzzy comparisons in radians.

The DS9-text adapter reproduces that compatibility boundary before entering the neutral semantic model:

- authored angles outside `[0,360)` and values separated by whole turns are normalized first;
- endpoint equality uses the DS9 float-epsilon angular tolerance rather than exact textual equality or the programmatic sector tolerance;
- fuzzy-equal endpoints canonicalize to one exact full-circle representation, so equivalent spellings receive the same semantic hash;
- an interval just outside that tolerance remains a genuine sector;
- ordinary descending/wrapped intervals retain their expected positive sweep;
- the same rule is applied to circular `panda` and elliptical `epanda`.

The programmatic sector API does **not** inherit DS9 parser fuzz. Programmatic quantities retain their explicit mathematical angle policy, including arbitrarily small positive non-full sweeps within that API's validity domain.

## Polygon and box edges

The package records polygon edges as shortest great-circle arcs. Under a gnomonic/TAN projection, great circles map to straight lines. The FK5 controlled-WCS test verifies that independently projected semantic edge midpoints lie on the straight reference-image chord to numerical precision.

This is the package's explicit celestial edge convention; it should not be described as proof of unrestricted reconstruction of arbitrary DS9 display geometry without the source WCS.

## DS9 panda / sector annuli

`astropy-regions` deliberately does not support `panda`, `epanda`, or `bpanda`, so `panda` and `epanda` are handled by the loss-intolerant DS9 adapter instead of being silently skipped.

For circular `panda(x,y,a1,a2,nangle,r1,r2,nradius)`:

- the source expands to `nangle × nradius` extraction cells;
- angular and radial boundaries are evenly spaced between the canonicalized endpoints;
- reflected directed endpoints are exchanged so the physical wedge retains a positive native sweep;
- the asymmetric real-data case `30 -> 110` maps to native `70 -> 150` and visually matches the unchanged source;
- a zero-inner-radius wedge uses the centre instead of a degenerate inner arc;
- a full-circle cell reuses the ordinary circle/annulus representation.

`epanda` uses the corresponding ellipse-local cuts and the same DS9 angular classification. Its local ray/ellipse intersection formula is algebraically the same as DS9's `BaseEllipse::intersect()` formula. Fresh A133 validation after the directed-angle correction recorded visual agreement for asymmetric MOS1/pn `epanda` cases and exact downstream SAS row-sequence agreement for the generated detector regions. Those are separate evidence classes: the visual/source comparison concerns the first arrow, while exact SAS rows concern consumer interpretation after projection.

Every supported sector boundary segment is passed through the same adaptive detector-space chord-error projection contract as other semantic geometry. Compatibility rendering/sample counts are not part of the celestial geometry hash. `bpanda` remains explicitly unsupported.
