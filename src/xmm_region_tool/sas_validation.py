"""Validate generated FITS REGION files through real SAS selectlib/evselect."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from regions import PixCoord, PolygonPixelRegion

from .asc_region import polygon_covers
from .limits import validate_region_table_budget
from .provenance import (
    EventIdentityError,
    canonical_event_hdu_index,
    check_region_binding,
    file_sha256,
)
from .sas_paths import (
    SasPathError,
    validate_sas_safe_resolved_path,
    validate_sas_temporary_root,
)
from .sas_task_producer import (
    SasTaskProducerError,
    SasTaskProducerIdentity,
    capture_sas_task_producer,
    validate_sas_task_producer_capture,
)
from .subprocess_capture import diagnostic_summary, run_bounded


class SasRegionValidationError(RuntimeError):
    """Raised when a real-SAS region validation cannot be completed."""


_VALIDATION_ROW_ID = "XMMRGROW"
# Validator-only availability limits.  The retained A133/A3376 corpus inspected
# for #112 contained 84 canonical products; its maxima were 1,087,832 rows and
# 48,952,440 declared EVENTS-table bytes.  Keep roughly fivefold headroom while
# bounding the current in-memory row-id snapshot construction.
_MAX_VALIDATION_EVENT_ROWS = 5_000_000
_MAX_VALIDATION_EVENT_TABLE_BYTES = 256 * 1024 * 1024
_MAX_FITS_HEADER_INTEGER = np.iinfo(np.int64).max


@dataclass(frozen=True)
class SasRegionValidationReport:
    """Row-level agreement between FITS REGION semantics and SAS evselect."""

    expected_selected: int
    sas_selected: int
    compared_columns: tuple[str, ...]
    row_sequence_match: bool
    command: tuple[str, ...]
    evselect_executable: Path
    evselect_producer: SasTaskProducerIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "evselect_executable", Path(self.evselect_executable).resolve())

    @property
    def compatible(self) -> bool:
        return self.expected_selected == self.sas_selected and self.row_sequence_match

    def format_text(self) -> str:
        return "\n".join(
            [
                "XMM FITS REGION real-SAS validation",
                f"expected selected rows: {self.expected_selected}",
                f"SAS selected rows: {self.sas_selected}",
                f"row sequence match: {'yes' if self.row_sequence_match else 'NO'}",
                f"row identity column: {_VALIDATION_ROW_ID} (temporary unique int32 row id)",
                "compared columns: " + ", ".join(self.compared_columns),
                f"evselect executable: {self.evselect_executable}",
                f"evselect task version: {self.evselect_producer.task_version}",
                f"evselect executable SHA256: {self.evselect_producer.executable_sha256}",
                f"evselect producer identity SHA256: {self.evselect_producer.identity_sha256}",
                "SAS release: " + (self.evselect_producer.sas_version or "unavailable"),
                f"compatible: {'yes' if self.compatible else 'NO'}",
            ]
        )


def _polygon_vertices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise SasRegionValidationError("FITS REGION polygon X/Y vectors are malformed")
    if len(x) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise SasRegionValidationError("FITS REGION polygon has unusable coordinates")

    stop = len(x)
    for index in range(1, len(x)):
        if x[index] == x[0] and y[index] == y[0]:
            stop = index
            break
    if stop < 3:
        raise SasRegionValidationError("FITS REGION polygon closes before three vertices")
    return np.column_stack((x[:stop], y[:stop]))


def fits_region_membership(
    region_fits: str | Path,
    detx: np.ndarray,
    dety: np.ndarray,
) -> np.ndarray:
    """Evaluate emitted ASC-FITS-REGION semantics with inclusive polygon boundaries."""
    region_path = Path(region_fits).expanduser().resolve()
    coordinates = PixCoord(x=np.asarray(detx, dtype=float), y=np.asarray(dety, dtype=float))
    with fits.open(region_path, memmap=True) as hdus:
        if "REGION" not in hdus:
            raise SasRegionValidationError(f"FITS file has no REGION extension: {region_path}")
        table = hdus["REGION"]
        try:
            validate_region_table_budget(table)
        except ValueError as exc:
            raise SasRegionValidationError(str(exc)) from exc
        names = {str(name).upper(): str(name) for name in table.columns.names}
        required = {"SHAPE", "X", "Y", "COMPONENT"}
        if not required.issubset(names):
            missing = ", ".join(sorted(required - set(names)))
            raise SasRegionValidationError(f"REGION table is missing columns: {missing}")

        component_rows: dict[int, list[tuple[bool, PolygonPixelRegion]]] = {}
        for row in table.data:
            shape = str(row[names["SHAPE"]]).strip().upper()
            negate = shape.startswith("!")
            base_shape = shape[1:] if negate else shape
            if base_shape != "POLYGON":
                raise SasRegionValidationError(
                    f"validator only supports generated POLYGON rows; got {shape!r}"
                )
            vertices = _polygon_vertices(row[names["X"]], row[names["Y"]])
            polygon = PolygonPixelRegion(PixCoord(x=vertices[:, 0], y=vertices[:, 1]))
            component = int(row[names["COMPONENT"]])
            component_rows.setdefault(component, []).append((negate, polygon))

    if not component_rows:
        raise SasRegionValidationError("REGION table contains no rows")

    result = np.zeros(np.asarray(detx).shape, dtype=bool)
    for rows in component_rows.values():
        component_mask = np.ones(np.asarray(detx).shape, dtype=bool)
        for negate, polygon in rows:
            element = polygon_covers(polygon, coordinates)
            component_mask &= ~element if negate else element
        result |= component_mask
    return result


def _canonical_event_index(hdus: fits.HDUList) -> int:
    try:
        return canonical_event_hdu_index(hdus)
    except EventIdentityError as exc:
        raise SasRegionValidationError(str(exc)) from exc


def _event_hdu(path: Path):
    hdus = fits.open(path, memmap=True)
    try:
        return hdus, hdus[_canonical_event_index(hdus)]
    except Exception:
        hdus.close()
        raise


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _required_header_integer(header: fits.Header, key: str, *, minimum: int) -> int:
    """Return a bounded integer FITS structural keyword without coercive parsing."""
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SasRegionValidationError(f"EVENTS {key} must be an integer FITS header value")
    result = int(value)
    if result < minimum:
        raise SasRegionValidationError(f"EVENTS {key} must be >= {minimum}; got {result}")
    if result > _MAX_FITS_HEADER_INTEGER:
        raise SasRegionValidationError(f"EVENTS {key} exceeds the supported FITS integer domain")
    return result


def _validation_event_table_preflight(event_hdu) -> int:
    """Validate validator workload from FITS metadata before touching table data."""
    header = event_hdu.header
    row_width = _required_header_integer(header, "NAXIS1", minimum=1)
    row_count = _required_header_integer(header, "NAXIS2", minimum=1)
    heap_bytes = _required_header_integer(header, "PCOUNT", minimum=0)

    if row_count > np.iinfo(np.int32).max:
        raise SasRegionValidationError(
            "event table has too many rows for SAS-compatible unique int32 validation ids: "
            f"{row_count}"
        )
    if row_count > _MAX_VALIDATION_EVENT_ROWS:
        raise SasRegionValidationError(
            "event table exceeds the real-SAS validator row workload limit: "
            f"{row_count} > {_MAX_VALIDATION_EVENT_ROWS}"
        )

    fixed_table_bytes = row_width * row_count
    total_table_bytes = fixed_table_bytes + heap_bytes
    if total_table_bytes > _MAX_VALIDATION_EVENT_TABLE_BYTES:
        raise SasRegionValidationError(
            "event table exceeds the real-SAS validator byte workload limit: "
            f"{total_table_bytes} > {_MAX_VALIDATION_EVENT_TABLE_BYTES}"
        )

    if "THEAP" in header:
        heap_start = _required_header_integer(header, "THEAP", minimum=0)
        if heap_start < fixed_table_bytes:
            raise SasRegionValidationError(
                "EVENTS THEAP points inside the fixed-width table data"
            )
        if heap_start > total_table_bytes:
            raise SasRegionValidationError(
                "EVENTS THEAP lies beyond the NAXIS1*NAXIS2+PCOUNT data area"
            )

    return row_count


def _validation_event_copy(source: Path, target: Path) -> Path:
    """Copy an event product and inject a unique row id into its EVENTS table.

    The user's science event file is never modified. The temporary row id makes
    the subsequent evselect comparison an exact proof of original row identity
    and ordering, rather than relying on a tuple of science columns being unique.

    SAS DAL does not accept the CFITSIO 64-bit integer type used by FITS ``K``
    columns in this path, so validation uses a signed 32-bit FITS ``J`` column.
    Structural metadata and the validator workload budget are checked before the
    EVENTS data are materialised.
    """
    with fits.open(source, memmap=True, lazy_load_hdus=True) as hdus:
        event_index = _canonical_event_index(hdus)
        event_hdu = hdus[event_index]
        row_count = _validation_event_table_preflight(event_hdu)

        existing = {str(name).upper() for name in event_hdu.columns.names}
        if _VALIDATION_ROW_ID in existing:
            raise SasRegionValidationError(
                f"event table already contains reserved validation column {_VALIDATION_ROW_ID}"
            )

        data = event_hdu.data
        if data is None:
            raise SasRegionValidationError("canonical EVENTS table contains no rows")
        if len(data) != row_count:
            raise SasRegionValidationError(
                "EVENTS table row count disagrees with preflight metadata"
            )

        row_id = np.arange(row_count, dtype=np.int32)
        row_column = fits.Column(name=_VALIDATION_ROW_ID, format="J", array=row_id)
        replacement = fits.BinTableHDU.from_columns(
            event_hdu.columns + row_column,
            header=event_hdu.header,
            name=event_hdu.name,
        )
        output_hdus = [hdu.copy() for hdu in hdus]
        output_hdus[event_index] = replacement
        fits.HDUList(output_hdus).writeto(target, overwrite=True, checksum=True)
    return target


def _producer_capture(evselect: str):
    try:
        return capture_sas_task_producer(evselect)
    except SasTaskProducerError as exc:
        raise SasRegionValidationError(f"cannot identify evselect producer: {exc}") from exc


def _validate_producer_capture(capture) -> None:
    try:
        validate_sas_task_producer_capture(capture)
    except SasTaskProducerError as exc:
        raise SasRegionValidationError(f"evselect producer identity is stale: {exc}") from exc


def validate_fits_region_with_evselect(
    region_fits: str | Path,
    event_file: str | Path,
    *,
    source_region: str | Path | None = None,
    evselect: str = "evselect",
) -> SasRegionValidationReport:
    """Run real ``evselect`` against one verified execution-owned event snapshot.

    The generated region is first checked against the caller's current event
    artifact. Validation then copies that exact accepted generation into a
    temporary snapshot and verifies the snapshot SHA before doing any expected-
    mask calculation. A second temporary file derived from that snapshot gets a
    unique row-id column; both the package membership calculation and SAS
    ``evselect`` consume this same row-id-bearing event generation. The user's
    science event file is never modified.

    The validator producer is captured independently from projection provenance.
    The exact resolved ``evselect`` path is execution metadata; task version,
    exact executable SHA256 and SAS release form a path-independent producer
    identity. Exact executable bytes are rechecked immediately before and after
    validation so a substituted generation fails closed instead of being
    reported under stale producer evidence.
    """
    binding = check_region_binding(region_fits, event_file, source_region=source_region)
    if not binding.compatible:
        raise SasRegionValidationError(binding.format_text())

    producer_capture = _producer_capture(evselect)

    event_path = Path(event_file).expanduser().resolve()
    try:
        region_path = validate_sas_safe_resolved_path(
            region_fits,
            role="real-SAS REGION blockspec",
        )
        temp_parent = validate_sas_temporary_root(role="real-SAS validation temporary root")
    except SasPathError as exc:
        raise SasRegionValidationError(str(exc)) from exc

    with tempfile.TemporaryDirectory(
        prefix="xmm-region-sas-validation-",
        dir=temp_parent,
    ) as directory:
        temp_root = Path(directory)
        raw_snapshot = validate_sas_safe_resolved_path(
            temp_root / "events-snapshot.fits",
            role="real-SAS validation event snapshot",
        )
        shutil.copyfile(event_path, raw_snapshot)
        snapshot_sha = file_sha256(raw_snapshot)
        if snapshot_sha != binding.event_file_sha256:
            raise SasRegionValidationError(
                "event file changed while the real-SAS validation snapshot was being captured; "
                "refusing to compare different event generations"
            )

        validation_event = _validation_event_copy(
            raw_snapshot,
            validate_sas_safe_resolved_path(
                temp_root / "events-with-row-id.fits",
                role="real-SAS validation row-id event",
            ),
        )
        hdus, event_hdu = _event_hdu(validation_event)
        try:
            data = event_hdu.data
            if data is None:
                raise SasRegionValidationError("event table contains no rows")
            names = {str(name).upper(): str(name) for name in event_hdu.columns.names}
            detx = np.asarray(data[names["DETX"]], dtype=float)
            dety = np.asarray(data[names["DETY"]], dtype=float)
            expected_mask = fits_region_membership(region_path, detx, dety)
            expected_row_ids = np.asarray(
                data[names[_VALIDATION_ROW_ID]][expected_mask],
                dtype=np.int64,
            )
            identity_candidates = (
                "TIME",
                "X",
                "Y",
                "DETX",
                "DETY",
                "PI",
                "PHA",
                "PATTERN",
                "CCDNR",
                "RAWX",
                "RAWY",
            )
            compare_names = tuple(name for name in identity_candidates if name in names)
            expected_columns = {
                name: np.asarray(data[names[name]][expected_mask]).copy()
                for name in compare_names
            }
            expected_selected = len(expected_row_ids)
        finally:
            hdus.close()

        filtered = validate_sas_safe_resolved_path(
            temp_root / "filtered-events.fits",
            role="real-SAS validation filtered event",
        )
        expression = f"region({region_path},DETX,DETY)"
        command = (
            str(producer_capture.executable_path),
            f"table={validation_event}:EVENTS",
            f"expression={expression}",
            "withfilteredset=yes",
            "keepfilteroutput=yes",
            f"filteredset={filtered}",
            "writedss=no",
            "updateexposure=no",
            "filterexposure=no",
        )
        _validate_producer_capture(producer_capture)
        result = run_bounded(command)
        _validate_producer_capture(producer_capture)
        if result.returncode != 0:
            details = diagnostic_summary(result.stderr or result.stdout)
            raise SasRegionValidationError(
                f"evselect failed with exit code {result.returncode}: {details}"
            )
        if not filtered.is_file():
            raise SasRegionValidationError(
                "evselect reported success but created no filtered event list"
            )

        filtered_hdus, filtered_hdu = _event_hdu(filtered)
        try:
            filtered_data = filtered_hdu.data
            if filtered_data is None:
                sas_selected = 0
                row_sequence_match = expected_selected == 0
            else:
                sas_selected = len(filtered_data)
                filtered_names = {
                    str(name).upper(): str(name) for name in filtered_hdu.columns.names
                }
                row_name = filtered_names.get(_VALIDATION_ROW_ID)
                row_sequence_match = row_name is not None and _arrays_equal(
                    expected_row_ids,
                    np.asarray(filtered_data[row_name], dtype=np.int64),
                )
                if row_sequence_match:
                    for name in compare_names:
                        if name not in filtered_names or not _arrays_equal(
                            expected_columns[name],
                            filtered_data[filtered_names[name]],
                        ):
                            row_sequence_match = False
                            break
        finally:
            filtered_hdus.close()

    return SasRegionValidationReport(
        expected_selected=expected_selected,
        sas_selected=sas_selected,
        compared_columns=compare_names,
        row_sequence_match=row_sequence_match,
        command=command,
        evselect_executable=producer_capture.executable_path,
        evselect_producer=producer_capture.identity,
    )
