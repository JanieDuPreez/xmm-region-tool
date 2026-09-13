"""Self-contained managed-safe public artifact workflow.

The implementation-level :mod:`xmm_region_tool.managed` helpers retain an
``EventIdentity`` input for compatibility with existing internal callers. The
supported top-level API deliberately does not trust a caller-constructed event
identity: ordinary callers provide the exact science event path, from which the
canonical science-role identity and exact bytes are derived internally.

For compatibility, a legacy ``event_identity=`` argument is accepted only as a
path carrier. Its semantic fields are never trusted; the file is re-read under
the same science-event role before managed materialisation or reuse.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from astropy.io import fits

from .artifacts import ArtifactMaterializationError, SasRegionfileMaterialization
from .calibration import CalibrationIdentity, SasProducerIdentity
from .context_identity import ContextIdentityError, validate_projection_context_identity
from .event_roles import read_science_calinfoset_snapshot
from .execution import ProjectionResult
from .json_policy import StrictJsonError, loads_strict
from .limits import bounded_text
from .managed import (
    BoundDetectorGeometryArtifact,
    load_bound_detector_geometry as _load_bound_detector_geometry,
    materialize_bound_sas_regionfile as _materialize_bound_sas_regionfile,
    write_bound_detector_geometry as _write_bound_detector_geometry,
)
from .path_safety import PathSafetyError, validate_output_namespace
from .producer_policy import ProducerEvidenceError, validate_durable_sas_producer
from .provenance import EventIdentity, EventIdentityError
from .publication import lexical_absolute_path
from .supporting_evidence import (
    SOURCE_ORIGINS,
    SupportingEvidenceError,
    canonical_sha256,
    supporting_evidence_sha256,
)

_EXPECTED_ESKY2DET_MODE = {
    "datastyle": "set",
    "calinfostyle": "set",
    "outunit": "det",
    "checkfov": "no",
    "witherrorcol": "no",
    "withouttab": "no",
}
_EXPECTED_ESKY2DET_KEYS = {*_EXPECTED_ESKY2DET_MODE, "intab", "calinfoset"}


def _event_path(
    *,
    event_file: str | Path | None,
    event_identity: EventIdentity | None,
) -> Path:
    if event_file is not None and event_identity is not None:
        raise TypeError("provide event_file or legacy event_identity, not both")
    if event_file is not None:
        return Path(event_file).expanduser().resolve()
    if event_identity is not None:
        if not isinstance(event_identity, EventIdentity):
            raise TypeError("event_identity must be EventIdentity")
        return Path(event_identity.path).expanduser().resolve()
    raise TypeError("event_file is required")


def _projection_event_binding(
    event_file: str | Path,
    result: ProjectionResult,
) -> EventIdentity:
    """Return the current canonical science identity iff it matches projection evidence."""
    try:
        identity, event_sha = read_science_calinfoset_snapshot(event_file)
    except (EventIdentityError, OSError) as exc:
        raise ArtifactMaterializationError(
            "cannot derive the canonical science-event binding for managed materialisation"
        ) from exc

    provenance = result.provenance
    if identity.identity_sha256 != provenance.event_identity_sha256:
        raise ArtifactMaterializationError(
            "current science-event identity does not match the event identity recorded by "
            "ProjectionResult"
        )
    if event_sha != provenance.event_file_sha256:
        raise ArtifactMaterializationError(
            "current event file bytes do not match the exact calinfoset used for projection"
        )
    return identity


def _artifact_event_binding(
    event_file: str | Path,
    artifact: BoundDetectorGeometryArtifact,
) -> EventIdentity:
    """Return the current canonical science identity iff it matches a bound artifact."""
    try:
        identity, event_sha = read_science_calinfoset_snapshot(event_file)
    except (EventIdentityError, OSError) as exc:
        raise ArtifactMaterializationError(
            "cannot derive the canonical science-event binding for managed reuse"
        ) from exc

    if identity.identity_sha256 != artifact.event_identity_sha256:
        raise ArtifactMaterializationError(
            "current science-event identity does not match the detector geometry binding"
        )
    if event_sha != artifact.event_file_sha256:
        raise ArtifactMaterializationError(
            "current event file bytes do not match the exact event used to create detector "
            "geometry"
        )
    return identity


def _validate_producer_for_managed_use(sas_producer: SasProducerIdentity) -> None:
    try:
        validate_durable_sas_producer(sas_producer)
    except ProducerEvidenceError as exc:
        raise ArtifactMaterializationError(
            f"managed detector geometry requires complete SAS producer evidence: {exc}"
        ) from exc


def _validate_result_context_identity(result: ProjectionResult) -> None:
    provenance = result.provenance
    try:
        validate_projection_context_identity(
            provenance.context_identity_sha256,
            calibration_identity_sha256=provenance.calibration_identity_sha256,
            producer_identity_sha256=provenance.producer_identity_sha256,
        )
    except ContextIdentityError as exc:
        raise ArtifactMaterializationError(
            f"managed detector geometry has invalid projection context identity: {exc}"
        ) from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactMaterializationError(
            f"managed detector geometry projection-evidence sidecar does not exist: {path}"
        )
    try:
        payload = loads_strict(bounded_text(path))
    except (OSError, StrictJsonError) as exc:
        raise ArtifactMaterializationError(
            f"managed detector geometry projection-evidence sidecar is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactMaterializationError(
            "projection-evidence sidecar must contain a JSON object"
        )
    return payload


def _evidence_path(
    geometry: str | Path,
    projection_evidence: str | Path | None,
) -> Path:
    path = Path(geometry).expanduser().resolve()
    if projection_evidence is not None:
        return Path(projection_evidence).expanduser().resolve()
    return path.with_suffix(".provenance.json")


def _validate_persisted_producer(
    artifact: BoundDetectorGeometryArtifact,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        evidence = Path(artifact.projection_evidence).expanduser().resolve()
        payload = _load_json_object(evidence)
    producer = payload.get("sas_producer")
    if not isinstance(producer, dict):
        raise ArtifactMaterializationError(
            "managed projection-evidence is missing the required SAS producer record"
        )
    snapshot = SimpleNamespace(
        esky2det_name=producer.get("esky2det_name"),
        esky2det_version=producer.get("esky2det_version"),
        esky2det_sha256=producer.get("esky2det_sha256"),
        sas_version=producer.get("sas_version"),
    )
    try:
        validate_durable_sas_producer(snapshot)
    except ProducerEvidenceError as exc:
        raise ArtifactMaterializationError(
            f"persisted managed artifact has incomplete SAS producer evidence: {exc}"
        ) from exc
    return payload


def _validate_persisted_context_identity(
    artifact: BoundDetectorGeometryArtifact,
    payload: dict[str, Any],
) -> None:
    projection = payload.get("projection")
    if not isinstance(projection, dict):
        raise ArtifactMaterializationError(
            "managed projection-evidence is missing the required projection record"
        )
    try:
        validate_projection_context_identity(
            projection.get("context_identity_sha256"),
            calibration_identity_sha256=artifact.calibration_identity_sha256,
            producer_identity_sha256=artifact.producer_identity_sha256,
        )
    except ContextIdentityError as exc:
        raise ArtifactMaterializationError(
            f"persisted managed artifact has invalid projection context identity: {exc}"
        ) from exc


def _canonical_historical_digest(value: object, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ArtifactMaterializationError(
            f"persisted managed artifact has invalid canonical {description} SHA256"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise ArtifactMaterializationError(
            f"persisted managed artifact has invalid canonical {description} SHA256"
        ) from exc
    return value


def _validate_historical_calibration_binding(
    artifact: BoundDetectorGeometryArtifact,
    payload: dict[str, Any],
) -> None:
    """Cross-check historical calibration evidence duplicated into FITS."""
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ArtifactMaterializationError(
            "managed projection-evidence is missing the required calibration record"
        )
    sidecar_cif = _canonical_historical_digest(
        calibration.get("cif_file_sha256"),
        description="exact CIF",
    )
    sidecar_calindex = _canonical_historical_digest(
        calibration.get("calindex_sha256"),
        description="CALINDEX",
    )
    replacements = calibration.get("replacements")
    if not isinstance(replacements, list):
        raise ArtifactMaterializationError(
            "managed projection-evidence calibration replacement record is invalid"
        )
    expected_replacement_count = len(replacements)

    geometry = Path(artifact.path).expanduser().resolve()
    try:
        with fits.open(geometry, memmap=False) as hdus:
            region_hdus = [hdu for hdu in hdus if hdu.name == "REGION"]
            if len(region_hdus) != 1:
                raise ArtifactMaterializationError(
                    "managed detector geometry must contain exactly one REGION extension"
                )
            header = region_hdus[0].header
            header_cif = _canonical_historical_digest(
                header.get("XMRGCIF"),
                description="FITS exact CIF",
            )
            header_calindex = _canonical_historical_digest(
                header.get("XMRGCAL"),
                description="FITS CALINDEX",
            )
            header_replacements = header.get("XMRGREP")
    except ArtifactMaterializationError:
        raise
    except OSError as exc:
        raise ArtifactMaterializationError(
            "cannot read detector geometry for historical calibration validation"
        ) from exc

    if header_cif != sidecar_cif:
        raise ArtifactMaterializationError(
            "managed detector geometry exact CIF evidence does not match its projection sidecar"
        )
    if header_calindex != sidecar_calindex:
        raise ArtifactMaterializationError(
            "managed detector geometry CALINDEX evidence does not match its projection sidecar"
        )
    if (
        not isinstance(header_replacements, int)
        or isinstance(header_replacements, bool)
        or header_replacements < 0
        or header_replacements != expected_replacement_count
    ):
        raise ArtifactMaterializationError(
            "managed detector geometry FITS replacement count does not match its projection sidecar"
        )


def _validate_converter_runtime_structure(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != "xmm-region-tool.converter-runtime/v1":
        raise ArtifactMaterializationError("managed converter/runtime evidence is invalid")
    package = value.get("package")
    python = value.get("python")
    dependencies = value.get("dependencies")
    if not isinstance(package, dict) or set(package) != {
        "distribution",
        "version",
        "source_sha256",
    }:
        raise ArtifactMaterializationError("managed converter/runtime package evidence is invalid")
    if package.get("distribution") != "xmm-region-tool" or not package.get("version"):
        raise ArtifactMaterializationError("managed converter/runtime package evidence is invalid")
    try:
        canonical_sha256(package.get("source_sha256"), description="converter source")
    except SupportingEvidenceError as exc:
        raise ArtifactMaterializationError(str(exc)) from exc
    if not isinstance(python, dict) or set(python) != {"implementation", "version"}:
        raise ArtifactMaterializationError("managed Python runtime evidence is invalid")
    if not python.get("implementation") or not python.get("version"):
        raise ArtifactMaterializationError("managed Python runtime evidence is invalid")
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "astropy",
        "numpy",
        "regions",
    }:
        raise ArtifactMaterializationError("managed dependency runtime evidence is invalid")
    if not all(isinstance(item, str) and item for item in dependencies.values()):
        raise ArtifactMaterializationError("managed dependency runtime evidence is invalid")
    return value


def _command_parameters(argv: object) -> dict[str, str]:
    if not isinstance(argv, list) or len(argv) != 1 + len(_EXPECTED_ESKY2DET_KEYS):
        raise ArtifactMaterializationError(
            "persisted esky2det command does not contain the canonical projection argument set"
        )
    if not isinstance(argv[0], str) or not argv[0]:
        raise ArtifactMaterializationError("persisted esky2det command has no executable")
    params: dict[str, str] = {}
    for token in argv[1:]:
        if not isinstance(token, str) or "=" not in token:
            raise ArtifactMaterializationError(
                "persisted esky2det command contains a non-canonical argument"
            )
        key, value = token.split("=", 1)
        if not key or key in params:
            raise ArtifactMaterializationError(
                "persisted esky2det command contains duplicate/empty parameters"
            )
        params[key] = value
    if set(params) != _EXPECTED_ESKY2DET_KEYS:
        raise ArtifactMaterializationError(
            "persisted esky2det command parameter names do not match the projection contract"
        )
    for key, expected in _EXPECTED_ESKY2DET_MODE.items():
        if params[key] != expected:
            raise ArtifactMaterializationError(
                f"persisted esky2det command {key}={params[key]!r} contradicts required {expected!r}"
            )
    if not params["intab"]:
        raise ArtifactMaterializationError(
            "persisted esky2det command has no temporary intab evidence"
        )
    if not params["calinfoset"]:
        raise ArtifactMaterializationError(
            "persisted esky2det command has no calinfoset evidence"
        )
    return params


def _validate_persisted_invocations(payload: dict[str, Any]) -> None:
    """Revalidate durable argv semantics without requiring historical paths to survive."""
    projection = payload.get("projection")
    producer = payload.get("sas_producer")
    if not isinstance(projection, dict) or not isinstance(producer, dict):
        raise ArtifactMaterializationError(
            "managed projection-evidence lacks invocation authority records"
        )
    primary = projection.get("command")
    history = projection.get("commands")
    if not isinstance(primary, list) or not isinstance(history, list) or not history:
        raise ArtifactMaterializationError(
            "persisted managed projection has no complete esky2det invocation history"
        )
    if primary != history[0]:
        raise ArtifactMaterializationError(
            "persisted projection primary command does not equal the first complete command-history entry"
        )
    producer_name = producer.get("esky2det_name")
    if not isinstance(producer_name, str) or not producer_name:
        raise ArtifactMaterializationError("persisted SAS producer has no esky2det task name")
    for argv in history:
        params = _command_parameters(argv)
        executable = Path(argv[0]).expanduser()
        if executable.name != producer_name:
            raise ArtifactMaterializationError(
                "persisted esky2det command task name contradicts the bound SAS producer"
            )
        # Promotion-time validation already hashes the exact calinfoset while it is
        # guaranteed available. Reload validates the immutable command structure;
        # path relocation later must not make a historical path semantic identity.
        if not params["calinfoset"]:
            raise ArtifactMaterializationError(
                "persisted esky2det command has no calinfoset evidence"
            )


def _validate_supporting_evidence(
    artifact: BoundDetectorGeometryArtifact,
    payload: dict[str, Any],
) -> None:
    record = payload.get("supporting_evidence")
    stored_digest = payload.get("supporting_evidence_sha256")
    if not isinstance(record, dict):
        raise ArtifactMaterializationError(
            "managed projection-evidence is missing required supporting evidence"
        )
    required_fields = {
        "schema",
        "projection_identity_sha256",
        "detector_geometry_sha256",
        "converter_runtime",
        "command",
        "commands",
        "relevant_environment",
        "refinement_diagnostics",
        "calibration",
        "sas_producer",
        "source_region_sha256",
        "source_region_origin",
    }
    if set(record) != required_fields:
        raise ArtifactMaterializationError("managed supporting evidence is not canonical")
    try:
        stored = canonical_sha256(stored_digest, description="supporting evidence")
        calculated = supporting_evidence_sha256(record)
        detector_sha = canonical_sha256(
            record.get("detector_geometry_sha256"),
            description="detector geometry",
        )
    except SupportingEvidenceError as exc:
        raise ArtifactMaterializationError(str(exc)) from exc
    if calculated != stored:
        raise ArtifactMaterializationError(
            "managed supporting evidence does not match its recorded SHA256"
        )
    if record.get("projection_identity_sha256") != artifact.projection_identity_sha256:
        raise ArtifactMaterializationError(
            "managed supporting evidence does not match the projection identity"
        )
    if detector_sha != artifact.geometry_sha256:
        raise ArtifactMaterializationError(
            "managed supporting evidence does not match detector geometry identity"
        )

    projection = payload.get("projection")
    calibration = payload.get("calibration")
    producer = payload.get("sas_producer")
    if not isinstance(projection, dict) or not isinstance(calibration, dict) or not isinstance(
        producer, dict
    ):
        raise ArtifactMaterializationError(
            "managed supporting evidence lacks authoritative records"
        )
    runtime = _validate_converter_runtime_structure(record.get("converter_runtime"))
    comparisons = {
        "converter runtime": (runtime, projection.get("converter_runtime")),
        "command": (record.get("command"), projection.get("command")),
        "commands": (record.get("commands"), projection.get("commands")),
        "relevant environment": (
            record.get("relevant_environment"),
            projection.get("relevant_environment"),
        ),
        "refinement diagnostics": (
            record.get("refinement_diagnostics"),
            projection.get("refinement_diagnostics"),
        ),
        "historical calibration": (record.get("calibration"), calibration),
        "SAS producer": (record.get("sas_producer"), producer),
    }
    for description, (bound, actual) in comparisons.items():
        if bound != actual:
            raise ArtifactMaterializationError(
                f"managed {description} evidence does not match its integrity-bound record"
            )

    source_sha = record.get("source_region_sha256")
    source_origin = record.get("source_region_origin")
    if source_sha is None:
        if source_origin is not None:
            raise ArtifactMaterializationError(
                "source-region provenance origin cannot be present without source evidence"
            )
    else:
        try:
            source_sha = canonical_sha256(source_sha, description="source-region evidence")
        except SupportingEvidenceError as exc:
            raise ArtifactMaterializationError(str(exc)) from exc
        if source_origin not in SOURCE_ORIGINS:
            raise ArtifactMaterializationError(
                "managed source-region evidence has an invalid authority origin"
            )

    geometry = Path(artifact.path).expanduser().resolve()
    try:
        with fits.open(geometry, memmap=False) as hdus:
            region_hdus = [hdu for hdu in hdus if hdu.name == "REGION"]
            if len(region_hdus) != 1:
                raise ArtifactMaterializationError(
                    "managed detector geometry must contain exactly one REGION extension"
                )
            header = region_hdus[0].header
            header_digest = _canonical_historical_digest(
                header.get("XMRGEVD"),
                description="FITS supporting-evidence",
            )
            header_source = header.get("XMRGSHA")
            if header_source is not None:
                header_source = _canonical_historical_digest(
                    str(header_source).strip(),
                    description="FITS source-region",
                )
            if header.get("ESKYVER") != producer.get("esky2det_version"):
                raise ArtifactMaterializationError(
                    "FITS esky2det producer evidence does not match its projection sidecar"
                )
            if header.get("SASVERS") != producer.get("sas_version"):
                raise ArtifactMaterializationError(
                    "FITS SAS release evidence does not match its projection sidecar"
                )
    except ArtifactMaterializationError:
        raise
    except OSError as exc:
        raise ArtifactMaterializationError(
            "cannot read detector geometry for supporting-evidence validation"
        ) from exc

    if header_digest != stored:
        raise ArtifactMaterializationError(
            "FITS supporting-evidence binding does not match the projection sidecar"
        )
    if header_source != source_sha:
        raise ArtifactMaterializationError(
            "FITS source-region evidence does not match its integrity-bound record"
        )


def _validate_persisted_managed_evidence(
    artifact: BoundDetectorGeometryArtifact,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = _load_json_object(Path(artifact.projection_evidence).expanduser().resolve())
    _validate_persisted_producer(artifact, payload)
    _validate_persisted_context_identity(artifact, payload)
    _validate_historical_calibration_binding(artifact, payload)
    _validate_persisted_invocations(payload)
    _validate_supporting_evidence(artifact, payload)
    return payload


def load_bound_detector_geometry(
    path: str | Path,
    *,
    projection_evidence: str | Path | None = None,
) -> BoundDetectorGeometryArtifact:
    """Load a persisted bound artifact and require complete semantic/evidentiary binding."""
    # Parse with the strict durable-JSON policy before the compatibility loader;
    # otherwise duplicate keys could be silently collapsed by ordinary json.loads.
    payload = _load_json_object(_evidence_path(path, projection_evidence))
    artifact = _load_bound_detector_geometry(
        path,
        projection_evidence=projection_evidence,
    )
    _validate_persisted_managed_evidence(artifact, payload=payload)
    return artifact


def write_bound_detector_geometry(
    path: str | Path,
    result: ProjectionResult,
    *,
    event_file: str | Path | None = None,
    event_identity: EventIdentity | None = None,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
    source_region_sha256: str | None = None,
    source_region_origin: str | None = None,
    max_components: int = 4096,
) -> BoundDetectorGeometryArtifact:
    """Write managed detector geometry after deriving event binding internally."""
    _validate_producer_for_managed_use(sas_producer)
    _validate_result_context_identity(result)
    current_path = _event_path(event_file=event_file, event_identity=event_identity)
    geometry = lexical_absolute_path(path)
    evidence = geometry.with_suffix(".provenance.json")
    try:
        validate_output_namespace(
            (("managed detector geometry", geometry), ("managed projection sidecar", evidence)),
            (("science event", current_path), ("active CIF", calibration_identity.cif_path)),
        )
    except PathSafetyError as exc:
        raise ArtifactMaterializationError(f"unsafe managed output namespace: {exc}") from exc
    current_identity = _projection_event_binding(current_path, result)
    artifact = _write_bound_detector_geometry(
        geometry,
        result,
        event_identity=current_identity,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
        source_region_sha256=source_region_sha256,
        source_region_origin=source_region_origin,
        max_components=max_components,
    )
    _validate_persisted_managed_evidence(artifact)
    return artifact


def materialize_bound_sas_regionfile(
    path: str | Path,
    artifact: BoundDetectorGeometryArtifact,
    *,
    event_file: str | Path | None = None,
    event_identity: EventIdentity | None = None,
    calibration_identity: CalibrationIdentity,
    sas_producer: SasProducerIdentity,
    expected_celestial_geometry_sha256: str,
    expected_projection_identity_sha256: str,
) -> SasRegionfileMaterialization:
    """Stage a managed SAS wrapper after re-deriving current event binding."""
    current_path = _event_path(event_file=event_file, event_identity=event_identity)
    wrapper = lexical_absolute_path(path)
    try:
        validate_output_namespace(
            (("managed SAS wrapper", wrapper),),
            (
                ("bound detector geometry", artifact.path),
                ("bound projection sidecar", artifact.projection_evidence),
                ("science event", current_path),
                ("active CIF", calibration_identity.cif_path),
            ),
        )
    except PathSafetyError as exc:
        raise ArtifactMaterializationError(f"unsafe managed staging namespace: {exc}") from exc
    _validate_producer_for_managed_use(sas_producer)
    _validate_persisted_managed_evidence(artifact)
    current_identity = _artifact_event_binding(current_path, artifact)
    return _materialize_bound_sas_regionfile(
        wrapper,
        artifact,
        event_identity=current_identity,
        calibration_identity=calibration_identity,
        sas_producer=sas_producer,
        expected_celestial_geometry_sha256=expected_celestial_geometry_sha256,
        expected_projection_identity_sha256=expected_projection_identity_sha256,
    )