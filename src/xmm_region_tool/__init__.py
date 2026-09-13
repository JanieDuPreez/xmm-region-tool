"""XMM-Newton detector-region conversion utilities.

The top-level exports are the supported fail-closed library boundary for
standalone callers and managed XMM provider integrations. Deliberately low-level
projection/serialization helpers remain available from their implementation
modules for debugging, but are not part of this managed-safe surface.
"""

import subprocess as _subprocess

from . import calibration as _calibration
from . import sas as _sas
from . import sas_validation as _sas_validation
from .artifacts import ArtifactMaterializationError, SasRegionfileMaterialization
from .constructors import (
    box_annulus_selection,
    box_selection,
    circle_selection,
    circular_annulus_selection,
    ellipse_selection,
    ellipse_local_sector_annulus_selection,
    elliptical_annulus_selection,
    elliptical_sector_annulus_selection,
    polygon_selection,
    sector_annulus_selection,
)
from .context_suitability import ObservationRecord
from .execution import (
    ProjectionProvenance,
    ProjectionResult,
    ProjectionRule,
    SasProjectionContextError,
)
from .geometry import load_ds9_selection, selection_from_sky_regions
from .managed import BoundDetectorGeometryArtifact
from .managed_execution import materialize_bound_sas_regionfile
from .model import (
    CelestialBoundary,
    CelestialRegion,
    CelestialSelection,
    DetectorBoundary,
    DetectorRegion,
    DetectorSelection,
    SelectionGeometryError,
)
from .public_batch import BatchConversionError, BatchConversionResult, convert_selection_batch
from .public_context import SasProjectionContext
from .public_managed import load_bound_detector_geometry, write_bound_detector_geometry
from .sas import SasConversionError, project_selection
from .version import distribution_version

# Historical tests and downstream debugging code monkeypatch these module-local
# subprocess seams. Keep them as aliases to the one stdlib module object used by
# subprocess_capture.run_bounded(); production calls remain bounded/file-backed.
_calibration.subprocess = _subprocess
_sas.subprocess = _subprocess
_sas_validation.subprocess = _subprocess

__all__ = [
    "ArtifactMaterializationError",
    "BatchConversionError",
    "BatchConversionResult",
    "BoundDetectorGeometryArtifact",
    "CelestialBoundary",
    "CelestialRegion",
    "CelestialSelection",
    "DetectorBoundary",
    "DetectorRegion",
    "DetectorSelection",
    "ObservationRecord",
    "ProjectionProvenance",
    "ProjectionResult",
    "ProjectionRule",
    "SasConversionError",
    "SasProjectionContext",
    "SasProjectionContextError",
    "SasRegionfileMaterialization",
    "SelectionGeometryError",
    "__version__",
    "box_annulus_selection",
    "box_selection",
    "circle_selection",
    "circular_annulus_selection",
    "convert_selection_batch",
    "ellipse_local_sector_annulus_selection",
    "ellipse_selection",
    "elliptical_annulus_selection",
    "elliptical_sector_annulus_selection",
    "load_bound_detector_geometry",
    "load_ds9_selection",
    "materialize_bound_sas_regionfile",
    "polygon_selection",
    "project_selection",
    "sector_annulus_selection",
    "selection_from_sky_regions",
    "write_bound_detector_geometry",
]
__version__ = distribution_version()
