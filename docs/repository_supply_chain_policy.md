# Repository supply-chain policy

This document records the repository and CI supply-chain contract for `xmm-region-tool`.

## CI trust and permissions

The committed GitHub Actions workflow is intentionally read-only by default. Its token permission is `contents: read`, and checkout is configured with `persist-credentials: false`.

Third-party/reusable workflow actions must be referenced by reviewed immutable commit SHA, with the corresponding human-readable release tag kept in a comment. Moving major-version tags such as `@v7` are not accepted in release CI.

The current Python bootstrap tool is also pinned exactly (`uv==0.12.13`). A `uv` update is an explicit dependency-management change and must be reviewed like any other CI supply-chain change.

## Reproducible dependency resolution

`uv.lock` is committed. Ordinary CI installs and executes the project with `--locked`, so a stale or missing lockfile is a failure rather than an opportunity for CI to silently resolve a different dependency graph.

The minimum-dependencies job is deliberately different after the locked project/test environment has been created: it overlays the declared minimum scientific stack (`numpy==1.26.0`, `astropy==6.0.0`, `regions==0.9`) and then runs with `--no-sync`. An exact version assertion prevents `uv run` from silently restoring the lockfile versions before the compatibility suite executes.

When dependencies or Python support are changed:

1. update `pyproject.toml` intentionally;
2. regenerate and review `uv.lock` with the pinned/reviewed `uv` version;
3. keep the declared-floor job consistent with the public dependency metadata;
4. require exact-head CI to pass on Python 3.10, Python 3.13 and the declared-floor stack before treating the change as accepted.

## Repository settings outside version control

Branch protection/rulesets are repository-administration state, not workflow-file state. The canonical public repository's `main` branch should require pull-request updates and the current required CI checks, and should disallow force pushes and deletion except for an explicitly justified administrative recovery path.

Those settings must be verified in GitHub itself. A green workflow is not evidence that branch protection exists.

Private vulnerability reporting and applicable GitHub security/dependency features are likewise repository-host settings rather than properties of this file or CI configuration.

## Publishing identity

PyPI publication should use Trusted Publishing/OIDC bound to the canonical public repository and the reviewed release workflow/environment. Reusable long-lived upload credentials should not be used when Trusted Publishing is available.

## Security scope

These controls reduce dependency-resolution drift and workflow supply-chain risk. They are not a claim that GitHub Actions, PyPI, the SAS installation, the calibration tree, event files or a developer workstation are hostile-writer-safe. The package's local provenance/hash checks are primarily reproducibility and stale/corruption detection within the documented trusted-filesystem boundary.
