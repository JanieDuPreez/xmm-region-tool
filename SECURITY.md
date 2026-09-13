# Security Policy

## Supported versions

Security fixes are provided for the current supported release line. During external testing, that means the latest `0.1.0` release candidate. Once `0.1.0` is released, support information will be updated here as later versions are published.

Historical development snapshots are not supported independently.

## Reporting a vulnerability

Please do **not** publish exploit details, credentials, private paths or science data in a normal GitHub issue.

Use this repository's **Security → Report a vulnerability** flow when available so the report is delivered privately to the maintainer. If the private form is unavailable, open a minimal issue asking the maintainer to establish a private security contact, but include no vulnerability details beyond the fact that private contact is required.

A useful private report should include the affected version, operating context, impact, reproduction steps that avoid sharing sensitive science products, and any suggested mitigation.

## Security and trust boundary

`xmm-region-tool` is a local scientific command-line/library tool, not a sandbox for mutually hostile users on the same machine.

The package deliberately fails closed for malformed or unsupported DS9, FITS, JSON and calibration state; checks input/output aliasing; uses symlink-aware atomic publication; and binds managed products to exact event/calibration/producer evidence. Hashes and before/after revalidation are intended to provide strong **scientific reproducibility, stale-state and ordinary corruption/replacement detection** under a trusted-filesystem assumption.

They are **not** a guarantee against an active same-machine attacker who can replace and restore pathname contents between those checks. During projection or validation, the trusted computing/filesystem boundary includes:

- the SAS installation and executables selected through the execution environment;
- the active CIF, CCF/calibration files and directories used to resolve them;
- the exact XMM science event products being processed;
- the directory namespace used for execution staging and final publication.

An actor with concurrent write access to those namespaces may be able to race pathname-based validation or transiently replace executable/data bytes. Fully defending that threat would require stronger operating-system isolation, private immutable staging or file-descriptor-oriented execution semantics and is outside the local-CLI threat model.

This limitation does not relax fail-closed malformed-input handling, path-collision checks, managed-evidence validation or symlink-safe publication.

## Generated files and privacy

Generated FITS REGION products, wrapper text files, provenance sidecars and workflow/batch manifests are scientific/runtime artifacts, not sanitised sharing formats. Provenance and manifests may intentionally record absolute local source, event, calibration, executable, staging or output paths so a run can be audited and reproduced.

Review those records before publication or attachment to public issues. Their integrity hashes prove neither confidentiality nor anonymity. The repository `.gitignore` reduces accidental `git add .` disclosure of common runtime products but is not a data-loss-prevention boundary.

Do not commit raw XMM event/calibration products, credentials, SAS environment secrets (if any), or private local-path evidence unless there is an explicit scientific reason and the contents have been reviewed.

## Dependency and CI supply chain

The CI and dependency-resolution policy is documented in `docs/repository_supply_chain_policy.md`. Release CI uses immutable action SHAs, a pinned `uv` bootstrap, a committed `uv.lock`, locked normal installs and read-only default GitHub-token permissions.

Repository rulesets/branch protection are administration state and must be verified in GitHub itself; a green workflow is not evidence that those settings exist.

## Distribution identity

The intended Python distribution is `xmm-region-tool`. Releases should use PyPI Trusted Publishing/OIDC from the canonical public repository and a dedicated publishing workflow/environment where practical. Long-lived upload tokens should not be used when Trusted Publishing is available.

## Repository-host security settings

For the canonical public repository, enable private vulnerability reporting and the applicable GitHub security features such as secret scanning and dependency/security monitoring. Protect `main` with the required CI checks, pull-request updates and force-push/deletion restrictions appropriate to the project.
