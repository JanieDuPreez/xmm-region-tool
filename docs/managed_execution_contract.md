# Managed execution staging contract

The supported top-level `xmm_region_tool.materialize_bound_sas_regionfile()` API closes the validation-to-SAS-consumption gap by separating the **managed/cache artifact** from the **execution-owned geometry**.

## Contract

A persisted `BoundDetectorGeometryArtifact` may live in a cache or other reusable location. Before it is made consumable by SAS, the top-level managed path:

1. revalidates the persisted FITS geometry and projection-evidence sidecar under the full managed provenance contract;
2. re-derives and checks the current science-event binding and requested celestial/projection identities;
3. copies the validated geometry bytes to a sibling execution path derived from the requested wrapper path;
4. SHA-256 hashes that staged copy and requires exact equality with `artifact.file_sha256`;
5. writes the SAS wrapper so that it references the **execution copy**, never the reusable/cache artifact path; and
6. atomically publishes the execution geometry and wrapper together using the package's shared symlink-safe transactional publication primitive.

The returned `SasRegionfileMaterialization.geometry_path` is therefore the file SAS is expected to consume. After successful materialisation, later replacement or mutation of `artifact.path` cannot change the bytes referenced by the emitted wrapper.

## Ownership after materialisation

The caller owns the returned wrapper and execution geometry as one execution unit. They must keep **that returned execution geometry path** present and unchanged until the relevant SAS/ESAS task has consumed the wrapper. The reusable managed/cache artifact may be relocated, replaced or removed after successful staging without changing the already-staged execution bytes.

This contract does not claim protection against an actor that can modify the execution-owned directory after `materialize_bound_sas_regionfile()` returns. Parent-directory trust remains an operating/deployment boundary. In a shared environment, stage and execute inside a directory whose namespace is controlled by the invoking process/user.

## Filesystem safety

Namespace checks use resolved/same-file semantics to reject aliases with protected event, CIF, cache geometry and sidecar inputs. Publication itself retains the lexical destination path and uses sibling staging plus atomic `os.replace()`, so a pre-existing or raced-in destination symlink is replaced as a directory entry rather than followed to its target.

If staging sees geometry bytes that do not hash to the already validated `artifact.file_sha256`, materialisation fails before the normal execution wrapper/geometry pair is published. Existing valid execution outputs are preserved on a failed rerun by the same rollback mechanism used for other supported products.

## Non-goals

This execution-copy contract does not change detector geometry, projection identity, FITS REGION semantics or SAS selectlib syntax. It also does not attempt a full `openat`/directory-FD sandbox against an adversary controlling the parent directory. The security/integrity guarantee is the transition from a validated reusable artifact to an independently verified execution copy in a caller-controlled execution namespace.
